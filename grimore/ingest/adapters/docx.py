"""
DOCX adapter — pure stdlib (``zipfile`` + ``xml.etree.ElementTree``).

Avoiding ``python-docx`` (and therefore ``lxml``) keeps the install
footprint zero-native-dep and sidesteps the Termux dynamic-linker
namespace issue that prevents lxml's .so from dlopening inside a venv.
Users who want richer DOCX handling can install the ``docx-rich`` extra
and pin a fork of this adapter — this one covers the 90% case.

The .docx container is a well-specified zip:

* ``word/document.xml``    — body content; paragraphs grouped under
                             headings via the ``w:pStyle`` attribute.
* ``docProps/core.xml``    — Dublin Core metadata (title, creator,
                             subject, description, keywords).

We only read the two files above; everything else (images, embedded
fonts, styles.xml) is ignored.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import ClassVar, Optional, Union

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
    ExtractedSection,
)
from grimore.ingest.adapters.registry import register
from grimore.ingest.adapters.safexml import safe_parse_xml
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_DOCX_BYTES = 25_000_000

# Office Open XML namespaces. ElementTree's XPath uses ``{uri}localname``
# notation so we predeclare the prefixes we care about.
_NS = {
    "w":  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}

# A paragraph styled "Heading 1" (or "Heading1", "Heading 2", etc.)
# starts a new section. Match any digit suffix so localisations like
# "Heading 1", "heading1", "Titre1" — wait, no, Word stores English
# style ids regardless of UI language, so this catches every case.
_HEADING_STYLE_RE = re.compile(r"^Heading\s*(\d+)$", re.IGNORECASE)


def _paragraph_text(p: ET.Element) -> str:
    """Concatenate every ``<w:t>`` run inside a paragraph.

    ``<w:tab/>`` and ``<w:br/>`` are flattened to whitespace so the
    rendered text mirrors what a reader would see.
    """
    parts: list[str] = []
    for child in p.iter():
        tag = child.tag.split("}", 1)[-1]
        if tag == "t" and child.text:
            parts.append(child.text)
        elif tag in ("tab",):
            parts.append("\t")
        elif tag in ("br", "cr"):
            parts.append("\n")
    return "".join(parts)


def _paragraph_style(p: ET.Element) -> Optional[str]:
    """Return the ``w:val`` of ``w:pStyle`` for ``p``, or None."""
    pPr = p.find("w:pPr", _NS)
    if pPr is None:
        return None
    pStyle = pPr.find("w:pStyle", _NS)
    if pStyle is None:
        return None
    return pStyle.get(f"{{{_NS['w']}}}val")


def _parse_core_props(zf: zipfile.ZipFile) -> dict[str, str]:
    """Return Dublin Core metadata from ``docProps/core.xml`` (or {})."""
    try:
        with zf.open("docProps/core.xml") as fh:
            root = safe_parse_xml(fh, what="docProps/core.xml")
    except (KeyError, ET.ParseError):
        return {}

    out: dict[str, str] = {}

    def grab(local: str, ns_key: str, out_key: str) -> None:
        node = root.find(f"{{{_NS[ns_key]}}}{local}")
        if node is not None and node.text:
            out[out_key] = node.text.strip()

    grab("title",       "dc", "title")
    grab("creator",     "dc", "author")
    grab("subject",     "dc", "subject")
    grab("description", "dc", "description")
    grab("keywords",    "cp", "keywords")
    return out


def _parse_body(zf: zipfile.ZipFile) -> tuple[list[ExtractedSection], Optional[str]]:
    """Walk ``word/document.xml`` and break it into sections + first-h1 hint.

    Returns ``(sections, first_h1_text)`` so the caller can use the first
    Heading 1 as a title fallback when core props don't carry one.
    """
    try:
        with zf.open("word/document.xml") as fh:
            root = safe_parse_xml(fh, what="word/document.xml")
    except (KeyError, ET.ParseError) as e:
        raise ValueError(f"cannot read word/document.xml: {e}") from e

    body = root.find("w:body", _NS)
    if body is None:
        return [], None

    sections: list[ExtractedSection] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []
    order = 0
    first_h1: Optional[str] = None

    def flush() -> None:
        nonlocal order
        body_text = "\n\n".join(line for line in current_lines if line)
        if body_text:
            sections.append(ExtractedSection(
                text=body_text, heading=current_heading, order=order,
            ))
            order += 1
        current_lines.clear()

    for p in body.findall("w:p", _NS):
        text = _paragraph_text(p).strip()
        style = _paragraph_style(p)
        heading_match = _HEADING_STYLE_RE.match(style) if style else None
        if heading_match:
            flush()
            current_heading = text or None
            if first_h1 is None and heading_match.group(1) == "1" and text:
                first_h1 = text
            continue
        if text:
            current_lines.append(text)

    flush()
    return sections, first_h1


class DocxAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("docx",)
    binary: ClassVar[bool] = True
    mutable_frontmatter: ClassVar[bool] = False

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        file_path = Path(path)
        if options.vault_root is not None:
            SecurityGuard.resolve_within_vault(file_path, options.vault_root)

        try:
            stat = file_path.stat()
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        size = stat.st_size
        if size > _MAX_DOCX_BYTES:
            logger.warning(
                "docx_too_large", path=str(file_path), size=size, max=_MAX_DOCX_BYTES,
            )
            raise ValueError(
                f"docx file exceeds {_MAX_DOCX_BYTES} bytes: {file_path} ({size} bytes)"
            )

        try:
            zf = zipfile.ZipFile(file_path)
        except zipfile.BadZipFile as e:
            raise ValueError(f"docx is not a valid zip: {file_path}: {e}") from e

        try:
            metadata = _parse_core_props(zf)
            sections, first_h1 = _parse_body(zf)
        finally:
            zf.close()

        title = metadata.get("title") or first_h1 or file_path.stem
        text = "\n\n".join(s.text for s in sections)

        return ExtractedDocument(
            source_path=file_path,
            format="docx",
            title=title,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata=metadata,
            sections=sections,
            size_bytes=size,
        )


register(DocxAdapter())
