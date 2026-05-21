"""
ODT adapter — pure-stdlib (``zipfile`` + ``xml.etree.ElementTree``).

An .odt is a zip with a layout fixed by the OpenDocument spec:

* ``mimetype``        — sanity check; should be
                        ``application/vnd.oasis.opendocument.text``.
* ``content.xml``     — the body. Paragraphs live in ``text:p``, headings
                        in ``text:h`` (with a ``text:outline-level``
                        attribute giving the depth — 1 is the most
                        important).
* ``meta.xml``        — Dublin Core metadata (``dc:title``, ``dc:creator``,
                        ``dc:subject``, etc.).

We only read the two XML files; styles, images and embedded objects are
ignored. This avoids the ``odfpy`` dep (which is pure-Python but adds a
non-trivial wheel for tiny gain) and stays Termux-safe.

Headings start new sections, in the same way DOCX heading styles do. The
first outline-level=1 heading is also used as a title fallback when
``meta.xml`` doesn't carry one.
"""
from __future__ import annotations

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
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_ODT_BYTES = 25_000_000

_NS = {
    "office":  "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text":    "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "meta":    "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
    "dc":      "http://purl.org/dc/elements/1.1/",
}


def _element_text(node: ET.Element) -> str:
    """Concatenate every text node under ``node``.

    ODT scatters runs across ``text:span`` children; ``itertext`` does the
    right thing without needing to know each style. Tabs and line breaks
    become whitespace so the rendered text mirrors what a reader sees.
    """
    parts: list[str] = []
    for child in node.iter():
        tag = child.tag.split("}", 1)[-1]
        if tag in ("tab",):
            parts.append("\t")
        elif tag in ("line-break", "s"):
            # text:line-break is an inline line feed; text:s is a single
            # space placeholder used inside spans.
            parts.append("\n" if tag == "line-break" else " ")
        elif child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_meta(zf: zipfile.ZipFile) -> dict[str, str]:
    """Return Dublin Core metadata from ``meta.xml`` (or {})."""
    try:
        with zf.open("meta.xml") as fh:
            tree = ET.parse(fh)
    except (KeyError, ET.ParseError):
        return {}

    root = tree.getroot()
    out: dict[str, str] = {}

    def grab(local: str, ns_key: str, out_key: str) -> None:
        # ``dc:title`` and friends sit under <office:meta>; xpath with the
        # namespace map keeps the lookup robust against namespace prefix
        # variation in the source file.
        node = root.find(f".//{{{_NS[ns_key]}}}{local}")
        if node is not None and node.text:
            out[out_key] = node.text.strip()

    grab("title",       "dc", "title")
    grab("creator",     "dc", "author")
    grab("subject",     "dc", "subject")
    grab("description", "dc", "description")
    grab("language",    "dc", "language")
    return out


def _parse_content(zf: zipfile.ZipFile) -> tuple[list[ExtractedSection], Optional[str]]:
    """Walk ``content.xml`` and build sections keyed on text:h headings.

    Returns ``(sections, first_h1_text)`` — the first outline-level=1
    heading is captured separately for use as a title fallback.
    """
    try:
        with zf.open("content.xml") as fh:
            tree = ET.parse(fh)
    except (KeyError, ET.ParseError) as e:
        raise ValueError(f"cannot read content.xml: {e}") from e

    body = tree.getroot().find(".//office:body/office:text", _NS)
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

    # Only iterate the direct children of <office:text> so nested lists
    # don't double-count. text:list contents are reachable via itertext
    # when the paragraph happens to be inside a list item — but at the
    # top level we want one paragraph = one line.
    for node in list(body):
        tag = node.tag.split("}", 1)[-1]
        if tag == "h":
            flush()
            text = _element_text(node).strip()
            current_heading = text or None
            level = node.get(f"{{{_NS['text']}}}outline-level", "1")
            if first_h1 is None and level == "1" and text:
                first_h1 = text
            continue
        if tag in ("p", "list"):
            text = _element_text(node).strip()
            if text:
                current_lines.append(text)

    flush()
    return sections, first_h1


class OdtAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("odt",)
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
        if size > _MAX_ODT_BYTES:
            logger.warning(
                "odt_too_large", path=str(file_path), size=size, max=_MAX_ODT_BYTES,
            )
            raise ValueError(
                f"odt file exceeds {_MAX_ODT_BYTES} bytes: {file_path} ({size} bytes)"
            )

        try:
            zf = zipfile.ZipFile(file_path)
        except zipfile.BadZipFile as e:
            raise ValueError(f"odt is not a valid zip: {file_path}: {e}") from e

        try:
            metadata = _parse_meta(zf)
            sections, first_h1 = _parse_content(zf)
        finally:
            zf.close()

        title = metadata.get("title") or first_h1 or file_path.stem
        text = "\n\n".join(s.text for s in sections)

        return ExtractedDocument(
            source_path=file_path,
            format="odt",
            title=title,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata=metadata,
            sections=sections,
            size_bytes=size,
        )


register(OdtAdapter())
