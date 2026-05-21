"""
EPUB adapter — pure-stdlib (``zipfile`` + ``xml.etree``) plus the
``beautifulsoup4`` already pulled in by the HTML adapter for chapter
XHTML extraction.

An ``.epub`` is a zip with a fixed entry-point layout (IDPF Open
Container Format):

* ``mimetype``                       — sanity check; should be
                                       ``application/epub+zip``.
* ``META-INF/container.xml``         — points to the OPF manifest.
* ``<opf>``                          — Dublin Core metadata in
                                       ``<metadata>``, files in
                                       ``<manifest>``, reading order in
                                       ``<spine>``.
* ``<spine items>`` (XHTML)          — the actual chapters.

We honour the spine order so the section list mirrors what a reader
sees. Each spine item becomes one :class:`ExtractedSection` keyed on
the chapter heading (first ``<h1>``-ish element in the XHTML, falling
back to the manifest id).
"""
from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import ClassVar, Optional, Union

from bs4 import BeautifulSoup

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
    ExtractedSection,
)
from grimore.ingest.adapters.html import _PARSER as _HTML_PARSER, _NOISE_TAGS
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_EPUB_BYTES = 50_000_000

_CONTAINER_PATH = "META-INF/container.xml"
_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf":       "http://www.idpf.org/2007/opf",
    "dc":        "http://purl.org/dc/elements/1.1/",
}


def _locate_opf(zf: zipfile.ZipFile) -> str:
    """Return the in-archive path of the OPF manifest."""
    try:
        with zf.open(_CONTAINER_PATH) as fh:
            tree = ET.parse(fh)
    except (KeyError, ET.ParseError) as e:
        raise ValueError(f"epub missing or unreadable {_CONTAINER_PATH}: {e}") from e
    root = tree.getroot()
    rootfile = root.find(".//container:rootfile", _NS)
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("epub container.xml has no rootfile path")
    return rootfile.get("full-path")


def _parse_opf(zf: zipfile.ZipFile, opf_path: str) -> tuple[dict[str, str], list[str]]:
    """Return ``(metadata, ordered_spine_paths_in_archive)``.

    Spine paths are resolved against the OPF's directory so absolute
    archive paths reach the right XHTML when we open them next.
    """
    try:
        with zf.open(opf_path) as fh:
            tree = ET.parse(fh)
    except (KeyError, ET.ParseError) as e:
        raise ValueError(f"unreadable opf {opf_path}: {e}") from e

    root = tree.getroot()
    metadata: dict[str, str] = {}

    def grab(local: str, key: str) -> None:
        node = root.find(f".//{{{_NS['dc']}}}{local}")
        if node is not None and node.text:
            metadata[key] = node.text.strip()

    grab("title",       "title")
    grab("creator",     "author")
    grab("subject",     "subject")
    grab("description", "description")
    grab("language",    "language")

    # Manifest: id → href
    manifest: dict[str, str] = {}
    for item in root.findall(".//opf:manifest/opf:item", _NS):
        idref = item.get("id")
        href = item.get("href")
        if idref and href:
            manifest[idref] = href

    # Spine: ordered list of manifest idrefs
    opf_dir = posixpath.dirname(opf_path)
    spine_paths: list[str] = []
    for itemref in root.findall(".//opf:spine/opf:itemref", _NS):
        idref = itemref.get("idref")
        if not idref:
            continue
        href = manifest.get(idref)
        if not href:
            continue
        # Resolve href relative to the OPF's location inside the archive.
        resolved = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
        spine_paths.append(resolved)

    return metadata, spine_paths


def _extract_chapter(zf: zipfile.ZipFile, archive_path: str) -> tuple[Optional[str], str]:
    """Return ``(heading, body_text)`` for one XHTML spine item."""
    try:
        with zf.open(archive_path) as fh:
            raw = fh.read()
    except KeyError:
        logger.warning("epub_chapter_missing", path=archive_path)
        return None, ""

    soup = BeautifulSoup(raw, _HTML_PARSER)
    for noise in _NOISE_TAGS:
        for node in soup.find_all(noise):
            node.decompose()

    heading: Optional[str] = None
    for level in ("h1", "h2", "h3"):
        node = soup.find(level)
        if node and node.get_text(strip=True):
            heading = node.get_text(strip=True)
            break

    body = soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    return heading, text


class EpubAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("epub",)
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
        if size > _MAX_EPUB_BYTES:
            logger.warning(
                "epub_too_large", path=str(file_path), size=size, max=_MAX_EPUB_BYTES,
            )
            raise ValueError(
                f"epub file exceeds {_MAX_EPUB_BYTES} bytes: {file_path} ({size} bytes)"
            )

        try:
            zf = zipfile.ZipFile(file_path)
        except zipfile.BadZipFile as e:
            raise ValueError(f"epub is not a valid zip: {file_path}: {e}") from e

        try:
            opf_path = _locate_opf(zf)
            metadata, spine = _parse_opf(zf, opf_path)
            sections: list[ExtractedSection] = []
            first_heading: Optional[str] = None
            for order, archive_path in enumerate(spine):
                heading, text = _extract_chapter(zf, archive_path)
                text = text.strip()
                if not text:
                    continue
                if first_heading is None and heading:
                    first_heading = heading
                sections.append(ExtractedSection(
                    text=text, page=None, heading=heading, order=order,
                ))
        finally:
            zf.close()

        title = (
            metadata.get("title")
            or first_heading
            or file_path.stem
        )
        body = "\n\n".join(s.text for s in sections)

        return ExtractedDocument(
            source_path=file_path,
            format="epub",
            title=title,
            text=body,
            content_hash=calculate_content_hash(body),
            file_hash=sha256_file(file_path),
            metadata=metadata,
            sections=sections,
            size_bytes=size,
        )


register(EpubAdapter())
