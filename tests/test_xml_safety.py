"""XML-bomb / entity-expansion hardening for the zip-XML adapters.

Covers audit finding M1: stdlib ``xml.etree`` expands internal entities,
so a tiny crafted ``.docx`` / ``.odt`` / ``.epub`` is a "billion laughs"
memory DoS. These tests assert the adapters now *refuse* such a file
(rather than expanding it) and that the shared ``safe_parse_xml`` helper
rejects DTD/entity/external constructs while still parsing benign XML.
"""
from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile

import pytest

from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.docx import DocxAdapter
from grimore.ingest.adapters.epub import EpubAdapter
from grimore.ingest.adapters.odt import OdtAdapter
from grimore.ingest.adapters import safexml
from grimore.ingest.adapters.safexml import UnsafeXmlError, safe_parse_xml


# A classic billion-laughs payload. Five entity levels = 10^5 expansion;
# a regressed (unprotected) parser would still expand it, so the
# ``pytest.raises`` below is the real regression guard — we never let it
# get far enough to matter.
_BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY a0 "AAAAAAAAAA">
 <!ENTITY a1 "&a0;&a0;&a0;&a0;&a0;&a0;&a0;&a0;&a0;&a0;">
 <!ENTITY a2 "&a1;&a1;&a1;&a1;&a1;&a1;&a1;&a1;&a1;&a1;">
 <!ENTITY a3 "&a2;&a2;&a2;&a2;&a2;&a2;&a2;&a2;&a2;&a2;">
 <!ENTITY a4 "&a3;&a3;&a3;&a3;&a3;&a3;&a3;&a3;&a3;&a3;">
]>
<lolz>&a4;</lolz>"""

# An external-entity reference (XXE file-read / SSRF vector).
_EXTERNAL_ENTITY = b"""<?xml version="1.0"?>
<!DOCTYPE r [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
<r>&xxe;</r>"""

_VALID_CONTAINER = (
    b'<?xml version="1.0"?>'
    b'<container version="1.0" '
    b'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    b'<rootfiles><rootfile full-path="content.opf" '
    b'media-type="application/oebps-package+xml"/></rootfiles>'
    b"</container>"
)


def _make_zip(tmp_path, name, members):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        for arc, data in members.items():
            zf.writestr(arc, data)
    return p


class TestXmlBombAdapters:
    """Each zip-XML adapter must reject a bomb, fast, as a ValueError."""

    def test_docx_rejects_billion_laughs(self, tmp_path):
        p = _make_zip(tmp_path, "bomb.docx", {"word/document.xml": _BILLION_LAUGHS})
        t = time.monotonic()
        with pytest.raises(ValueError):
            DocxAdapter().extract(p, options=AdapterOptions())
        assert time.monotonic() - t < 2.0  # no expansion happened

    def test_docx_rejects_bomb_in_core_props(self, tmp_path):
        # A bomb hidden only in the metadata member is still refused (the
        # whole file is skipped) rather than silently expanded.
        body = (
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/'
            b'wordprocessingml/2006/main"><w:body/></w:document>'
        )
        p = _make_zip(tmp_path, "metabomb.docx", {
            "word/document.xml": body,
            "docProps/core.xml": _BILLION_LAUGHS,
        })
        with pytest.raises(ValueError):
            DocxAdapter().extract(p, options=AdapterOptions())

    def test_odt_rejects_billion_laughs(self, tmp_path):
        p = _make_zip(tmp_path, "bomb.odt", {"content.xml": _BILLION_LAUGHS})
        t = time.monotonic()
        with pytest.raises(ValueError):
            OdtAdapter().extract(p, options=AdapterOptions())
        assert time.monotonic() - t < 2.0

    def test_epub_rejects_bomb_in_container(self, tmp_path):
        p = _make_zip(tmp_path, "bomb.epub", {
            "META-INF/container.xml": _BILLION_LAUGHS,
        })
        with pytest.raises(ValueError):
            EpubAdapter().extract(p, options=AdapterOptions())

    def test_epub_rejects_bomb_in_opf(self, tmp_path):
        # Valid container.xml that points at a bomb OPF manifest.
        p = _make_zip(tmp_path, "opfbomb.epub", {
            "META-INF/container.xml": _VALID_CONTAINER,
            "content.opf": _BILLION_LAUGHS,
        })
        with pytest.raises(ValueError):
            EpubAdapter().extract(p, options=AdapterOptions())


class TestSafeParseXml:
    """Unit-level contract for the shared helper."""

    def test_parses_benign_xml_from_bytes(self):
        root = safe_parse_xml(b'<r xmlns:w="urn:x"><w:p>hi</w:p></r>')
        assert root.tag == "r"
        assert root.find("{urn:x}p").text == "hi"

    def test_parses_benign_xml_from_file_object(self):
        root = safe_parse_xml(io.BytesIO(b"<r><c>x</c></r>"))
        assert root.find("c").text == "x"

    def test_rejects_billion_laughs(self):
        with pytest.raises(UnsafeXmlError):
            safe_parse_xml(_BILLION_LAUGHS, what="bomb")

    def test_rejects_external_entity(self):
        with pytest.raises(UnsafeXmlError):
            safe_parse_xml(_EXTERNAL_ENTITY, what="xxe")

    def test_rejects_bare_doctype(self):
        # No entities, but a DOCTYPE is still refused (forbid_dtd / marker).
        with pytest.raises(UnsafeXmlError):
            safe_parse_xml(b'<?xml version="1.0"?><!DOCTYPE html><html/>')

    def test_unsafe_error_is_valueerror(self):
        # So adapters' "raise ValueError → scan loop logs a skip" holds.
        assert issubclass(UnsafeXmlError, ValueError)

    def test_malformed_xml_raises_parseerror(self):
        # Merely broken (non-malicious) XML keeps the type adapters catch.
        with pytest.raises(ET.ParseError):
            safe_parse_xml(b"<a><b></a>")


class TestZipBombCap:
    """Decompressed-size ceiling on archive members.

    The adapters cap the archive size on disk, but deflate reaches
    ~1000:1 — a few-MB member inside a size-legal file could inflate to
    gigabytes in memory. ``read_bounded`` refuses past the cap; the cap
    is monkeypatched small here so the tests stay fast and light.
    """

    def test_read_bounded_rejects_oversized_stream(self):
        with pytest.raises(UnsafeXmlError):
            safexml.read_bounded(io.BytesIO(b"A" * 4096), max_bytes=1024)

    def test_read_bounded_passes_small_stream(self):
        assert safexml.read_bounded(io.BytesIO(b"ok"), max_bytes=1024) == b"ok"

    def test_default_cap_is_read_at_call_time(self, monkeypatch):
        # max_bytes=None must pick up the *current* module constant, so
        # both this suite and any operator override take effect.
        monkeypatch.setattr(safexml, "MAX_MEMBER_BYTES", 8)
        with pytest.raises(UnsafeXmlError):
            safexml.read_bounded(io.BytesIO(b"123456789"))

    def test_docx_rejects_inflating_member(self, tmp_path, monkeypatch):
        monkeypatch.setattr(safexml, "MAX_MEMBER_BYTES", 1024)
        # Highly compressible: small on disk, over the cap when inflated.
        big = (
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/'
            b'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
            + b"A" * 100_000
            + b"</w:t></w:r></w:p></w:body></w:document>"
        )
        p = _make_zip(tmp_path, "inflate.docx", {"word/document.xml": big})
        with pytest.raises(ValueError):
            DocxAdapter().extract(p, options=AdapterOptions())

    def test_odt_rejects_inflating_member(self, tmp_path, monkeypatch):
        monkeypatch.setattr(safexml, "MAX_MEMBER_BYTES", 1024)
        big = (
            b'<office:document-content '
            b'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            b'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
            b"<office:body><office:text><text:p>"
            + b"A" * 100_000
            + b"</text:p></office:text></office:body></office:document-content>"
        )
        p = _make_zip(tmp_path, "inflate.odt", {"content.xml": big})
        with pytest.raises(ValueError):
            OdtAdapter().extract(p, options=AdapterOptions())

    def test_epub_rejects_inflating_chapter(self, tmp_path, monkeypatch):
        # Chapters bypass safe_parse_xml (they're XHTML for bs4), so the
        # bounded read must guard that path too.
        monkeypatch.setattr(safexml, "MAX_MEMBER_BYTES", 1024)
        opf = (
            b'<?xml version="1.0"?>'
            b'<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            b'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:title>T</dc:title></metadata>"
            b'<manifest><item id="c1" href="ch1.xhtml" '
            b'media-type="application/xhtml+xml"/></manifest>'
            b'<spine><itemref idref="c1"/></spine>'
            b"</package>"
        )
        chapter = b"<html><body><p>" + b"A" * 100_000 + b"</p></body></html>"
        p = _make_zip(tmp_path, "inflate.epub", {
            "META-INF/container.xml": _VALID_CONTAINER,
            "content.opf": opf,
            "ch1.xhtml": chapter,
        })
        with pytest.raises(ValueError):
            EpubAdapter().extract(p, options=AdapterOptions())


class TestBenignStillParses:
    """The hardening must not break normal documents."""

    def test_benign_docx_extracts(self, tmp_path):
        document_xml = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main"><w:body>'
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            "<w:r><w:t>Title</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Body text.</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        p = _make_zip(tmp_path, "ok.docx", {"word/document.xml": document_xml})
        doc = DocxAdapter().extract(p, options=AdapterOptions())
        assert "Body text." in doc.text
        assert doc.title == "Title"
