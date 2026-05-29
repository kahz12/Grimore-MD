"""
Per-adapter unit tests for the RTF + ODT + legacy .doc formats and
the engine-selection plumbing on the PDF adapter.

Fixtures are built in-process — no large binaries enter the repo. ODT is
a zip with XML files inside, so we hand-craft one the same way we do for
DOCX. RTF is a tiny escape-coded text format we can write directly. The
.doc adapter is exercised by mocking ``shutil.which`` and
``subprocess.run`` so the test suite passes on machines without antiword
installed (the common case).
"""
from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.doc import DocAdapter, antiword_available
from grimore.ingest.adapters.odt import OdtAdapter, _MAX_ODT_BYTES
from grimore.ingest.adapters.pdf import PdfAdapter, _SUPPORTED_ENGINES
from grimore.ingest.adapters.rtf import RtfAdapter, _MAX_RTF_BYTES


# ── RTF ────────────────────────────────────────────────────────────────────


_MINIMAL_RTF = (
    r"{\rtf1\ansi\ansicpg1252\deff0\nouicompat\deflang1033"
    r"{\fonttbl{\f0\fnil\fcharset0 Calibri;}}"
    r"{\*\generator Grimore test;}"
    r"\viewkind4\uc1\pard\sa200\sl276\slmult1\f0\fs22\lang9"
    r" hello world\par"
    r" second paragraph\par"
    r"}"
)


class TestRtfAdapter:
    def test_happy_path(self, tmp_path):
        f = tmp_path / "note.rtf"
        f.write_text(_MINIMAL_RTF)
        doc = RtfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "rtf"
        assert doc.title == "note"
        assert "hello world" in doc.text
        assert "second paragraph" in doc.text
        assert doc.metadata == {}
        # RTF has no structural anchors — sections must stay empty so the
        # embedding path falls back to the anchor-free flow.
        assert doc.sections == []
        assert doc.size_bytes == f.stat().st_size
        assert doc.content_hash and doc.file_hash

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot stat"):
            RtfAdapter().extract(
                tmp_path / "nope.rtf", options=AdapterOptions(vault_root=tmp_path),
            )

    def test_rejects_oversized(self, tmp_path):
        f = tmp_path / "huge.rtf"
        # Don't actually write 25 MB — monkeypatch the cap to a tiny value.
        f.write_bytes(b"{\\rtf1 short body}")
        import grimore.ingest.adapters.rtf as rtf_mod
        original = rtf_mod._MAX_RTF_BYTES
        rtf_mod._MAX_RTF_BYTES = 4
        try:
            with pytest.raises(ValueError, match="exceeds"):
                RtfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        finally:
            rtf_mod._MAX_RTF_BYTES = original


# ── ODT ────────────────────────────────────────────────────────────────────


_ODT_MIMETYPE = "application/vnd.oasis.opendocument.text"
_ODT_MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
  <manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>
  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

_ODT_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/"'
)


def _make_odt(
    path: Path,
    *,
    title: str | None,
    paragraphs: list[tuple[str | None, str]],
) -> None:
    """Build a minimal .odt zip with content + meta + manifest.

    ``paragraphs`` is a list of ``(heading_text_or_None, body_text)``
    tuples. Heading entries become ``<text:h text:outline-level="1">``;
    body entries become ``<text:p>``.
    """
    parts: list[str] = []
    for heading, body in paragraphs:
        if heading is not None:
            parts.append(
                f'<text:h text:outline-level="1">{heading}</text:h>'
            )
        if body:
            parts.append(f"<text:p>{body}</text:p>")
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content {_ODT_NS}>'
        f"<office:body><office:text>"
        + "".join(parts)
        + "</office:text></office:body></office:document-content>"
    )
    meta = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-meta {_ODT_NS}>'
        f"<office:meta>"
        + (f"<dc:title>{title}</dc:title>" if title else "")
        + "</office:meta></office:document-meta>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        # ODT spec: mimetype first, uncompressed, no extra header bytes.
        zf.writestr("mimetype", _ODT_MIMETYPE, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", _ODT_MANIFEST)
        zf.writestr("content.xml", content)
        zf.writestr("meta.xml", meta)


class TestOdtAdapter:
    def test_title_from_meta(self, tmp_path):
        f = tmp_path / "a.odt"
        _make_odt(f, title="Real Title", paragraphs=[(None, "para")])
        doc = OdtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "odt"
        assert doc.title == "Real Title"
        assert doc.metadata.get("title") == "Real Title"

    def test_title_falls_back_to_first_h1_then_stem(self, tmp_path):
        f1 = tmp_path / "h.odt"
        _make_odt(
            f1, title=None,
            paragraphs=[("First Heading", "body"), (None, "more body")],
        )
        d1 = OdtAdapter().extract(f1, options=AdapterOptions(vault_root=tmp_path))
        assert d1.title == "First Heading"

        f2 = tmp_path / "stem.odt"
        _make_odt(f2, title=None, paragraphs=[(None, "no heading at all")])
        d2 = OdtAdapter().extract(f2, options=AdapterOptions(vault_root=tmp_path))
        assert d2.title == "stem"

    def test_headings_become_sections(self, tmp_path):
        f = tmp_path / "secs.odt"
        _make_odt(
            f, title=None,
            paragraphs=[
                ("Alpha", "alpha body"),
                ("Beta",  "beta body"),
            ],
        )
        doc = OdtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        headings = [s.heading for s in doc.sections]
        assert headings == ["Alpha", "Beta"]
        alpha = next(s for s in doc.sections if s.heading == "Alpha")
        assert alpha.text == "alpha body"

    def test_corrupt_zip_raises_valueerror(self, tmp_path):
        f = tmp_path / "broken.odt"
        f.write_bytes(b"not a zip at all")
        with pytest.raises(ValueError, match="not a valid zip"):
            OdtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_missing_content_xml_raises_valueerror(self, tmp_path):
        f = tmp_path / "empty.odt"
        with zipfile.ZipFile(f, "w") as zf:
            zf.writestr("mimetype", _ODT_MIMETYPE)
            zf.writestr("meta.xml", "<x/>")
        with pytest.raises(ValueError, match="content.xml"):
            OdtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_rejects_oversized(self, tmp_path):
        f = tmp_path / "big.odt"
        _make_odt(f, title="x", paragraphs=[("h", "body")])
        import grimore.ingest.adapters.odt as odt_mod
        original = odt_mod._MAX_ODT_BYTES
        odt_mod._MAX_ODT_BYTES = 8
        try:
            with pytest.raises(ValueError, match="exceeds"):
                OdtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        finally:
            odt_mod._MAX_ODT_BYTES = original


# ── DOC (antiword) ────────────────────────────────────────────────────────


class TestDocAdapter:
    def test_actionable_error_when_antiword_missing(self, tmp_path):
        f = tmp_path / "legacy.doc"
        f.write_bytes(b"\xd0\xcf\x11\xe0fake")
        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value=None,
        ):
            with pytest.raises(ValueError, match="antiword"):
                DocAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_happy_path_mocks_subprocess(self, tmp_path):
        f = tmp_path / "legacy.doc"
        f.write_bytes(b"\xd0\xcf\x11\xe0fake")

        class _Completed:
            returncode = 0
            stdout = b"plain text from antiword\nsecond line\n"
            stderr = b""

        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value="/usr/bin/antiword",
        ), patch(
            "grimore.ingest.adapters.doc.subprocess.run", return_value=_Completed(),
        ):
            doc = DocAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "doc"
        assert doc.title == "legacy"
        assert "plain text from antiword" in doc.text
        assert doc.sections == []  # no anchors from antiword output

    def test_nonzero_exit_raises(self, tmp_path):
        f = tmp_path / "broken.doc"
        f.write_bytes(b"\xd0\xcf\x11\xe0fake")

        class _Completed:
            returncode = 2
            stdout = b""
            stderr = b"antiword: parse failed"

        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value="/usr/bin/antiword",
        ), patch(
            "grimore.ingest.adapters.doc.subprocess.run", return_value=_Completed(),
        ):
            with pytest.raises(ValueError, match="antiword failed"):
                DocAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_timeout_raises(self, tmp_path):
        f = tmp_path / "slow.doc"
        f.write_bytes(b"\xd0\xcf\x11\xe0fake")
        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value="/usr/bin/antiword",
        ), patch(
            "grimore.ingest.adapters.doc.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="antiword", timeout=1),
        ):
            with pytest.raises(ValueError, match="timed out"):
                DocAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_antiword_available_reflects_path(self):
        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value=None,
        ):
            assert antiword_available() is False
        with patch(
            "grimore.ingest.adapters.doc.shutil.which", return_value="/usr/bin/antiword",
        ):
            assert antiword_available() is True


# ── PDF engine selection ─────────────────────────────────────────────────


class TestPdfEngineSelection:
    def test_rejects_unknown_engine_at_construct(self):
        with pytest.raises(ValueError, match="unknown pdf_engine"):
            PdfAdapter(engine="bogus")

    def test_supported_engines_advertised(self):
        # Sanity: the dispatch dict and the export agree.
        assert set(_SUPPORTED_ENGINES) == {"pypdf", "pdfplumber", "pymupdf"}

    def test_options_engine_overrides_construct_default(self, tmp_path, monkeypatch):
        """``AdapterOptions.pdf_engine`` must beat the construct-time default
        so a config swap on disk takes effect on the very next scan."""
        from grimore.ingest.adapters import pdf as pdf_mod

        calls: dict[str, int] = {}

        def fake_pypdf(path):
            calls["pypdf"] = calls.get("pypdf", 0) + 1
            return ({}, [(1, "from pypdf")])

        def fake_plumber(path):
            calls["pdfplumber"] = calls.get("pdfplumber", 0) + 1
            return ({}, [(1, "from pdfplumber")])

        monkeypatch.setitem(pdf_mod._ENGINE_DISPATCH, "pypdf", fake_pypdf)
        monkeypatch.setitem(pdf_mod._ENGINE_DISPATCH, "pdfplumber", fake_plumber)

        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-1.4\nplaceholder")

        adapter = PdfAdapter(engine="pypdf")
        # No override → uses construct-time default.
        doc1 = adapter.extract(
            f, options=AdapterOptions(vault_root=tmp_path),
        )
        # Override flips to pdfplumber.
        doc2 = adapter.extract(
            f, options=AdapterOptions(vault_root=tmp_path, pdf_engine="pdfplumber"),
        )
        assert "from pypdf" in doc1.text
        assert "from pdfplumber" in doc2.text
        assert calls == {"pypdf": 1, "pdfplumber": 1}

    def test_unknown_engine_in_options_raises(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-1.4\nplaceholder")
        with pytest.raises(ValueError, match="unknown pdf_engine"):
            PdfAdapter().extract(
                f, options=AdapterOptions(vault_root=tmp_path, pdf_engine="bogus"),
            )

    def test_pdfplumber_missing_extra_is_actionable(self, tmp_path, monkeypatch):
        """When the user opts into pdfplumber without installing the
        extra, the adapter must raise a ValueError that names the extra
        to install — not an ImportError that crashes the scan."""
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-1.4\nplaceholder")

        # Force the real import path to fail; the adapter catches it.
        import builtins
        real_import = builtins.__import__

        def deny_pdfplumber(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("No module named 'pdfplumber'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", deny_pdfplumber)
        with pytest.raises(ValueError, match="pdf-plumber"):
            PdfAdapter().extract(
                f, options=AdapterOptions(vault_root=tmp_path, pdf_engine="pdfplumber"),
            )


# ── OCR fallback ─────────────────────────────────────────────────────────


class TestPdfOcrFallback:
    def test_ocr_fills_empty_page_when_enabled(self, tmp_path, monkeypatch):
        """A page that the engine returns as empty must be promoted via
        OCR when the toggle is on and the OCR helper produces text.

        We monkeypatch both the engine and the OCR helper so the test
        runs without tesseract installed.
        """
        from grimore.ingest.adapters import pdf as pdf_mod

        def fake_engine(path):
            return ({}, [(1, ""), (2, "normal text")])

        def fake_ocr(path, page_no, timeout_s):
            return f"ocr text from page {page_no}" if page_no == 1 else None

        monkeypatch.setitem(pdf_mod._ENGINE_DISPATCH, "pypdf", fake_engine)
        monkeypatch.setattr(pdf_mod, "_ocr_page", fake_ocr)

        f = tmp_path / "scan.pdf"
        f.write_bytes(b"%PDF-1.4\nplaceholder")

        doc = PdfAdapter().extract(
            f, options=AdapterOptions(vault_root=tmp_path, ocr=True),
        )
        # Both pages produced sections — the first via OCR (tagged), the
        # second straight from the engine.
        assert len(doc.sections) == 2
        ocr_section = next(s for s in doc.sections if s.page == 1)
        assert ocr_section.heading == "(ocr)"
        assert "ocr text" in ocr_section.text
        normal = next(s for s in doc.sections if s.page == 2)
        assert normal.heading is None
        assert normal.text == "normal text"

    def test_ocr_disabled_drops_empty_pages(self, tmp_path, monkeypatch):
        from grimore.ingest.adapters import pdf as pdf_mod

        def fake_engine(path):
            return ({}, [(1, ""), (2, "real")])

        def fake_ocr(path, page_no, timeout_s):  # pragma: no cover
            raise AssertionError("OCR must not be called when toggle is off")

        monkeypatch.setitem(pdf_mod._ENGINE_DISPATCH, "pypdf", fake_engine)
        monkeypatch.setattr(pdf_mod, "_ocr_page", fake_ocr)

        f = tmp_path / "scan.pdf"
        f.write_bytes(b"%PDF-1.4\nplaceholder")

        doc = PdfAdapter().extract(
            f, options=AdapterOptions(vault_root=tmp_path, ocr=False),
        )
        # Only the non-empty page survives.
        assert [s.page for s in doc.sections] == [2]
