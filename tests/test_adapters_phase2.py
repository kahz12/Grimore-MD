"""
Per-adapter unit tests for the Phase 2 formats (TXT, HTML, DOCX).

Each test builds its fixture in-process — no binary blobs in the repo.
For DOCX we hand-craft the minimal Office Open XML zip the stdlib
adapter looks at (``word/document.xml`` + ``docProps/core.xml``).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.docx import DocxAdapter, _MAX_DOCX_BYTES
from grimore.ingest.adapters.html import HtmlAdapter, _MAX_HTML_BYTES
from grimore.ingest.adapters.txt import TxtAdapter, _MAX_TXT_BYTES


# ── TXT ────────────────────────────────────────────────────────────────────


class TestTxtAdapter:
    def test_happy_path(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line one\nline two\n")
        doc = TxtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "txt"
        assert doc.title == "log"
        assert doc.text == "line one\nline two\n"
        assert doc.metadata == {}
        assert doc.sections == []
        assert doc.size_bytes == f.stat().st_size
        assert doc.content_hash and doc.file_hash

    def test_replaces_invalid_utf8(self, tmp_path):
        f = tmp_path / "weird.txt"
        f.write_bytes(b"ok \xff\xfe junk\n")  # invalid UTF-8 in the middle
        doc = TxtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        # errors='replace' substitutes U+FFFD; the readable text must survive.
        assert "ok " in doc.text and "junk" in doc.text

    def test_rejects_oversized(self, tmp_path, monkeypatch):
        f = tmp_path / "huge.txt"
        f.write_bytes(b"x" * (_MAX_TXT_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            TxtAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot stat"):
            TxtAdapter().extract(
                tmp_path / "nope.txt", options=AdapterOptions(vault_root=tmp_path),
            )


# ── HTML ───────────────────────────────────────────────────────────────────


class TestHtmlAdapter:
    def test_title_from_title_tag(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text(
            "<html><head><title>Doc Title</title></head>"
            "<body><h1>Top</h1><p>hello</p></body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "html"
        assert doc.title == "Doc Title"
        assert "hello" in doc.text

    def test_title_falls_back_to_h1_then_filename(self, tmp_path):
        f1 = tmp_path / "a.html"
        f1.write_text("<html><body><h1>Heading One</h1><p>x</p></body></html>")
        assert HtmlAdapter().extract(
            f1, options=AdapterOptions(vault_root=tmp_path)
        ).title == "Heading One"

        f2 = tmp_path / "stem.html"
        f2.write_text("<html><body><p>no heading</p></body></html>")
        assert HtmlAdapter().extract(
            f2, options=AdapterOptions(vault_root=tmp_path)
        ).title == "stem"

    def test_strips_noise_tags(self, tmp_path):
        f = tmp_path / "noisy.html"
        f.write_text(
            "<html><head><title>T</title>"
            "<style>.x{color:red}</style></head>"
            "<body>"
            "  <nav>menu link</nav>"
            "  <script>alert(1)</script>"
            "  <main><p>real content</p></main>"
            "  <footer>copyright bar</footer>"
            "</body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert "real content" in doc.text
        assert "menu link" not in doc.text
        assert "alert(1)" not in doc.text
        assert "copyright bar" not in doc.text

    def test_prefers_main_over_body(self, tmp_path):
        f = tmp_path / "main.html"
        f.write_text(
            "<html><body>"
            "<aside><p>sidebar should be dropped</p></aside>"
            "<main><h1>Real</h1><p>real content</p></main>"
            "</body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        # The body fallback would include the aside; <main> selection drops it.
        assert "real content" in doc.text
        assert "sidebar" not in doc.text

    def test_sectionising_records_headings(self, tmp_path):
        f = tmp_path / "secs.html"
        f.write_text(
            "<html><body>"
            "<h1>Alpha</h1><p>a body</p>"
            "<h2>Beta</h2><p>b body</p>"
            "</body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        headings = [s.heading for s in doc.sections]
        assert "Alpha" in headings and "Beta" in headings

    def test_rejects_oversized(self, tmp_path):
        f = tmp_path / "huge.html"
        f.write_bytes(b"x" * (_MAX_HTML_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_blockquote_wrapping_paragraph_is_not_double_counted(self, tmp_path):
        # Regression: <blockquote><p>X</p></blockquote> used to emit "X"
        # twice (once for the blockquote, once for the inner paragraph).
        f = tmp_path / "q.html"
        f.write_text(
            "<html><body><h1>T</h1>"
            "<blockquote><p>quoted text</p></blockquote>"
            "</body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.text.count("quoted text") == 1

    def test_nested_list_keeps_both_outer_and_inner_text(self, tmp_path):
        # Regression: the dedup that fixed the blockquote case must not
        # drop the outer <li>'s own text. The outer leaf grabs the whole
        # subtree so we get both bits in one bundle.
        f = tmp_path / "nest.html"
        f.write_text(
            "<html><body><h1>T</h1>"
            "<ul><li>top<ul><li>nested</li></ul></li></ul>"
            "</body></html>"
        )
        doc = HtmlAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert "top" in doc.text
        assert "nested" in doc.text


# ── DOCX ───────────────────────────────────────────────────────────────────


_WORDML_NS = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""
_CORE_PROPS_NS = (
    "xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\""
    " xmlns:dc=\"http://purl.org/dc/elements/1.1/\""
)


def _make_docx(path: Path, *, title: str | None, paragraphs: list[tuple[str | None, str]]) -> None:
    """Build a minimal .docx with optional core props and styled paragraphs."""
    body_parts = []
    for style, text in paragraphs:
        pPr = (
            f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
        )
        body_parts.append(f"<w:p>{pPr}<w:r><w:t>{text}</w:t></w:r></w:p>")
    document_xml = (
        f"<?xml version=\"1.0\"?><w:document {_WORDML_NS}><w:body>"
        + "".join(body_parts)
        + "</w:body></w:document>"
    )
    core_xml = (
        f"<?xml version=\"1.0\"?><cp:coreProperties {_CORE_PROPS_NS}>"
        + (f"<dc:title>{title}</dc:title>" if title else "")
        + "</cp:coreProperties>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("docProps/core.xml", core_xml)
    path.write_bytes(buf.getvalue())


class TestDocxAdapter:
    def test_title_from_core_props(self, tmp_path):
        f = tmp_path / "a.docx"
        _make_docx(f, title="Real Title", paragraphs=[(None, "para")])
        doc = DocxAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "docx"
        assert doc.title == "Real Title"
        assert doc.metadata.get("title") == "Real Title"

    def test_title_falls_back_to_first_heading1_then_stem(self, tmp_path):
        f1 = tmp_path / "h.docx"
        _make_docx(
            f1, title=None,
            paragraphs=[("Heading1", "First Heading"), (None, "body")],
        )
        doc1 = DocxAdapter().extract(f1, options=AdapterOptions(vault_root=tmp_path))
        assert doc1.title == "First Heading"

        f2 = tmp_path / "stem.docx"
        _make_docx(f2, title=None, paragraphs=[(None, "no heading at all")])
        doc2 = DocxAdapter().extract(f2, options=AdapterOptions(vault_root=tmp_path))
        assert doc2.title == "stem"

    def test_headings_become_sections(self, tmp_path):
        f = tmp_path / "secs.docx"
        _make_docx(
            f, title=None,
            paragraphs=[
                ("Heading1", "Alpha"), (None, "alpha body"),
                ("Heading2", "Beta"),  (None, "beta body"),
            ],
        )
        doc = DocxAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        headings = [s.heading for s in doc.sections]
        assert headings == ["Alpha", "Beta"]
        # Each section's body should correspond to its paragraphs only.
        first = next(s for s in doc.sections if s.heading == "Alpha")
        assert first.text == "alpha body"

    def test_corrupt_zip_raises_valueerror(self, tmp_path):
        f = tmp_path / "broken.docx"
        f.write_bytes(b"not a zip at all")
        with pytest.raises(ValueError, match="not a valid zip"):
            DocxAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_missing_document_xml_raises_valueerror(self, tmp_path):
        f = tmp_path / "empty.docx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("docProps/core.xml", "<x/>")
        f.write_bytes(buf.getvalue())
        with pytest.raises(ValueError, match="word/document.xml"):
            DocxAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_rejects_oversized(self, tmp_path):
        f = tmp_path / "big.docx"
        f.write_bytes(b"\x50\x4b" + b"\x00" * _MAX_DOCX_BYTES)  # PK header + bulk
        with pytest.raises(ValueError, match="exceeds"):
            DocxAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
