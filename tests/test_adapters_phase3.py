"""
Per-adapter unit tests for the Phase 3 formats (PDF, EPUB) and the
anchor-propagation chain.

Each test builds its fixture in-process — PDFs are synthesized via
``pypdf`` (the same library the adapter uses to read them back), and
EPUBs are hand-built minimal zips conforming to the IDPF Open Container
Format.

The final ``TestAnchorPropagation`` class pins the end-to-end contract
that page / heading anchors survive the chain:

    ExtractedSection  →  chunk_sections  →  embed_sections
                                      ↘   db.store_embedding (page/heading cols)
                                                ↘   db.get_chunk_anchors
                                                          ↘   Oracle citation label
"""
from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from grimore.cognition.chunker import Chunk, chunk_sections
from grimore.cognition.oracle import _format_source_label
from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedSection,
)
from grimore.ingest.adapters.epub import EpubAdapter, _MAX_EPUB_BYTES
from grimore.ingest.adapters.pdf import PdfAdapter, _MAX_PDF_BYTES
from grimore.memory.db import Database


# ── PDF ────────────────────────────────────────────────────────────────────


def _make_pdf(
    path: Path,
    pages: list[str],
    *,
    title: str | None = None,
    author: str | None = None,
) -> None:
    """Build a real PDF in-process via pypdf. We use ``add_blank_page`` and
    inject text via a tiny content stream so ``page.extract_text()`` returns
    something deterministic on read-back.

    pypdf's high-level writers don't have a text-drawing helper, so we
    synthesize the minimal ``BT … ET`` content stream by hand and attach
    it to each page via ``PageObject.merge_page`` on a content-stream
    page — but the simplest path that actually round-trips through
    ``extract_text`` is to use ``reportlab`` if available, else fall back
    to a hand-rolled raw PDF. We avoid the reportlab dependency by
    crafting a minimal PDF entirely with pypdf's low-level objects.
    """
    from pypdf import PdfWriter
    from pypdf.generic import (
        ContentStream,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        ArrayObject,
    )

    writer = PdfWriter()
    for body in pages:
        page = writer.add_blank_page(width=612, height=792)
        # Build a content stream that draws ``body`` at (72, 720) using Helvetica.
        # ``Tj`` operator requires parenthesised literal strings; escape () and \\.
        escaped = body.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_bytes = (
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        ).encode("latin-1")
        content = DecodedStreamObject()
        content.set_data(stream_bytes)
        page[NameObject("/Contents")] = content
        # Register Helvetica as /F1 on the page's resource dict.
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        resources = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        page[NameObject("/Resources")] = resources

    metadata: dict[str, str] = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if metadata:
        writer.add_metadata(metadata)

    with open(path, "wb") as fh:
        writer.write(fh)


class TestPdfAdapter:
    def test_happy_path_per_page_sections(self, tmp_path):
        f = tmp_path / "doc.pdf"
        _make_pdf(f, ["Hello page one", "Hello page two", "Hello page three"])
        doc = PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "pdf"
        # Every non-empty page becomes a section keyed on its 1-based page no.
        pages = [s.page for s in doc.sections]
        assert pages == [1, 2, 3]
        # PDF sections never carry a heading — only the page anchor.
        assert all(s.heading is None for s in doc.sections)
        # Body should be the concatenation of every section's text.
        assert "page one" in doc.text
        assert "page three" in doc.text
        # File-level provenance fields are filled.
        assert doc.size_bytes == f.stat().st_size
        assert doc.content_hash and doc.file_hash
        assert doc.content_hash != doc.file_hash

    def test_title_from_metadata(self, tmp_path):
        f = tmp_path / "meta.pdf"
        _make_pdf(f, ["body"], title="From Info Dict")
        doc = PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.title == "From Info Dict"
        assert doc.metadata.get("title") == "From Info Dict"

    def test_title_falls_back_to_first_page_heading(self, tmp_path):
        f = tmp_path / "heading.pdf"
        # No /Title in info dict; first line of page 1 should win.
        _make_pdf(f, ["First Line As Heading", "body text on page two"])
        doc = PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        # Either the heading or its trimmed prefix — both ok per the regex.
        assert doc.title.startswith("First Line")

    def test_title_falls_back_to_filename_stem(self, tmp_path):
        f = tmp_path / "stem.pdf"
        # First page only has lowercase punctuation that won't pass the
        # "heading-ish" regex, but the regex's fallback returns the first
        # non-empty line trimmed — so we explicitly empty the first page
        # by writing whitespace-only content and put body on later pages.
        _make_pdf(f, [" ", " ", " "])
        doc = PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        # All pages collapsed to empty after strip — sections list empty,
        # title falls all the way through to filename stem.
        assert doc.title == "stem"
        assert doc.sections == []

    def test_corrupt_pdf_raises_valueerror(self, tmp_path):
        f = tmp_path / "broken.pdf"
        f.write_bytes(b"%PDF-1.4\nthis is not a real pdf body")
        with pytest.raises(ValueError):
            PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot stat"):
            PdfAdapter().extract(
                tmp_path / "nope.pdf", options=AdapterOptions(vault_root=tmp_path),
            )

    def test_rejects_oversized(self, tmp_path):
        # Don't actually allocate 100MB — monkeypatch the cap instead via
        # a placeholder file that just trips the size check.
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"%PDF-1.4\n" + b"\x00" * 16)
        # Patch the module-level cap to something tiny for this assertion.
        import grimore.ingest.adapters.pdf as pdf_mod
        original = pdf_mod._MAX_PDF_BYTES
        pdf_mod._MAX_PDF_BYTES = 8
        try:
            with pytest.raises(ValueError, match="exceeds"):
                PdfAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        finally:
            pdf_mod._MAX_PDF_BYTES = original


# ── EPUB ───────────────────────────────────────────────────────────────────


_MIMETYPE = "application/epub+zip"
_CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


def _make_epub(
    path: Path,
    *,
    title: str | None,
    chapters: list[tuple[str, str, str]],
    omit_container: bool = False,
) -> None:
    """Build a minimal EPUB.

    ``chapters`` is a list of ``(idref, heading_or_None, body_text)``
    tuples — they go into the manifest, the spine (in order), and as
    separate XHTML files. ``heading`` is rendered as ``<h1>`` if set.
    """
    manifest_items = "\n".join(
        f'    <item id="{idref}" href="{idref}.xhtml" media-type="application/xhtml+xml"/>'
        for idref, _, _ in chapters
    )
    spine_items = "\n".join(
        f'    <itemref idref="{idref}"/>' for idref, _, _ in chapters
    )
    dc_title = f"<dc:title>{title}</dc:title>" if title is not None else ""

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    {dc_title}
    <dc:identifier id="bid">urn:test:1</dc:identifier>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
{manifest_items}
  </manifest>
  <spine>
{spine_items}
  </spine>
</package>"""

    with zipfile.ZipFile(path, "w") as zf:
        # mimetype must be the first entry and uncompressed per the spec —
        # the adapter doesn't validate that, but writing it correctly keeps
        # the fixtures honest.
        zf.writestr("mimetype", _MIMETYPE, compress_type=zipfile.ZIP_STORED)
        if not omit_container:
            zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf)
        for idref, heading, body in chapters:
            heading_html = f"<h1>{heading}</h1>" if heading else ""
            xhtml = (
                f'<?xml version="1.0" encoding="utf-8"?>'
                f'<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{idref}</title></head>"
                f"<body>{heading_html}<p>{body}</p></body></html>"
            )
            zf.writestr(f"OEBPS/{idref}.xhtml", xhtml)


class TestEpubAdapter:
    def test_happy_path_chapters_to_sections(self, tmp_path):
        f = tmp_path / "book.epub"
        _make_epub(
            f,
            title="Book Title",
            chapters=[
                ("ch1", "Chapter One", "first body"),
                ("ch2", "Chapter Two", "second body"),
            ],
        )
        doc = EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.format == "epub"
        assert doc.title == "Book Title"
        assert doc.metadata.get("title") == "Book Title"
        assert doc.metadata.get("language") == "en"
        # Spine order is preserved as section order.
        headings = [s.heading for s in doc.sections]
        assert headings == ["Chapter One", "Chapter Two"]
        # EPUB sections carry headings, never page numbers.
        assert all(s.page is None for s in doc.sections)
        # Body text from each chapter shows up.
        assert "first body" in doc.text
        assert "second body" in doc.text
        # Size + hashes filled in.
        assert doc.size_bytes == f.stat().st_size
        assert doc.content_hash and doc.file_hash

    def test_title_falls_back_to_first_heading(self, tmp_path):
        f = tmp_path / "noinfo.epub"
        _make_epub(
            f, title=None,
            chapters=[("intro", "First Heading", "x")],
        )
        doc = EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.title == "First Heading"

    def test_title_falls_back_to_filename_stem(self, tmp_path):
        f = tmp_path / "stem.epub"
        _make_epub(
            f, title=None,
            chapters=[("only", None, "body without a heading")],
        )
        doc = EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert doc.title == "stem"

    def test_empty_chapters_are_dropped(self, tmp_path):
        f = tmp_path / "sparse.epub"
        _make_epub(
            f, title="X",
            chapters=[
                ("ch1", "Real", "actual content"),
                ("ch2", None, ""),  # Empty body → must not produce a section.
            ],
        )
        doc = EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        # Only one section, from the chapter with real content.
        assert len(doc.sections) == 1
        assert doc.sections[0].heading == "Real"

    def test_bad_zip_raises_valueerror(self, tmp_path):
        f = tmp_path / "broken.epub"
        f.write_bytes(b"not a zip at all")
        with pytest.raises(ValueError, match="not a valid zip"):
            EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_missing_container_xml_raises_valueerror(self, tmp_path):
        f = tmp_path / "nocontainer.epub"
        _make_epub(
            f, title="X",
            chapters=[("ch1", "H", "body")],
            omit_container=True,
        )
        with pytest.raises(ValueError, match="container.xml"):
            EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))

    def test_rejects_oversized(self, tmp_path):
        f = tmp_path / "huge.epub"
        _make_epub(f, title="X", chapters=[("c", "H", "b")])
        # Patch the cap to a value below the real file size.
        import grimore.ingest.adapters.epub as epub_mod
        original = epub_mod._MAX_EPUB_BYTES
        epub_mod._MAX_EPUB_BYTES = 8
        try:
            with pytest.raises(ValueError, match="exceeds"):
                EpubAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        finally:
            epub_mod._MAX_EPUB_BYTES = original


# ── Anchor propagation: chunker → embedder → db → oracle ──────────────────


class TestChunkSectionsAnchors:
    def test_page_anchor_propagates_to_each_chunk(self):
        sections = [
            ExtractedSection(text="alpha body", page=1, heading=None, order=0),
            ExtractedSection(text="beta body", page=42, heading=None, order=1),
        ]
        chunks = chunk_sections(sections)
        assert all(isinstance(c, Chunk) for c in chunks)
        # Map page → set of chunk texts that came from it.
        by_page = {c.page: c.text for c in chunks}
        assert by_page[1] == "alpha body"
        assert by_page[42] == "beta body"
        assert all(c.heading is None for c in chunks)

    def test_heading_anchor_propagates_to_each_chunk(self):
        sections = [
            ExtractedSection(text="intro", page=None, heading="Intro", order=0),
            ExtractedSection(text="deep dive", page=None, heading="Deep Dive", order=1),
        ]
        chunks = chunk_sections(sections)
        by_heading = {c.heading: c.text for c in chunks}
        assert by_heading["Intro"] == "intro"
        assert by_heading["Deep Dive"] == "deep dive"
        assert all(c.page is None for c in chunks)

    def test_oversize_section_splits_inherit_anchor(self):
        # A single section bigger than max_chars must produce multiple
        # chunks, all of which still cite back to the same page.
        big = "X" * 5000  # well above DEFAULT_MAX_CHARS = 1500
        sections = [ExtractedSection(text=big, page=7, heading=None, order=0)]
        chunks = chunk_sections(sections)
        assert len(chunks) > 1
        assert all(c.page == 7 for c in chunks)

    def test_empty_sections_dropped(self):
        sections = [
            ExtractedSection(text="real", page=1, heading=None, order=0),
            ExtractedSection(text="   ", page=2, heading=None, order=1),
        ]
        chunks = chunk_sections(sections)
        assert len(chunks) == 1
        assert chunks[0].page == 1


class _RecordingEmbedder:
    """Mimics the bits of ``Embedder`` that ``embed_sections`` calls.

    We avoid hitting Ollama by stubbing ``embed`` with a deterministic
    fake vector. The real ``embed_sections`` logic — chunk + embed +
    pair with anchors — is what we're actually testing.
    """
    def __init__(self):
        self.calls = 0

    def embed(self, text: str):
        self.calls += 1
        # Tiny but unit-normalisable vector; structure doesn't matter
        # since we never compare these in the test.
        return [1.0, 0.0]


def test_embed_sections_returns_anchor_tuples():
    from grimore.cognition.embedder import Embedder

    sections = [
        ExtractedSection(text="alpha body", page=1, heading=None, order=0),
        ExtractedSection(text="chapter body", page=None, heading="Chapter", order=1),
    ]
    # Call the unbound method directly so we don't need the network-touching
    # constructor; the fake replaces self.embed.
    fake = _RecordingEmbedder()
    results = Embedder.embed_sections(fake, sections)

    assert fake.calls == 2
    assert len(results) == 2
    by_text = {text: (page, heading) for text, _, page, heading in results}
    assert by_text["alpha body"] == (1, None)
    assert by_text["chapter body"] == (None, "Chapter")


def test_embed_sections_skips_when_embed_fails():
    from grimore.cognition.embedder import Embedder

    class _PartialEmbedder:
        def __init__(self):
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            # First chunk fails, second succeeds.
            return None if self.calls == 1 else [1.0, 0.0]

    sections = [
        ExtractedSection(text="will fail", page=1, heading=None, order=0),
        ExtractedSection(text="will succeed", page=2, heading=None, order=1),
    ]
    fake = _PartialEmbedder()
    results = Embedder.embed_sections(fake, sections)
    assert len(results) == 1
    assert results[0][0] == "will succeed"
    assert results[0][2] == 2


# ── DB ↔ Oracle round-trip ────────────────────────────────────────────────


def _fake_vector_blob() -> bytes:
    # 2-float vector packed the same way Embedder.serialize_vector does.
    return struct.pack("2f", 1.0, 0.0)


class TestDbAnchorRoundTrip:
    def test_store_and_retrieve_chunk_anchors(self, tmp_path):
        db = Database(str(tmp_path / "grimore.db"))
        note_id = db.upsert_note(
            path="/v/doc.pdf", title="Doc", content_hash="c" * 64,
            format="pdf", file_hash="f" * 64, size_bytes=1234,
        )
        # Two chunks at different page anchors plus one heading-only.
        db.store_embedding(note_id, 0, "page-one text", _fake_vector_blob(), page=1)
        db.store_embedding(note_id, 1, "page-two text", _fake_vector_blob(), page=2)
        db.store_embedding(
            note_id, 2, "heading-only text", _fake_vector_blob(),
            heading="Intro",
        )

        assert db.get_chunk_anchors(note_id, "page-one text") == (1, None)
        assert db.get_chunk_anchors(note_id, "page-two text") == (2, None)
        assert db.get_chunk_anchors(note_id, "heading-only text") == (None, "Intro")
        # Unknown text returns the empty tuple, not an error.
        assert db.get_chunk_anchors(note_id, "nothing matches") == (None, None)

    def test_legacy_embedding_has_no_anchors(self, tmp_path):
        """Back-compat: store_embedding without page/heading kwargs must
        still work and yield ``(None, None)`` anchors."""
        db = Database(str(tmp_path / "grimore.db"))
        note_id = db.upsert_note(
            path="/v/note.md", title="Note", content_hash="c" * 64,
        )
        db.store_embedding(note_id, 0, "md body chunk", _fake_vector_blob())
        assert db.get_chunk_anchors(note_id, "md body chunk") == (None, None)


# ── Oracle citation label ────────────────────────────────────────────────


class TestOracleCitationLabel:
    def test_page_wins_when_both_present(self):
        assert _format_source_label("Doc", page=42, heading="Intro") == "Doc#p.42"

    def test_heading_used_when_page_absent(self):
        assert _format_source_label("Doc", page=None, heading="Chapter One") == "Doc#Chapter One"

    def test_bare_title_when_no_anchors(self):
        assert _format_source_label("Doc", page=None, heading=None) == "Doc"

    def test_heading_is_collapsed_and_trimmed(self):
        # Newlines and pipes would break wikilink rendering — the helper
        # collapses whitespace and bounds the heading length.
        ugly = "Bad\nHeading\n\nwith\tlots of whitespace " + ("z" * 200)
        label = _format_source_label("T", page=None, heading=ugly)
        assert "\n" not in label
        # Total label length: "T#" + heading clamp (80 chars).
        assert len(label) <= 2 + 80
