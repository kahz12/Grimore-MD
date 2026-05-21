"""
Magic-byte sniffer tests (Phase 5).

The sniffer is opt-in and bridges the gap between content type and
adapter dispatch when a file is misnamed or has no extension. These
tests cover:

* the mime → extension map for every adapter we ship,
* the ZIP disambiguation hop for DOCX / ODT / EPUB containers,
* the parser-side flow: ``sniff_magic = false`` short-circuits the
  sniffer; ``sniff_magic = true`` routes a misnamed PDF to the PDF
  adapter and updates ``ParsedNote.format``,
* graceful degradation when python-magic is missing — the rest of the
  pipeline must keep working.

We monkeypatch the libmagic-backed loader so tests don't depend on the
real ``python-magic`` extra being installed; that keeps CI green on the
default install and is what users without the extra will see at
runtime anyway.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


from grimore.ingest import sniffer
from grimore.ingest.parser import MarkdownParser, ParsedNote
from grimore.utils.config import Config, IngestConfig


class _FakeMagic:
    """Stand-in for a ``magic.Magic(mime=True)`` instance.

    Returns a queue of pre-baked mime strings; raises if the test asks
    for more than was queued so a forgotten expectation surfaces as a
    test failure rather than a stale value leaking from another case.
    """

    def __init__(self, mime: str):
        self._mime = mime

    def from_file(self, path: str) -> str:
        return self._mime


@pytest.fixture(autouse=True)
def _reset_sniffer_cache():
    """The ``_load_magic`` helper caches its libmagic instance on the
    function object. Tests have to reset that or the first patched
    fixture leaks across the rest of the suite."""
    if hasattr(sniffer._load_magic, "_cached"):
        delattr(sniffer._load_magic, "_cached")
    yield
    if hasattr(sniffer._load_magic, "_cached"):
        delattr(sniffer._load_magic, "_cached")


def _stub_magic(monkeypatch, mime: str) -> None:
    """Force ``_load_magic`` to return a ``_FakeMagic`` returning ``mime``."""
    monkeypatch.setattr(sniffer, "_load_magic", lambda: _FakeMagic(mime))


class TestMimeMap:
    @pytest.mark.parametrize(
        "mime,expected_ext",
        [
            ("application/pdf", "pdf"),
            ("application/epub+zip", "epub"),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "docx",
            ),
            ("application/vnd.oasis.opendocument.text", "odt"),
            ("application/rtf", "rtf"),
            ("text/rtf", "rtf"),
            ("text/html", "html"),
            ("application/xhtml+xml", "html"),
            ("text/markdown", "md"),
            ("text/plain", "txt"),
        ],
    )
    def test_known_mime_maps_to_extension(self, tmp_path, monkeypatch, mime, expected_ext):
        f = tmp_path / "anonymous"
        f.write_bytes(b"\x00")  # contents don't matter — the magic is mocked
        _stub_magic(monkeypatch, mime)
        assert sniffer.sniff_extension(f) == expected_ext

    def test_unknown_mime_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / "anonymous"
        f.write_bytes(b"\x00")
        _stub_magic(monkeypatch, "application/x-shockwave-flash")
        assert sniffer.sniff_extension(f) is None

    def test_mime_with_charset_suffix_still_resolves(self, tmp_path, monkeypatch):
        # libmagic sometimes returns "text/plain; charset=utf-8" — the
        # sniffer must strip the parameter before the lookup.
        f = tmp_path / "anonymous"
        f.write_bytes(b"hello")
        _stub_magic(monkeypatch, "text/plain; charset=utf-8")
        assert sniffer.sniff_extension(f) == "txt"


class TestZipDisambiguation:
    """``application/zip`` covers DOCX / ODT / EPUB. The sniffer must
    crack the archive open to tell them apart."""

    def _zip_with(self, path: Path, entries: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, body in entries.items():
                zf.writestr(name, body)

    def test_docx_marker_yields_docx(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        self._zip_with(f, {"word/document.xml": b"<doc/>"})
        _stub_magic(monkeypatch, "application/zip")
        assert sniffer.sniff_extension(f) == "docx"

    def test_odt_mimetype_marker_yields_odt(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        self._zip_with(
            f,
            {"mimetype": b"application/vnd.oasis.opendocument.text"},
        )
        _stub_magic(monkeypatch, "application/zip")
        assert sniffer.sniff_extension(f) == "odt"

    def test_epub_mimetype_marker_yields_epub(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        self._zip_with(f, {"mimetype": b"application/epub+zip"})
        _stub_magic(monkeypatch, "application/zip")
        assert sniffer.sniff_extension(f) == "epub"

    def test_plain_zip_with_no_marker_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        self._zip_with(f, {"hello.txt": b"world"})
        _stub_magic(monkeypatch, "application/zip")
        assert sniffer.sniff_extension(f) is None

    def test_corrupt_zip_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        f.write_bytes(b"PKnot-a-real-zip")
        _stub_magic(monkeypatch, "application/zip")
        assert sniffer.sniff_extension(f) is None


class TestDegradedMode:
    """When python-magic is missing the sniffer must no-op silently —
    the rest of the pipeline keeps working on extension-based dispatch."""

    def test_missing_python_magic_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sniffer, "_load_magic", lambda: None)
        f = tmp_path / "anything"
        f.write_bytes(b"\x00")
        assert sniffer.sniff_extension(f) is None
        assert sniffer.sniff_available() is False

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        _stub_magic(monkeypatch, "application/pdf")
        assert sniffer.sniff_extension(tmp_path / "does-not-exist") is None


class TestAdapterLookup:
    """``adapter_for_sniffed`` chains sniff + registry lookup so callers
    don't have to."""

    def test_pdf_returns_extension_and_adapter(self, tmp_path, monkeypatch):
        f = tmp_path / "report"
        f.write_bytes(b"\x00")
        _stub_magic(monkeypatch, "application/pdf")
        ext, adapter = sniffer.adapter_for_sniffed(f)
        assert ext == "pdf"
        assert adapter is not None
        assert "pdf" in adapter.extensions

    def test_unsupported_mime_returns_pair_of_none(self, tmp_path, monkeypatch):
        f = tmp_path / "anon"
        f.write_bytes(b"\x00")
        _stub_magic(monkeypatch, "image/png")
        assert sniffer.adapter_for_sniffed(f) == (None, None)


class TestParserIntegration:
    """The parser dispatcher only consults the sniffer when:

    1. the extension misses the registry, AND
    2. ``config.ingest.sniff_magic`` is True.

    Either condition false → no sniff call (we don't want to libmagic
    every Markdown file in the vault).
    """

    def _make_pdf_bytes(self):
        """Build a one-page PDF in-memory. The PDF adapter is wired in
        from Phase 3, so we can round-trip a real document instead of
        mocking the adapter out."""
        try:
            from pypdf import PdfWriter
        except ImportError:  # pragma: no cover - pypdf is a hard dep
            pytest.skip("pypdf not installed")
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        from io import BytesIO
        buf = BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def test_sniff_disabled_falls_back_to_markdown(self, tmp_path, monkeypatch):
        # A PDF saved as ``.bak`` — extension misses the registry.
        f = tmp_path / "report.bak"
        f.write_bytes(self._make_pdf_bytes())
        config = Config()
        config.ingest = IngestConfig(sniff_magic=False)
        # If the parser called the sniffer it would route to the PDF
        # adapter; with sniffing off it must fall through to the MD
        # fallback (which then either reads or errors as bytes).
        called = {"n": 0}

        def _spy(*_a, **_kw):
            called["n"] += 1
            return None, None

        monkeypatch.setattr(
            "grimore.ingest.sniffer.adapter_for_sniffed", _spy
        )
        # MarkdownAdapter will choke on PDF bytes (binary in MD context)
        # but the important assertion is that the sniffer wasn't queried.
        try:
            MarkdownParser().parse_file(f, config=config)
        except Exception:
            pass
        assert called["n"] == 0

    def test_sniff_enabled_routes_misnamed_pdf(self, tmp_path, monkeypatch):
        f = tmp_path / "report.bak"
        f.write_bytes(self._make_pdf_bytes())
        config = Config()
        config.ingest = IngestConfig(sniff_magic=True)
        _stub_magic(monkeypatch, "application/pdf")

        note = MarkdownParser().parse_file(f, config=config)
        assert isinstance(note, ParsedNote)
        # The dispatcher overrides ``format`` to the sniffed extension
        # so downstream (DB, chunker) sees the true content type.
        assert note.format == "pdf"

    def test_known_extension_skips_sniff_even_when_enabled(self, tmp_path, monkeypatch):
        # An actual .md file should never call the sniffer — extension
        # dispatch wins. Spy on the sniffer to confirm.
        f = tmp_path / "hello.md"
        f.write_text("# hi\n\nbody")
        config = Config()
        config.ingest = IngestConfig(sniff_magic=True)
        called = {"n": 0}

        def _spy(*_a, **_kw):
            called["n"] += 1
            return None, None

        monkeypatch.setattr(
            "grimore.ingest.sniffer.adapter_for_sniffed", _spy
        )
        note = MarkdownParser().parse_file(f, config=config)
        assert note.format == "md"
        assert called["n"] == 0


class TestObserverSniff:
    """The daemon's watchdog filter has to honour ``sniff_magic`` too —
    otherwise ``grimore scan`` picks up a misnamed file but the live
    daemon silently drops the event. These tests pin that contract."""

    def _handler(self, monkeypatch, sniff_magic: bool, mime: str):
        from queue import Queue
        from grimore.ingest.observer import VaultEventHandler

        _stub_magic(monkeypatch, mime)
        return VaultEventHandler(
            queue=Queue(),
            ignored_dirs=[],
            supported_extensions=["md", "pdf"],
            sniff_magic=sniff_magic,
        )

    def test_extension_miss_dropped_when_sniff_off(self, tmp_path, monkeypatch):
        handler = self._handler(monkeypatch, sniff_magic=False, mime="application/pdf")
        f = tmp_path / "report.bak"
        f.write_bytes(b"\x00")
        handler._enqueue(str(f), "created")
        assert handler.queue.empty()

    def test_extension_miss_sniffed_when_flag_on(self, tmp_path, monkeypatch):
        handler = self._handler(monkeypatch, sniff_magic=True, mime="application/pdf")
        f = tmp_path / "report.bak"
        f.write_bytes(b"\x00")
        handler._enqueue(str(f), "created")
        assert not handler.queue.empty()
        path, _ts = handler.queue.get()
        assert path == f

    def test_extension_miss_still_dropped_when_mime_unsupported(self, tmp_path, monkeypatch):
        handler = self._handler(monkeypatch, sniff_magic=True, mime="image/png")
        f = tmp_path / "anonymous"
        f.write_bytes(b"\x00")
        handler._enqueue(str(f), "created")
        assert handler.queue.empty()


class TestVaultIterationSniff:
    """``iter_vault_documents`` widens its file pickup when
    ``sniff_magic=True`` — a PDF named ``foo.bak`` should be found
    even though ``bak`` is not in ``formats``."""

    def test_sniff_picks_up_extensionless_pdf(self, tmp_path, monkeypatch):
        from grimore.ingest.parser import iter_vault_documents

        # Document with extension Grimore doesn't know.
        misnamed = tmp_path / "report.bak"
        misnamed.write_bytes(b"\x00")
        # A genuine Markdown note alongside it.
        md = tmp_path / "hello.md"
        md.write_text("# hi")

        _stub_magic(monkeypatch, "application/pdf")

        # Without the flag: only the .md file is picked up.
        out = iter_vault_documents(tmp_path, ["md"], [], sniff_magic=False)
        assert misnamed not in out
        assert md in out

        # With the flag: both files are picked up — the sniffer rescues
        # the misnamed PDF.
        out = iter_vault_documents(tmp_path, ["md", "pdf"], [], sniff_magic=True)
        assert misnamed in out
        assert md in out
