"""PDF extraction settings invalidate only the affected indexed documents."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from grimore.cognition.embedder import Embedder
from grimore.daemon import GrimoreDaemon
from grimore.ingest.adapters.base import ExtractedSection, ExtractionIssue
from grimore.ingest.extraction_profile import (
    pdf_extraction_profile, pdf_index_profile, pdf_retry_pending,
)
from grimore.ingest.parser import ParsedNote
from grimore.memory.db import Database
from grimore.utils.config import Config
from grimore.utils.security import SecurityGuard


@pytest.mark.parametrize("surface", ["scan", "daemon"])
@pytest.mark.parametrize("setting", ["ocr", "language", "budget"])
def test_ocr_setting_reindexes_unchanged_pdf_and_refreshes_anchor(
    tmp_path, monkeypatch, surface, setting,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "book.pdf"
    source.write_bytes(b"placeholder")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    config.ingest.ocr = setting != "ocr"

    parser = MagicMock()

    def parse(*_args, **_kwargs):
        changed = {
            "language": config.ingest.ocr_language == "spa",
            "budget": config.ingest.ocr_max_pages == 2,
            "ocr": config.ingest.ocr,
        }[setting]
        page = 2 if changed else 1
        return ParsedNote(
            path=source, title="Book", metadata={}, content="Stable text.",
            content_hash="stable-content", format="pdf",
            sections=[ExtractedSection("Stable text.", page=page)],
        )

    parser.parse_file.side_effect = parse
    tagger = MagicMock()
    tagger.tag_note.return_value = {"tags": [], "summary": "", "category": ""}
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.return_value = [[1.0, 0.0]]
    writer = MagicMock()
    writer.write_metadata.return_value = source
    writer.preview_metadata.return_value = None

    with Database(config.memory.db_path) as db:
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.MarkdownParser", lambda: parser)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
            monkeypatch.setattr("grimore.cli.FrontmatterWriter", lambda: writer)

            def process(*, dry_run=False):
                result = CliRunner().invoke(
                    app, ["scan", "--dry-run" if dry_run else "--no-dry-run"],
                )
                assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = parser, SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer, daemon.tagger, daemon.embedder = writer, tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0

            def process(*, dry_run=False):
                config.output.dry_run = dry_run
                daemon.process_file(source)

        process()
        with db._get_connection() as conn:
            note_id = conn.execute(
                "SELECT id FROM notes WHERE path = ?", (str(source),)
            ).fetchone()[0]
        assert db.get_chunk_records(note_id)[0][1] == 1
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)

        if setting == "language":
            config.ingest.ocr_language = "spa"
        elif setting == "budget":
            config.ingest.ocr_max_pages = 2
        else:
            config.ingest.ocr = True
        process(dry_run=True)
        assert db.get_chunk_records(note_id)[0][1] == 1
        assert db.get_ingest_state(str(source))[4] != pdf_extraction_profile(config)

        process()
        assert db.get_chunk_records(note_id)[0][1] == 2
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)
        assert parser.parse_file.call_count == 3
        assert tagger.tag_note.call_count == 3
        assert embedder.embed_batch.call_count == 1

        process()
        assert parser.parse_file.call_count == 3

        # Legacy rows upgraded from a database without this column are
        # re-extracted once, then resume the cheap skip path.
        with db._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET extraction_profile = NULL WHERE id = ?", (note_id,)
            )
        process()
        assert parser.parse_file.call_count == 4
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)


def test_inactive_ocr_timeout_does_not_change_profile():
    config = Config()
    first = pdf_extraction_profile(config)
    config.ingest.ocr_timeout_s = 60
    assert pdf_extraction_profile(config) == first

    config.ingest.ocr = True
    assert pdf_extraction_profile(config) != first
    enabled = pdf_extraction_profile(config)
    config.ingest.ocr_timeout_s = 90
    assert pdf_extraction_profile(config) != enabled

    config.ingest.pdf_engine = "pdfplumber"
    assert pdf_extraction_profile(config) != enabled


def test_language_changes_only_active_ocr_profile_and_keeps_english_compatibility():
    config = Config()
    first = pdf_extraction_profile(config)
    config.ingest.ocr_language = "spa"
    assert pdf_extraction_profile(config) == first
    config.ingest.ocr = True
    assert ":language=spa" in pdf_extraction_profile(config)
    config.ingest.ocr_language = "eng"
    assert pdf_extraction_profile(config) == "pdf-layout-v1:engine=pypdf:ocr=1:timeout=30"
    config.ingest.ocr_language = "spa+eng"
    combined = pdf_extraction_profile(config)
    config.ingest.ocr_language = "eng+spa"
    assert pdf_extraction_profile(config) != combined


def test_only_transient_ocr_failures_mark_profile_for_retry():
    config = Config()
    config.ingest.ocr = True
    base = pdf_extraction_profile(config)

    for reason in ("ocr_unavailable", "ocr_render_failed", "ocr_recognition_failed"):
        stored = pdf_index_profile(config, {
            "status": "partial", "issues": [{"reason": reason, "recovered": False}],
        })
        assert stored != base
        assert pdf_retry_pending(stored, config)

    for reason in ("ocr_no_text",):
        stored = pdf_index_profile(config, {
            "status": "partial", "issues": [{"reason": reason, "recovered": False}],
        })
        assert stored == base
        assert not pdf_retry_pending(stored, config)


@pytest.mark.parametrize("surface", ["scan", "daemon"])
def test_transient_ocr_failure_retries_unchanged_pdf_without_retagging(
    tmp_path, monkeypatch, surface,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "book.pdf"
    source.write_bytes(b"placeholder")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    config.ingest.ocr = True

    attempts = []
    parser = MagicMock()

    def parse(*_args, **_kwargs):
        attempts.append(len(attempts) + 1)
        reason = "ocr_render_failed" if len(attempts) < 3 else "ocr_no_text"
        return ParsedNote(
            path=source, title="Book", metadata={}, content="Stable text.",
            content_hash="stable-content", format="pdf",
            sections=[ExtractedSection("Stable text.", page=1)],
            issues=[ExtractionIssue(
                "page_text_unavailable", page=2, engine="pypdf",
                ocr_requested=True, reason=reason,
            )],
        )

    parser.parse_file.side_effect = parse
    tagger = MagicMock()
    tagger.tag_note.return_value = {"tags": [], "summary": "", "category": ""}
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.return_value = [[1.0, 0.0]]
    writer = MagicMock()
    writer.write_metadata.return_value = source
    writer.refresh_extraction_metadata.return_value = None

    with Database(config.memory.db_path) as db:
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.MarkdownParser", lambda: parser)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
            monkeypatch.setattr("grimore.cli.FrontmatterWriter", lambda: writer)

            def process():
                result = CliRunner().invoke(app, ["scan", "--no-dry-run"])
                assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = parser, SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer, daemon.tagger, daemon.embedder = writer, tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0

            def process():
                daemon.process_file(source)

        process()
        assert pdf_retry_pending(db.get_ingest_state(str(source))[4], config)
        assert tagger.tag_note.call_count == 1
        assert embedder.embed_batch.call_count == 1

        process()
        assert pdf_retry_pending(db.get_ingest_state(str(source))[4], config)
        assert tagger.tag_note.call_count == 1
        assert embedder.embed_batch.call_count == 1

        process()
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)
        assert tagger.tag_note.call_count == 1
        assert embedder.embed_batch.call_count == 1

        process()
        assert parser.parse_file.call_count == 3
