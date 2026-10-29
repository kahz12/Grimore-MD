"""A PDF with unchanged text can still need fresh extraction metadata."""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import frontmatter
import pytest
from typer.testing import CliRunner

from grimore.cognition.embedder import Embedder
from grimore.daemon import GrimoreDaemon
from grimore.ingest.adapters.base import ExtractedSection, ExtractionIssue
from grimore.ingest.pdf_reconcile import reconcile_unchanged_document
from grimore.ingest.parser import ParsedNote
from grimore.memory.db import Database
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.utils.config import Config
from grimore.utils.hashing import sha256_file
from grimore.utils.paths import sidecar_path_for
from grimore.utils.security import SecurityGuard


@pytest.mark.parametrize("surface", ["scan", "daemon"])
def test_same_text_pdf_refreshes_existing_sidecar_without_llm(
    tmp_path, monkeypatch, surface,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "book.pdf"
    source.write_bytes(b"first-version")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    state = {"partial": False}
    parser = MagicMock()

    def parse(*_args, **_kwargs):
        issues = (
            [ExtractionIssue("page_text_unavailable", page=2, engine="pypdf")]
            if state["partial"] else []
        )
        return ParsedNote(
            path=source, title="Book", metadata={}, content="Same text.",
            content_hash="stable-content", format="pdf",
            file_hash=sha256_file(source),
            sections=[ExtractedSection("Same text.", page=2 if state["partial"] else 1)],
            issues=issues,
        )

    parser.parse_file.side_effect = parse
    tagger = MagicMock()
    tagger.tag_note.return_value = {
        "tags": ["curated"], "summary": "Original summary", "category": "reference",
    }
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.return_value = [[1.0, 0.0]]
    writer = FrontmatterWriter()

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
        note_id = db.get_note_id_by_path(str(source))
        assert db.get_chunk_records(note_id)[0][1] == 1
        sidecar = sidecar_path_for(source, vault)
        original = frontmatter.load(sidecar)
        assert original.metadata["extraction"]["status"] == "complete"
        original.content += "\n\nPersonal annotation."
        sidecar.write_text(frontmatter.dumps(original), encoding="utf-8")
        prior_hash = db.get_file_hash(str(source))

        source.write_bytes(b"second-version")
        state["partial"] = True
        process(dry_run=True)
        assert frontmatter.load(sidecar).metadata["extraction"]["status"] == "complete"
        assert db.get_file_hash(str(source)) == prior_hash
        assert db.get_chunk_records(note_id)[0][1] == 1

        process()
        refreshed = frontmatter.load(sidecar)
        assert refreshed.metadata["extraction"]["status"] == "partial"
        assert refreshed.metadata["extraction"]["issues"][0]["page"] == 2
        assert refreshed.metadata["file_hash"] == sha256_file(source)
        assert refreshed.metadata["tags"] == ["curated"]
        assert refreshed.metadata["summary"] == "Original summary"
        assert "Personal annotation." in refreshed.content
        assert db.get_file_hash(str(source)) == sha256_file(source)
        assert db.get_chunk_records(note_id)[0][1] == 2
        with db._get_connection() as conn:
            indexed_extraction = conn.execute(
                "SELECT extraction_json FROM notes WHERE id = ?", (note_id,)
            ).fetchone()[0]
        assert json.loads(indexed_extraction)["status"] == "partial"
        assert tagger.tag_note.call_count == 1
        assert embedder.embed_batch.call_count == 1

        process()
        assert parser.parse_file.call_count == 3

        source.write_bytes(b"third-version")
        state["partial"] = False
        previous_hash = db.get_file_hash(str(source))
        original_refresh = writer.refresh_extraction_metadata

        def fail_write(*args, **kwargs):
            if not kwargs.get("dry_run", True):
                raise OSError("disk full")
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(writer, "refresh_extraction_metadata", fail_write)
        process()
        assert db.get_file_hash(str(source)) == previous_hash
        assert db.get_chunk_records(note_id)[0][1] == 2
        assert frontmatter.load(sidecar).metadata["extraction"]["status"] == "partial"

        monkeypatch.setattr(writer, "refresh_extraction_metadata", original_refresh)
        process()
        assert db.get_file_hash(str(source)) == sha256_file(source)
        assert db.get_chunk_records(note_id)[0][1] == 1
        assert frontmatter.load(sidecar).metadata["extraction"]["status"] == "complete"
        assert tagger.tag_note.call_count == 1
        assert embedder.embed_batch.call_count == 1


def test_changed_pdf_chunk_boundaries_retry_after_embedding_failure(tmp_path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"placeholder")
    config = Config()
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.side_effect = [
        [[1.0, 0.0]],
        [None, None],
        [[0.0, 1.0], [1.0, 0.0]],
    ]

    def note(sections):
        return ParsedNote(
            path=source, title="Book", metadata={}, content="Alpha\n\nBeta",
            content_hash="stable-content", format="pdf", sections=sections,
        )

    original = note([ExtractedSection("Alpha\n\nBeta", page=1)])
    repaginated = note([
        ExtractedSection("Alpha", page=1),
        ExtractedSection("Beta", page=2),
    ])
    with Database(str(tmp_path / "index.db")) as db:
        note_id = db.upsert_note(str(source), "Book", "", format="pdf")
        reconcile_unchanged_document(
            db, embedder, original, config,
            clean_content=original.content, file_hash="first", size_bytes=12,
            mtime_ns=1, document_dates={},
        )
        assert db.get_chunk_records(note_id)[0][1] == 1

        with pytest.raises(RuntimeError, match="embedding incomplete"):
            reconcile_unchanged_document(
                db, embedder, repaginated, config,
                clean_content=repaginated.content, file_hash="second",
                size_bytes=12, mtime_ns=2, document_dates={},
            )
        assert db.get_file_hash(str(source)) == "first"

        reconcile_unchanged_document(
            db, embedder, repaginated, config,
            clean_content=repaginated.content, file_hash="second",
            size_bytes=12, mtime_ns=2, document_dates={},
        )
        assert db.get_file_hash(str(source)) == "second"
        assert len(db.get_chunk_records(note_id)) == 2
        assert db.get_chunk_records(note_id)[1][1] == 2
