"""Failed indexing remains retryable without another edit to the source."""
import threading
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from grimore.cognition.chunker import Chunk
from grimore.cognition.embedder import Embedder
from grimore.daemon import GrimoreDaemon
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.utils.config import Config
from grimore.utils.security import SecurityGuard


@pytest.mark.parametrize("surface", ["daemon", "scan"])
@pytest.mark.parametrize("failure", ["missing", "exception", "sensitive_remote", "write"])
def test_indexing_retries_missing_chunks_after_restart(tmp_path, monkeypatch, surface, failure):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "note.md"
    source.write_text("# Note\n\nFirst paragraph.\n\nSecond paragraph.\n")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    if failure == "sensitive_remote":
        source.write_text("# Note\n\nContact alice@example.com.\n")
        config.cognition.allow_remote = True
        monkeypatch.setenv("OLLAMA_HOST", "https://8.8.8.8")
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    first = [[1., 0.], None] if failure == "missing" else RuntimeError("offline")
    retry = [[0., 1.]] if failure == "missing" else [[1., 0.], [0., 1.]]
    embedder.embed_batch.side_effect = [first, retry]
    if failure in {"sensitive_remote", "write"}:
        embedder.embed_batch.side_effect = [[[1., 0.], [0., 1.]]]
    tagger = MagicMock()
    tagger.tag_note.return_value = {"tags": [], "summary": "", "category": ""}
    writer = MagicMock()
    writer.write_metadata.return_value = source
    if failure == "write":
        writer.write_metadata.side_effect = [OSError("disk full"), source]
    chunks = [Chunk("First paragraph."), Chunk("Second paragraph.")]
    module = "grimore.daemon" if surface == "daemon" else "grimore.cli"
    monkeypatch.setattr(module + ".build_candidate_chunks", lambda *a: chunks)

    if surface == "scan":
        from grimore.cli import app
        monkeypatch.setattr("grimore.cli.load_config", lambda: config)
        monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
        monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
        monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
        monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
        monkeypatch.setattr("grimore.cli.FrontmatterWriter", lambda: writer)

    def process(db):
        if surface == "scan":
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            result = CliRunner().invoke(app, ["scan", "--no-dry-run"])
            assert result.exit_code == 0, result.output
            return
        daemon = GrimoreDaemon.__new__(GrimoreDaemon)
        daemon.config, daemon.db, daemon.vault_root = config, db, vault
        daemon.parser, daemon.security = MarkdownParser(), SecurityGuard(str(vault))
        daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
        daemon.writer, daemon.tagger, daemon.embedder = writer, tagger, embedder
        daemon._counter_lock = threading.Lock()
        daemon.processed_count = 0
        daemon.process_file(source)

    with Database(config.memory.db_path) as db:
        process(db)
        if failure == "sensitive_remote":
            assert db.get_content_hash_by_path(str(source)) is None
            tagger.tag_note.assert_not_called()
            embedder.embed_batch.assert_not_called()
            config.cognition.allow_remote = False
            monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        elif failure == "write":
            assert db.get_content_hash_by_path(str(source)) is None
            embedder.embed_batch.assert_not_called()
        else:
            assert db.get_content_hash_by_path(str(source)) == ""
        assert db.get_file_hash(str(source)) is None
        assert db.embeddings_signature()[0] == (1 if failure == "missing" else 0)

    # A fresh Database ensures retryability is durable, not just local state.
    with Database(config.memory.db_path) as db:
        process(db)
        assert db.embeddings_signature()[0] == 2
        assert db.get_content_hash_by_path(str(source))
        assert db.get_file_state(str(source))[1:] == (
            source.stat().st_size, source.stat().st_mtime_ns,
        )
        expected_calls = 1 if failure in {"sensitive_remote", "write"} else 2
        assert embedder.embed_batch.call_count == expected_calls
        process(db)
        assert embedder.embed_batch.call_count == expected_calls


@pytest.mark.parametrize("surface", ["daemon", "scan"])
def test_indexing_preserves_manual_classification_override(tmp_path, monkeypatch, surface):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "note.md"
    source.write_text("# Note\n\nFirst paragraph.\n\nSecond paragraph.\n")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.return_value = [[1., 0.], [0., 1.]]
    tagger = MagicMock()
    tagger.tag_note.return_value = {
        "tags": ["generated"], "summary": "Summary", "category": "model/category",
    }
    writer = MagicMock()
    writer.write_metadata.return_value = source
    chunks = [Chunk("First paragraph."), Chunk("Second paragraph.")]
    module = "grimore.daemon" if surface == "daemon" else "grimore.cli"
    monkeypatch.setattr(module + ".build_candidate_chunks", lambda *a: chunks)

    if surface == "scan":
        from grimore.cli import app
        monkeypatch.setattr("grimore.cli.load_config", lambda: config)
        monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
        monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
        monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
        monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
        monkeypatch.setattr("grimore.cli.FrontmatterWriter", lambda: writer)

    with Database(config.memory.db_path) as db:
        note_id = db.upsert_note(str(source), "Note", content_hash="old")
        assert note_id is not None
        db.set_classification_override(
            note_id=note_id,
            category="human/reference",
            tags=["curated", "trusted"],
            updated_at="2026-09-25T12:00:00Z",
        )
        if surface == "scan":
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            result = CliRunner().invoke(app, ["scan", "--no-dry-run"])
            assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = MarkdownParser(), SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer, daemon.tagger, daemon.embedder = writer, tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0
            daemon.process_file(source)

        metadata = writer.write_metadata.call_args.args[1]
        visible = db.get_note_classifications([note_id])[note_id]
        timeline = db.get_note_timeline(note_id)
        assert metadata["category"] == "human/reference"
        assert metadata["tags"] == ["curated", "trusted"]
        assert visible["category"] == "human/reference"
        assert visible["tags"] == ["curated", "trusted"]
        assert timeline["published"] == {"value": None, "source": None}
        assert timeline["modified"]["source"] == "filesystem:mtime"
        assert timeline["indexed"]["source"] == "grimore:index"
        assert timeline["verified"] == {"value": None, "source": None}
