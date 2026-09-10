"""Ingest previews must not mutate indexed state, including daemon deletions."""
import threading
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from grimore.daemon import GrimoreDaemon
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.utils.config import Config
from grimore.utils.hashing import sha256_file
from grimore.utils.security import SecurityGuard


@pytest.mark.parametrize("surface,event", [
    (surface, event)
    for surface in ("daemon", "scan")
    for event in ("new", "changed", "same_bytes", "same_text", "delete")
    if surface == "daemon" or event != "delete"
])
def test_dry_run_preserves_files_and_entire_database(tmp_path, monkeypatch, event, surface):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "note.md"
    source.write_text("# Note\n\nOriginal document.\n")
    config = Config()
    config.vault.path = str(vault)
    config.output.dry_run = True
    config.output.auto_commit = False
    with Database(str(tmp_path / "index.db")) as db:
        parsed = MarkdownParser().parse_file(source)
        if event != "new":
            note_id = db.upsert_note(
                str(source), parsed.title, parsed.content_hash,
                file_hash=sha256_file(source), size_bytes=-1, mtime_ns=-1,
            )
            db.store_embedding(note_id, 0, "old chunk", b"\x00" * 8)
        if event == "changed":
            source.write_text("# Note\n\nChanged document.\n")
        elif event == "same_text":
            source.write_text("---\ntags: [manual]\n---\n# Note\n\nOriginal document.\n")
        elif event == "delete":
            source.unlink()
        before_source = source.read_bytes() if source.exists() else None
        with db._get_connection() as conn:
            before_db = list(conn.iterdump())

        daemon = GrimoreDaemon.__new__(GrimoreDaemon)
        daemon.config, daemon.db, daemon.vault_root = config, db, vault
        daemon.parser = MarkdownParser()
        daemon.security = SecurityGuard(str(vault))
        daemon.writer = FrontmatterWriter()
        daemon.tagger = MagicMock()
        daemon.tagger.tag_note.return_value = {"tags": ["new"], "summary": "Preview", "category": ""}
        daemon.embedder = MagicMock()
        daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
        daemon._counter_lock = threading.Lock()
        daemon.processed_count = 0
        if surface == "scan":
            from grimore.cli import app
            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: daemon.embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: daemon.tagger)
            result = CliRunner().invoke(app, ["scan", "--dry-run"])
            assert result.exit_code == 0, result.output
        elif event == "delete":
            daemon.remove_file(source)
        else:
            daemon.process_file(source)

        with db._get_connection() as conn:
            assert list(conn.iterdump()) == before_db
        assert (source.read_bytes() if source.exists() else None) == before_source
        daemon.embedder.embed_batch.assert_not_called()
        assert daemon.processed_count == 0
