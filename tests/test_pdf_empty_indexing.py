"""A textless PDF stays visible as incomplete, without fabricated metadata."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import frontmatter
import pytest
from typer.testing import CliRunner

from grimore.cognition.embedder import Embedder
from grimore.daemon import GrimoreDaemon
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.utils.config import Config
from grimore.utils.paths import sidecar_path_for
from grimore.utils.security import SecurityGuard


SCAN = (
    Path(__file__).resolve().parents[1]
    / "test_vault" / "test_vault" / "informe_humedal_escaneado.pdf"
)


@pytest.mark.parametrize("surface", ["scan", "daemon"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_textless_pdf_is_recorded_without_model_generated_metadata(
    tmp_path, monkeypatch, surface, preexisting,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "scan.pdf"
    shutil.copyfile(SCAN, source)

    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    config.ingest.ocr = False

    tagger = MagicMock()
    tagger.tag_note.side_effect = AssertionError("empty PDF must not reach LLM")
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.embed_batch.side_effect = AssertionError("empty PDF has no chunks")

    with Database(config.memory.db_path) as db:
        if preexisting:
            old_id = db.upsert_note(
                str(source), "Old searchable title", "old-text",
                format="pdf", file_hash="old-bytes", size_bytes=1, mtime_ns=1,
            )
            db.store_embedding(
                old_id, 0, "stale passage", Embedder.serialize_vector([1.0, 0.0]),
                page=1,
            )
            db.upsert_tags(old_id, ["stale-tag"])
            db.set_note_category(old_id, "stale-category")
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            result = CliRunner().invoke(app, ["scan", "--no-dry-run"])
            assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = MarkdownParser(), SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer = FrontmatterWriter()
            daemon.tagger, daemon.embedder = tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0
            daemon.process_file(source)

        assert db.get_note_id_by_path(str(source)) is not None
        assert db.get_file_hash(str(source))
        assert db.embeddings_signature()[0] == 0
        assert db.get_notes_by_tag("stale-tag") == []

    tagger.tag_note.assert_not_called()
    embedder.embed_batch.assert_not_called()
    sidecar = sidecar_path_for(source, vault, config.vault.sidecar_dir)
    metadata = frontmatter.load(sidecar).metadata
    assert metadata["summary"] == ""
    assert metadata["tags"] == []
    assert metadata["extraction"]["status"] == "partial"
    assert metadata["extraction"]["issues"][0]["code"] == "page_text_unavailable"
