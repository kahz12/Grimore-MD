"""Upgrading optional layout refreshes persisted passages on both scan paths.

Only model output/ranking is controlled. Real parsing, SQLite, chunks, vector
cache, sidecar preservation and evidence mapping run on an unchanged PDF.
"""
from __future__ import annotations

import shutil
import threading
from unittest.mock import MagicMock

import frontmatter
import pytest
from typer.testing import CliRunner

from eval.build_difficult_pdf import DEFAULT_OUTPUT
from grimore.cognition.embedder import Embedder
from grimore.cognition.oracle import Oracle
from grimore.daemon import GrimoreDaemon
from grimore.ingest import pdf_layout
from grimore.ingest.extraction_profile import pdf_extraction_profile
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.utils.config import Config
from grimore.utils.paths import sidecar_path_for
from grimore.utils.security import SecurityGuard


@pytest.mark.parametrize("surface", ["scan", "daemon"])
def test_old_pdfplumber_layout_profile_reextracts_without_modifying_original(
    tmp_path, monkeypatch, surface,
):
    pytest.importorskip("pdfplumber")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "difficult.pdf"
    shutil.copyfile(DEFAULT_OUTPUT, source)
    original_bytes, original_mtime = source.read_bytes(), source.stat().st_mtime_ns
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run, config.output.auto_commit = False, False
    config.ingest.pdf_engine, config.ingest.ocr = "pdfplumber", False
    config.cognition.hybrid_search = False
    parser = MagicMock(wraps=MarkdownParser())
    tagger = MagicMock()
    tagger.tag_note.return_value = {"tags": ["control"], "summary": "Controlled output", "category": ""}
    corrected_extract = pdf_layout.extract_page
    monkeypatch.setattr(pdf_layout, "extract_page", lambda _page, **kw: kw["baseline_text"])

    with Database(config.memory.db_path) as db:
        embedder = Embedder(config, cache=db)
        embedder.session.close()
        embedder.session = MagicMock()

        def vectors(*_args, **kwargs):
            response = MagicMock()
            response.json.return_value = {"embeddings": [[1.0, 0.0] for _ in kwargs["json"]["input"]]}
            return response

        embedder.session.post.side_effect = vectors
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.MarkdownParser", lambda: parser)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)

            def process(*, dry_run=False):
                result = CliRunner().invoke(app, ["scan", "--dry-run" if dry_run else "--no-dry-run"])
                assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = parser, SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer = FrontmatterWriter()
            daemon.tagger, daemon.embedder = tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0

            def process(*, dry_run=False):
                config.output.dry_run = dry_run
                daemon.process_file(source)

        process()
        note_id = db.get_note_id_by_path(str(source))
        assert note_id is not None
        old_chunks = db.get_chunk_records(note_id)
        sidecar = sidecar_path_for(source, vault, config.vault.sidecar_dir)
        annotation = frontmatter.load(sidecar)
        annotation.content += "\n\nPersonal annotation: preserve this."
        sidecar.write_text(frontmatter.dumps(annotation), encoding="utf-8")
        with db._get_connection() as conn:
            conn.execute("UPDATE notes SET extraction_profile = ? WHERE id = ?",
                         ("pdf-layout-v1:engine=pdfplumber:ocr=0", note_id))
        monkeypatch.setattr(pdf_layout, "extract_page", corrected_extract)
        process(dry_run=True)
        assert db.get_chunk_records(note_id) == old_chunks
        assert db.get_ingest_state(str(source))[4].startswith("pdf-layout-v1:")

        process()
        records = db.get_evidence_records(key[0] for key in db.get_embedding_keys())
        table = [row["full_text"] for row in records.values() if row["page"] == 2]
        assert any("Cooperativa Verde Vivo\tRetiro de jacinto y revegetación" in text for text in table)
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)
        assert db.get_chunk_records(note_id) != old_chunks
        assert "Personal annotation: preserve this." in frontmatter.load(sidecar).content
        assert frontmatter.load(sidecar)["extraction"]["issues"][0]["page"] == 3

        oracle = Oracle.__new__(Oracle)
        oracle.config, oracle.db, oracle.context_max_chars = config, db, 32_000
        oracle.embedder = MagicMock()
        oracle.embedder.embed.return_value = [1.0, 0.0]
        oracle.connector = MagicMock()
        oracle.connector.find_similar_notes.return_value = [
            {"embedding_id": row["embedding_id"], "note_id": row["note_id"],
             "text": row["excerpt"], "score": .99} for row in records.values()
        ]
        _context, _labels, _hits, details = oracle._build_context_payload("control", len(records))
        assert {detail["locator"]["page"] for detail in details} == {1, 2}
        assert any("Cooperativa Verde Vivo\tRetiro de jacinto y revegetación" in detail["passage"]["text"]
                   and detail["label"].endswith("#p.2") for detail in details)
        assert all(detail["source_state"] == "current" for detail in details)

        calls = (parser.parse_file.call_count, tagger.tag_note.call_count, embedder.session.post.call_count)
        process()
        assert calls == (parser.parse_file.call_count, tagger.tag_note.call_count, embedder.session.post.call_count)
        assert source.read_bytes() == original_bytes
        assert source.stat().st_mtime_ns == original_mtime
