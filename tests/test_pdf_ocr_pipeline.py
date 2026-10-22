"""Real optional OCR reaches persisted chunks, sidecars and citation evidence.

Only model responses and retrieval ranking are stand-ins. Parsing, OCR,
chunking, vector caching, SQLite and sidecar writes use production code.
"""
from __future__ import annotations

import shutil
import threading
from unittest.mock import MagicMock

import frontmatter
import pytest
from typer.testing import CliRunner

from eval.benchmark_extraction import _unavailable
from eval.build_mixed_pdf import DEFAULT_OUTPUT
from grimore.cognition.embedder import Embedder
from grimore.cognition.oracle import Oracle
from grimore.daemon import GrimoreDaemon
from grimore.ingest.adapters import pdf
from grimore.ingest.extraction_profile import pdf_extraction_profile
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.utils.config import Config
from grimore.utils.hashing import sha256_file
from grimore.utils.paths import sidecar_path_for
from grimore.utils.security import SecurityGuard


def _evidence(db, config):
    records = db.get_evidence_records(key[0] for key in db.get_embedding_keys())
    oracle = Oracle.__new__(Oracle)
    oracle.config, oracle.db = config, db
    oracle.context_max_chars = 32_000
    oracle.embedder = MagicMock()
    oracle.embedder.embed.return_value = [1.0, 0.0]
    oracle.connector = MagicMock()
    oracle.connector.find_similar_notes.return_value = [
        {"embedding_id": row["embedding_id"], "note_id": row["note_id"],
         "text": row["excerpt"], "score": 0.99}
        for row in records.values()
    ]
    return oracle._build_context_payload("control", len(records))


@pytest.mark.parametrize("surface", ["scan", "daemon"])
@pytest.mark.parametrize("engine", ["pypdf", "pdfplumber"])
@pytest.mark.parametrize("transition", ["enable", "retry"])
def test_real_ocr_survives_the_complete_indexing_pipeline(
    tmp_path, monkeypatch, surface, engine, transition,
):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    missing = _unavailable(engine, True, "spa")
    if missing:
        pytest.skip(", ".join(missing))

    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "mixed.pdf"
    shutil.copyfile(DEFAULT_OUTPUT, source)
    original_bytes, original_stat = source.read_bytes(), source.stat()
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False
    config.ingest.pdf_engine = engine
    config.ingest.ocr_language = "spa"
    config.ingest.ocr = transition == "retry"
    config.cognition.hybrid_search = False
    parser = MagicMock(wraps=MarkdownParser())
    tagger = MagicMock()
    tagger.tag_note.return_value = {
        "tags": ["control"], "summary": "Controlled model response.", "category": "",
    }
    real_ocr = pdf._ocr_page
    ocr = MagicMock(return_value=pdf._OcrResult("", "ocr_unavailable"))
    monkeypatch.setattr(pdf, "_ocr_page", ocr)

    with Database(config.memory.db_path) as db:
        # Keep the real cache path, including hits when page 3 changes chunk
        # index after the recovered page 2 is inserted or removed.
        embedder = Embedder(config, cache=db)
        embedder.session.close()
        embedder.session = MagicMock()

        def model_response(*_args, **kwargs):
            response = MagicMock()
            response.json.return_value = {
                "embeddings": [[1.0, 0.0] for _ in kwargs["json"]["input"]],
            }
            return response

        embedder.session.post.side_effect = model_response
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **kw: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.MarkdownParser", lambda: parser)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)

            def process():
                result = CliRunner().invoke(app, ["scan", "--no-dry-run"])
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

            def process():
                daemon.process_file(source)

        process()
        note_id = db.get_note_id_by_path(str(source))
        assert note_id is not None
        partial_chunks = db.get_chunk_records(note_id)
        assert {row[1] for row in partial_chunks.values()} == {1, 3}
        sidecar = sidecar_path_for(source, vault, config.vault.sidecar_dir)
        post = frontmatter.load(sidecar)
        assert post["extraction"]["status"] == "partial"
        assert post["extraction"]["issues"][0]["page"] == 2
        if transition == "retry":
            assert db.get_ingest_state(str(source))[4].endswith(":retry-ocr")
            ocr.assert_called_once_with(source, 2, config.ingest.ocr_timeout_s, "spa")
        else:
            ocr.assert_not_called()
        context, _labels, _retrieved, details = _evidence(db, config)
        assert "PDF extraction warning" in context
        assert all(detail["indexed_extraction"]["missing_pages"] == [2]
                   for detail in details)

        # A reviewer's sidecar notes must survive the genuine content change.
        post.content += "\n\nPersonal review note: keep this annotation."
        sidecar.write_text(frontmatter.dumps(post), encoding="utf-8")
        config.ingest.ocr = True
        ocr = MagicMock(wraps=real_ocr)
        monkeypatch.setattr(pdf, "_ocr_page", ocr)
        embedder.session.post.reset_mock()
        process()

        ocr.assert_called_once_with(source, 2, config.ingest.ocr_timeout_s, "spa")
        recovered_chunks = db.get_chunk_records(note_id)
        assert {row[1] for row in recovered_chunks.values()} == {1, 2, 3}
        assert db.get_ingest_state(str(source))[4] == pdf_extraction_profile(config)
        for page in (1, 3):
            assert {row[0] for row in partial_chunks.values() if row[1] == page} == {
                row[0] for row in recovered_chunks.values() if row[1] == page
            }
        records = db.get_evidence_records(key[0] for key in db.get_embedding_keys())
        recovered_texts = {row["full_text"] for row in records.values() if row["page"] == 2}
        sent_texts = [text for call in embedder.session.post.call_args_list
                      for text in call.kwargs["json"]["input"]]
        assert sent_texts and set(sent_texts) == recovered_texts
        post = frontmatter.load(sidecar)
        assert post["extraction"] == {
            "status": "recovered", "issues": [{
                "code": "ocr_used", "page": 2, "engine": engine,
                "ocr_requested": True, "recovered": True,
            }],
        }
        assert "Personal review note: keep this annotation." in post.content
        context, labels, _retrieved, details = _evidence(db, config)
        assert "PDF extraction warning" not in context
        assert {detail["locator"]["page"] for detail in details} == {1, 2, 3}
        for detail in details:
            assert detail["label"].endswith(f"#p.{detail['locator']['page']}")
            assert detail["label"] in labels
            assert detail["source_state"] == "current"
            assert detail["passage"]["completeness"] == "full"
            assert detail["indexed_extraction"] == {
                "status": "recovered", "missing_pages": [], "missing_page_count": 0,
            }
        assert any("37,5" in detail["passage"]["text"]
                   for detail in details if detail["locator"]["page"] == 2)

        # Stable bytes/profile skip before parsing, recognition or model work.
        calls = (parser.parse_file.call_count, tagger.tag_note.call_count,
                 embedder.session.post.call_count)
        sidecar_bytes = sidecar.read_bytes()
        process()
        assert calls == (parser.parse_file.call_count, tagger.tag_note.call_count,
                         embedder.session.post.call_count)
        assert db.get_chunk_records(note_id) == recovered_chunks
        assert sidecar.read_bytes() == sidecar_bytes
        assert ocr.call_count == 1

        # Disabling OCR removes obsolete recovered passages, not digital pages.
        config.ingest.ocr = False
        embedder.session.post.reset_mock()
        process()
        assert db.get_chunk_records(note_id) == partial_chunks
        embedder.session.post.assert_not_called()
        assert frontmatter.load(sidecar)["extraction"]["status"] == "partial"
        context, _labels, _retrieved, details = _evidence(db, config)
        assert "PDF extraction warning" in context
        assert {detail["locator"]["page"] for detail in details} == {1, 3}
        assert "Personal review note: keep this annotation." in frontmatter.load(sidecar).content
        assert parser.parse_file.call_count == tagger.tag_note.call_count == 3
        assert ocr.call_count == 1
        assert source.read_bytes() == original_bytes
        assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert db.get_file_hash(str(source)) == sha256_file(source)
