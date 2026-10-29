"""Structured source evidence and indexed-version state detection."""
from __future__ import annotations

import sqlite3
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from grimore.cognition.embedder import Embedder
from grimore.cognition.chunker import build_candidate_chunks
from grimore.cognition.connector import Connector
from grimore.cognition.oracle import Oracle, _indexed_extraction_summary
from grimore.cognition.reembed import reembed_note
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.ingest.extraction_profile import pdf_extraction_profile
from grimore.memory.db import Database
from grimore.utils.config import Config
from grimore.utils.hashing import sha256_file


def _config(vault: Path):
    return SimpleNamespace(
        vault=SimpleNamespace(path=str(vault)),
        cognition=SimpleNamespace(
            hybrid_search=False,
            context_max_chars=16_000,
            conditional_rewrite=True,
            rrf_k=60,
            rerank=False,
            rerank_pool=20,
        ),
        ingest=SimpleNamespace(
            chunker="markdown", chunk_max_chars=1_500,
            semantic_threshold=0.55, pdf_engine="pypdf", ocr=False,
            sniff_magic=False,
        ),
    )


def _oracle_for_source(tmp_path: Path, *, persist_full: bool, fmt: str = "md",
                       excerpt_chars: int = 100):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / f"note.{fmt}"
    full_text = "Evidence begins. " + "complete indexed passage " * 35 + "END-MARKER"
    source.write_text(full_text, encoding="utf-8")
    if fmt == "md":
        # What the chunker stores for this note, title breadcrumb included.
        note = MarkdownParser().parse_file(source, vault_root=vault, config=_config(vault))
        full_text = build_candidate_chunks(note, note.content, None, _config(vault))[0].text

    db = Database(str(tmp_path / "evidence.db"))
    stat = source.stat()
    file_hash = sha256_file(source)
    note_id = db.upsert_note(
        str(source), "Evidence Note", "content-v1", format=fmt,
        file_hash=file_hash, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
    )
    assert note_id is not None
    excerpt = full_text[:excerpt_chars]
    model = "stub-embed"
    db.store_embedding(
        note_id, 0, excerpt, b"\x00" * 8,
        chunk_hash=Embedder.chunk_hash(full_text, model),
        full_text_content=full_text if persist_full else None,
    )
    embedding_id = db.get_embedding_keys()[0][0]

    oracle = Oracle.__new__(Oracle)
    oracle.config = _config(vault)
    oracle.db = db
    oracle.router = MagicMock()
    oracle.embedder = MagicMock()
    oracle.embedder.model = model
    oracle.embedder.embed.return_value = [1.0, 0.0]
    oracle.connector = MagicMock()
    oracle.connector.find_similar_notes.return_value = [{
        "embedding_id": embedding_id,
        "note_id": note_id,
        "text": excerpt,
        "score": 0.99,
    }]
    oracle.context_max_chars = 16_000
    oracle.conditional_rewrite = True
    oracle.rewrite_timeout_s = 60
    oracle.system_prompt_template = "{context}"
    return oracle, db, source, full_text


@pytest.mark.parametrize("locator", ["page", "heading", "unknown"])
@pytest.mark.parametrize("collision", ["identical", "prefix"])
@pytest.mark.parametrize("citation_form", ["wikilink", "short"])
def test_repeated_excerpts_cite_the_retrieved_chunk_not_the_first_text_match(
    tmp_path, locator, collision, citation_form,
):
    oracle, db, source, first_full = _oracle_for_source(
        tmp_path, persist_full=True, fmt="docx" if locator == "heading" else "pdf",
        excerpt_chars=500,
    )
    try:
        note_id = db.get_note_id_by_path(str(source))
        first_anchor = (None, "First section") if locator == "heading" else (1, None)
        later_anchor = {
            "page": (3, None), "heading": (None, "Later section"), "unknown": (None, None),
        }[locator]
        db.update_chunk_anchors(note_id, 0, *first_anchor)
        later_full = first_full if collision == "identical" else first_full[:500] + " DIFFERENT LATER TAIL"
        db.store_embedding(
            note_id, 1, first_full[:500], Embedder.serialize_vector([1.0, 0.0]),
            page=later_anchor[0], heading=later_anchor[1],
            chunk_hash=Embedder.chunk_hash(later_full, oracle.embedder.model),
            full_text_content=later_full,
        )
        ids = [key[0] for key in db.get_embedding_keys()]
        records = db.get_evidence_records(ids)
        # Deliberately rank the later occurrence first. Both returned excerpts
        # are identical, so a (note_id, excerpt) lookup alone would select
        # page 1; the embedding_id is what identifies the right occurrence.
        oracle.connector.find_similar_notes.return_value = [
            {"embedding_id": embedding_id, "note_id": note_id,
             "text": first_full[:500], "score": 0.99}
            for embedding_id in reversed(ids)
        ]
        first_label = "Evidence Note#First section" if locator == "heading" else "Evidence Note#p.1"
        later_label = {
            "page": "Evidence Note#p.3", "heading": "Evidence Note#Later section",
            "unknown": "Evidence Note",
        }[locator]

        context, labels, _retrieved, details = oracle._build_context_payload("q", 2)

        assert labels == [later_label, first_label]
        assert f"--- Source: [[{later_label}]]" in context
        assert [detail["locator"]["chunk_index"] for detail in details] == [1, 0]
        for detail, embedding_id in zip(details, reversed(ids), strict=True):
            record = records[embedding_id]
            assert detail["locator"] == {
                "chunk_index": record["chunk_index"], "page": record["page"],
                "heading": record["heading"],
            }
            assert detail["passage"] == {"text": record["full_text"], "completeness": "full"}
            assert detail["source_state"] == "current"
        assert details[0]["label"] == later_label

        # The generated citation must survive validation in both response
        # surfaces, with the same label and structured page/section locator.
        answer = f"Controlled response [[{later_label}]]."
        raw_answer = (
            f"Controlled response [{details[0]['citation_ref']}]."
            if citation_form == "short" else answer
        )
        oracle.router.complete.return_value = {"answer": raw_answer, "status": "supported"}
        result = oracle.ask("q", top_k=2)
        assert result["answer"] == answer
        assert result["sources"] == labels
        assert result["source_details"] == details
        assert result["dropped_citations"] == 0
        selection = result["citation_selections"][0]
        if citation_form == "short" or locator != "unknown":
            assert selection["selection"] == "exact"
            assert selection["source_detail_indices"] == [0]
            assert details[selection["source_detail_indices"][0]]["id"] == details[0]["id"]
        else:
            assert selection["selection"] == "ambiguous"
            assert selection["source_detail_indices"] == [0, 1]
        oracle.router.complete_streaming.return_value = iter([raw_answer])
        done = list(oracle.ask_stream("q", top_k=2))[-1]
        assert done["sources"] == labels
        assert done["source_details"] == details
        assert done["dropped_citations"] == 0
        assert done["citation_selections"] == result["citation_selections"]
    finally:
        db.close()


def test_complete_passage_and_source_state_follow_indexed_version(tmp_path):
    oracle, db, source, full_text = _oracle_for_source(tmp_path, persist_full=True)
    try:
        context, labels, _retrieved, details = oracle._build_context_payload("q", 1)
        detail = details[0]
        assert labels == ["Evidence Note"]
        assert detail["document"]["path"] == "note.md"
        assert detail["locator"]["chunk_index"] == 0
        assert detail["passage"] == {
            "text": full_text, "completeness": "full",
        }
        assert detail["source_state"] == "current"
        assert "END-MARKER" in context

        source.write_text("This is a changed source.", encoding="utf-8")
        *_head, changed = oracle._build_context_payload("q", 1)
        assert changed[0]["source_state"] == "modified"
        # Evidence remains the exact indexed version, never the changed file.
        assert changed[0]["passage"]["text"] == full_text

        source.unlink()
        *_head, missing = oracle._build_context_payload("q", 1)
        assert missing[0]["source_state"] == "missing"
        assert missing[0]["passage"]["text"] == full_text
    finally:
        db.close()


def test_legacy_retrieval_without_embedding_id_keeps_text_anchor_fallback(tmp_path):
    oracle, db, source, full_text = _oracle_for_source(tmp_path, persist_full=True, fmt="pdf")
    try:
        note_id = db.get_note_id_by_path(str(source))
        db.update_chunk_anchors(note_id, 0, 2, None)
        oracle.connector.find_similar_notes.return_value[0].pop("embedding_id")
        _context, labels, _retrieved, details = oracle._build_context_payload("q", 1)
        assert labels == ["Evidence Note#p.2"]
        assert details[0]["locator"]["page"] == 2
        assert details[0]["passage"] == {"text": full_text[:100], "completeness": "excerpt"}
        assert details[0]["source_state"] == "unverified"
    finally:
        db.close()


def test_mismatched_embedding_id_never_attaches_another_documents_evidence(tmp_path):
    oracle, db, source, _full_text = _oracle_for_source(tmp_path, persist_full=True, fmt="pdf")
    try:
        note_id = db.get_note_id_by_path(str(source))
        db.update_chunk_anchors(note_id, 0, 1, None)
        other_id = db.upsert_note(str(source.parent / "other.pdf"), "Other", "other-hash", format="pdf")
        db.store_embedding(
            other_id, 0, "Other private evidence", Embedder.serialize_vector([1.0, 0.0]),
            page=9, full_text_content="Other private evidence must not be attached.",
        )
        other_embedding_id = db.get_embedding_keys()[-1][0]
        oracle.connector.find_similar_notes.return_value[0]["embedding_id"] = other_embedding_id
        context, labels, _retrieved, details = oracle._build_context_payload("q", 1)
        assert labels == ["Evidence Note#p.1"]
        assert "Other private evidence" not in context
        assert details[0]["document"]["note_id"] == note_id
        assert details[0]["source_state"] == "unverified"
    finally:
        db.close()


@pytest.mark.parametrize("engine", ["pypdf", "pdfplumber"])
def test_real_ocr_repeated_pages_keep_distinct_citations_through_retrieval(
    tmp_path, monkeypatch, engine,
):
    from eval.benchmark_extraction import _unavailable
    from eval.build_ocr_budget_pdf import DEFAULT_OUTPUT

    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = _unavailable(engine, True, "spa")
    if missing:
        pytest.skip(", ".join(missing))
    oracle, db, source, _full_text = _oracle_for_source(tmp_path, persist_full=True, fmt="pdf")
    try:
        shutil.copyfile(DEFAULT_OUTPUT, source)
        config = Config()
        config.vault.path = str(source.parent)
        config.cognition.hybrid_search = False
        config.ingest.pdf_engine = engine
        config.ingest.ocr = True
        config.ingest.ocr_language = "spa"
        note = MarkdownParser().parse_file(source, vault_root=source.parent, config=config)
        assert [section.page for section in note.sections] == [1, 2, 3]
        note_id = db.get_note_id_by_path(str(source))
        db.delete_note_embeddings(note_id)
        embedder = MagicMock()
        embedder.model = oracle.embedder.model
        embedder.serialize_vector.side_effect = Embedder.serialize_vector
        embedder.embed_batch.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        chunks = build_candidate_chunks(note, note.content, embedder, config)
        indexed = reembed_note(db, embedder, note_id, chunks)
        assert indexed.stored == len(chunks)
        stat = source.stat()
        db.complete_note_indexing(
            note_id, note.content_hash, sha256_file(source), stat.st_size, stat.st_mtime_ns,
            extraction_profile=pdf_extraction_profile(config), extraction=extraction_diagnostics(note),
        )
        records = db.get_evidence_records(key[0] for key in db.get_embedding_keys())
        # Ensure this really exercises identical excerpts on separate pages,
        # rather than coincidentally producing a unique prefix per occurrence.
        assert len({row["excerpt"] for row in records.values()}) < len(records)
        oracle.config = config
        oracle.connector = Connector(db, oracle.embedder, vector_backend="numpy", matrix_cache_enabled=False)
        hybrid = MagicMock(wraps=oracle.connector.find_hybrid)
        oracle.connector.find_hybrid = hybrid
        for use_hybrid in ([False, True] if db.fts_available else [False]):
            config.cognition.hybrid_search = use_hybrid
            context, labels, _retrieved, details = oracle._build_context_payload("Fósforo", len(chunks))
            assert set(labels) == {f"Evidence Note#p.{page}" for page in (1, 2, 3)}
            assert len(details) == len(chunks)
            assert len({detail["id"] for detail in details}) == len(chunks)
            for detail in details:
                assert detail["label"] == f"Evidence Note#p.{detail['locator']['page']}"
                assert f"--- Source: [[{detail['label']}]]" in context
                assert detail["source_state"] == "current"
                assert detail["passage"]["completeness"] == "full"
                assert detail["indexed_extraction"]["status"] == "recovered"
        assert hybrid.call_count == int(db.fts_available)
    finally:
        db.close()


def test_legacy_excerpt_is_recovered_only_from_unchanged_source(tmp_path):
    oracle, db, _source, full_text = _oracle_for_source(tmp_path, persist_full=False)
    try:
        _context, _labels, _retrieved, details = oracle._build_context_payload("q", 1)
        assert details[0]["source_state"] == "current"
        assert details[0]["passage"] == {
            "text": full_text, "completeness": "full",
        }
    finally:
        db.close()


def test_oracle_exposes_indexed_pdf_gaps_in_context_and_source_details(tmp_path):
    oracle, db, source, _full_text = _oracle_for_source(
        tmp_path, persist_full=True, fmt="pdf",
    )
    try:
        note_id = db.get_note_id_by_path(str(source))
        db.update_chunk_anchors(note_id, 0, 1, None)
        stat = source.stat()
        db.complete_note_indexing(
            note_id, "content-v1", sha256_file(source),
            stat.st_size, stat.st_mtime_ns,
            extraction_profile="pdf-layout-v1:engine=pypdf:ocr=0",
            extraction={
                "status": "partial",
                "issues": [{"code": "page_text_unavailable", "page": 2,
                            "engine": "pypdf", "recovered": False}],
            },
        )
        oracle.router.complete.return_value = {
            "answer": "The passage supports this [[Evidence Note#p.1]].",
            "status": "supported",
        }

        result = oracle.ask("What does it say?", top_k=1)

        assert result["answer_status"] == "supported"
        assert result["source_details"][0]["indexed_extraction"] == {
            "status": "partial", "missing_pages": [2], "missing_page_count": 1,
        }
        assert "PDF extraction warning" in oracle.router.complete.call_args.kwargs["system_prompt"]
        assert "whole PDF" in oracle.router.complete.call_args.kwargs["system_prompt"]

        oracle.router.complete_streaming.return_value = iter([
            "The passage supports this [[Evidence Note#p.1]].",
        ])
        done = list(oracle.ask_stream("What does it say?", top_k=1))[-1]
        assert done["source_details"][0]["indexed_extraction"]["status"] == "partial"

        with db._get_connection() as conn:
            conn.execute("UPDATE notes SET extraction_json = NULL WHERE id = ?", (note_id,))
        context, _labels, _retrieved, details = oracle._build_context_payload("q", 1)
        assert details[0]["indexed_extraction"]["status"] == "unknown"
        assert "coverage was not recorded" in context
    finally:
        db.close()


def test_indexed_pdf_page_list_is_bounded_but_count_is_preserved():
    summary = _indexed_extraction_summary({
        "format": "pdf",
        "extraction": {
            "status": "partial",
            "issues": [
                {"code": "page_text_unavailable", "page": page, "recovered": False}
                for page in range(1, 31)
            ],
        },
    })

    assert summary["missing_pages"] == list(range(1, 25))
    assert summary["missing_page_count"] == 30


def test_existing_database_migrates_full_passage_column(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, "
        "title TEXT, content_hash TEXT, last_seen DATETIME, last_tagged DATETIME)"
    )
    conn.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, note_id INTEGER, "
        "chunk_index INTEGER, text_content TEXT, vector BLOB)"
    )
    conn.commit()
    conn.close()

    db = Database(str(path))
    try:
        with db._get_connection() as migrated:
            columns = {
                row[1] for row in migrated.execute("PRAGMA table_info(embeddings)")
            }
        assert "full_text_content" in columns
    finally:
        db.close()
