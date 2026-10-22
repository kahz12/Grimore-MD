"""Read-only source/review contracts and non-executing Obsidian links."""
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest

from grimore.cognition.embedder import Embedder
from grimore.cognition.oracle import Oracle
from grimore.integrations.obsidian import open_target
from grimore.mcp_server import MCPServer
from grimore.session import Session
from grimore.utils.config import Config, MemoryConfig, OutputConfig, VaultConfig
from grimore.utils.hashing import sha256_file
from tests.test_mcp_server import _call, _content_payload, _stub_session


@pytest.fixture
def indexed(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    source = root / "Nota á & ?.md"
    text = "# Evidencia\n\nPrimer pasaje.\n\nSegundo pasaje con una conclusión distinta."
    source.write_text(text, encoding="utf-8")
    session = Session(Config(
        vault=VaultConfig(path=str(root), ignored_dirs=[]),
        memory=MemoryConfig(db_path=str(tmp_path / "index.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
    ))
    db = session.db
    file_hash = sha256_file(source)
    note_id = db.upsert_note(str(source), "Evidence", "v1", file_hash=file_hash)
    for index, passage in enumerate(("Repeated preview. FIRST", "Repeated preview. SECOND")):
        db.store_embedding(
            note_id, index, "Repeated preview.", Embedder.serialize_vector([1.0, 0.0]),
            full_text_content=passage, heading="Evidencia & ?",
        )
    records = list(db.get_evidence_records(key[0] for key in db.get_embedding_keys()).values())
    oracle = Oracle.__new__(Oracle)
    oracle.db, oracle.config = db, session.config
    oracle.embedder, oracle.router, oracle.connector = MagicMock(), MagicMock(), MagicMock()
    oracle._recover_full_passage = MagicMock(side_effect=AssertionError("Must not reconstruct"))
    session._oracle, session._embedder = oracle, oracle.embedder
    oracle.embedder.embed.return_value = [1.0, 0.0]
    oracle.connector.find_hybrid.return_value = []
    oracle.connector.find_similar_notes.return_value = []
    yield SimpleNamespace(
        session=session, server=MCPServer(session), db=db, source=source,
        records=records, file_hash=file_hash,
    )
    db.close()


def _id(record):
    return MCPServer._record_source_id(record)


def _get(indexed, **extra):
    return _content_payload(_call(indexed.server, "grimore_get_source", {
        "source_id": _id(indexed.records[1]), **extra,
    }))


def test_library_identity_is_metadata_only_and_matches_local_protocol(indexed):
    canonical = os.path.normcase(str(indexed.session.vault_root.resolve()))
    before = indexed.db.list_review_decisions()
    body = _content_payload(_call(indexed.server, "grimore_library_info", {}))
    assert body["contract_version"] == 1
    assert body["vault_identity"] == {
        "algorithm": "sha256-local-path-v1",
        "fingerprint": sha256(("grimore-vault-v1\0" + canonical).encode("utf-8")).hexdigest(),
    }
    assert body["model_execution"]["embeddings"]["mode"] == "local"
    assert canonical not in str(body)
    assert indexed.session.oracle.router.mock_calls == []
    assert indexed.session.embedder.mock_calls == []
    assert indexed.db.list_review_decisions() == before


def test_library_identity_rejects_unknown_fields_without_database():
    session = _stub_session()
    response = _call(MCPServer(session), "grimore_library_info", {"path": "/secret"})
    assert response["error"]["code"] == -32602
    assert session.db.mock_calls == []


def test_get_source_exact_occurrence_and_no_models_or_writes(indexed):
    with indexed.db._get_connection() as conn:
        before = conn.total_changes
        payload = _get(indexed, expected_indexed_file_hash=indexed.file_hash,
                       obsidian_vault="Mi bóveda & trabajo")
        assert conn.total_changes == before
    detail = payload["source"]
    assert detail["passage"] == {"text": "Repeated preview. SECOND", "completeness": "full"}
    assert detail["locator"]["chunk_index"] == 1
    assert detail["source_state"] == "current"
    assert detail["document"]["path"] == indexed.source.name
    target = payload["open_target"]
    assert target["scope"] == "heading"
    assert parse_qs(urlsplit(target["uri"]).query) == {
        "vault": ["Mi bóveda & trabajo"],
        "file": [indexed.source.name + "#Evidencia & ?"],
    }
    assert "%20" in target["uri"] and "%26" in target["uri"] and "%23" in target["uri"]
    assert indexed.session.oracle.router.mock_calls == []
    assert indexed.session.embedder.mock_calls == []
    indexed.session.oracle._recover_full_passage.assert_not_called()
    assert sha256_file(indexed.source) == indexed.file_hash


def test_modified_source_preserves_indexed_passage_but_opens_document(indexed):
    indexed.source.write_text("New live text", encoding="utf-8")
    body = _get(indexed, obsidian_vault="vault")
    assert body["source"]["source_state"] == "modified"
    assert body["source"]["passage"]["text"] == "Repeated preview. SECOND"
    assert body["open_target"]["scope"] == "document"
    assert "indexed_heading_not_used_without_current_version" in body["open_target"]["limitations"]


def test_missing_file_still_returns_indexed_evidence_without_open_link(indexed):
    indexed.source.unlink()
    body = _get(indexed, obsidian_vault="vault")
    assert body["found"] is True
    assert body["source"]["source_state"] == "missing"
    assert body["source"]["passage"]["text"].endswith("SECOND")
    assert body["open_target"]["uri"] is None


def test_pdf_source_preserves_page_and_coverage_without_parsing(indexed, monkeypatch):
    with indexed.db._get_connection() as conn:
        conn.execute("UPDATE notes SET format = 'pdf', extraction_json = ?", (json.dumps({
            "status": "partial", "issues": [{"code": "page_text_unavailable", "page": 2}],
        }),))
        conn.execute("UPDATE embeddings SET page = 3, heading = NULL")
        conn.commit()
    parser = MagicMock(side_effect=AssertionError("Must not parse or OCR"))
    monkeypatch.setattr("grimore.mcp_server.MarkdownParser", parser)
    body = _get(indexed, obsidian_vault="vault")
    assert body["source"]["label"] == "Evidence#p.3"
    assert body["source"]["locator"]["page"] == 3
    assert body["source"]["indexed_extraction"]["status"] == "partial"
    assert body["source"]["indexed_extraction"]["missing_pages"] == [2]
    assert body["open_target"]["reason"] == "unsupported_format"
    parser.assert_not_called()


def test_stdio_discovery_and_source_delivery_remain_json_frames(indexed):
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "grimore_get_source", "arguments": {"source_id": _id(indexed.records[1])},
        }},
    ]
    output = io.StringIO()
    indexed.server.serve(stdin=io.StringIO("\n".join(json.dumps(r) for r in requests)), stdout=output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 2
    assert {"grimore_get_source", "grimore_review_inbox"} <= {
        tool["name"] for tool in responses[0]["result"]["tools"]
    }
    assert _content_payload(responses[1])["source"]["passage"]["text"].endswith("SECOND")


def test_legacy_record_is_explicit_excerpt_without_reconstruction(indexed):
    with indexed.db._get_connection() as conn:
        conn.execute("UPDATE embeddings SET full_text_content = NULL")
        conn.commit()
    body = _get(indexed)
    assert body["source"]["passage"] == {"text": "Repeated preview.", "completeness": "excerpt"}
    assert "open_target" not in body
    indexed.session.oracle._recover_full_passage.assert_not_called()
    assert indexed.session.embedder.mock_calls == []


def test_expected_indexed_version_rejects_reindex(indexed):
    indexed.db.upsert_note(str(indexed.source), "Evidence", "v2", file_hash="a" * 64)
    assert _get(indexed, expected_indexed_file_hash=indexed.file_hash) == {
        "found": False, "reason": "source_version_changed",
    }


@pytest.mark.parametrize("change", ["note", "chunk", "deleted"])
def test_stale_or_mismatched_id_never_falls_back(indexed, change):
    record = indexed.records[1]
    bad = dict(record)
    if change == "note":
        bad["note_id"] += 1
    elif change == "chunk":
        bad["chunk_index"] += 1
    else:
        with indexed.db._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE id = ?", (record["embedding_id"],))
            conn.commit()
    body = _content_payload(_call(indexed.server, "grimore_get_source", {"source_id": _id(bad)}))
    assert body == {"found": False, "reason": "source_not_found"}


def test_escaping_source_path_does_not_read_external_file(indexed, tmp_path, monkeypatch):
    secret = tmp_path / "private.md"
    secret.write_text("Private live body", encoding="utf-8")
    with indexed.db._get_connection() as conn:
        conn.execute("UPDATE notes SET path = ?", (str(secret),))
        conn.commit()
    hashing = MagicMock(side_effect=AssertionError("Must not hash external file"))
    monkeypatch.setattr("grimore.cognition.oracle.sha256_file", hashing)
    body = _get(indexed, obsidian_vault="vault")
    assert body["source"]["source_state"] == "unavailable"
    assert body["source"]["document"]["path"] is None
    assert "Private live body" not in str(body)
    assert body["open_target"]["uri"] is None
    hashing.assert_not_called()


@pytest.mark.parametrize("hybrid", [True, False])
def test_search_exports_occurrence_id_without_full_passages(indexed, hybrid):
    indexed.session.config.cognition.hybrid_search = hybrid
    hits = [{"embedding_id": record["embedding_id"], "note_id": record["note_id"],
             "text": "Repeated preview.", "score": 0.9} for record in reversed(indexed.records)]
    connector = indexed.session.oracle.connector
    connector.find_hybrid.return_value = hits
    connector.find_similar_notes.return_value = hits
    body = _content_payload(_call(indexed.server, "grimore_search", {"query": "evidence"}))
    assert [hit["source_id"] for hit in body["hits"]] == [_id(r) for r in reversed(indexed.records)]
    assert all(hit["indexed_file_hash"] == indexed.file_hash for hit in body["hits"])
    assert "SECOND" not in str(body) and "FIRST" not in str(body)
    retrieval = connector.find_hybrid if hybrid else connector.find_similar_notes
    assert retrieval.call_args.kwargs["include_embedding_id"] is True
    assert indexed.session.oracle.router.mock_calls == []


def test_search_does_not_export_foreign_note_source_id(indexed):
    record = indexed.records[1]
    indexed.session.oracle.connector.find_hybrid.return_value = [{
        "embedding_id": record["embedding_id"], "note_id": record["note_id"] + 1,
        "text": "preview", "score": 0.9,
    }]
    body = _content_payload(_call(indexed.server, "grimore_search", {"query": "q"}))
    assert body["hits"][0]["source_id"] is None
    assert body["hits"][0]["indexed_file_hash"] is None


@pytest.mark.parametrize("arguments", [
    {}, {"source_id": "note:1:unresolved"}, {"source_id": "pinned:Note"},
    {"source_id": "note:1:chunk:0:embedding:9999999999999999999"},
    {"source_id": "note:1:chunk:0:embedding:1\n"}, {"source_id": True},
    {"source_id": "note:1:chunk:0:embedding:1", "expected_indexed_file_hash": "not-a-hash"},
    {"source_id": "note:1:chunk:0:embedding:1", "obsidian_vault": " "},
    {"source_id": "note:1:chunk:0:embedding:1", "obsidian_vault": "x" * 161},
    {"source_id": "note:1:chunk:0:embedding:1", "obsidian_vault": "x\nnew"},
    {"source_id": "note:1:chunk:0:embedding:1", "path": "/etc/passwd"},
    [], None, "", False,
])
def test_invalid_source_inputs_are_rejected_before_db(arguments):
    session = _stub_session()
    response = _call(MCPServer(session), "grimore_get_source", arguments)
    assert response["error"]["code"] == -32602
    session.db.get_evidence_records.assert_not_called()


def _candidate(path, kind="review_due", index=0, text="Evidence"):
    return {
        "id": f"candidate:{index}", "kind": kind, "status": "candidate",
        "reason": "Needs human review", "priority": "low",
        "documents": [{"note_id": index + 1, "path": str(path), "title": "Note"}],
        "passages": [{"note_id": index + 1, "path": str(path), "text": text, "complete": True}],
        "coverage": {"status": "partial", "limitations": ["age_not_content_validity"]},
    }


def test_review_filter_pagination_and_delivery_caps(indexed, monkeypatch):
    candidates = [_candidate(indexed.source, index=i) for i in range(3)]
    candidates.append(_candidate(indexed.source, kind="contradiction", index=3))
    candidates[1]["reason"] = "r" * 2001
    candidates[1]["passages"] *= 21
    candidates[1]["documents"] *= 21
    for passage in candidates[1]["passages"]:
        passage["text"] = "t" * 3001
    original = deepcopy(candidates)
    builder = MagicMock(return_value={
        "generated_at": "2026-10-07T00:00:00Z", "status": "candidates_only",
        "counts": {"all": 4, "review_due": 3, "contradiction": 1},
        "items": candidates, "history": [{"private": "history snapshot"}],
    })
    monkeypatch.setattr("grimore.mcp_server.build_review_inbox", builder)
    body = _content_payload(_call(indexed.server, "grimore_review_inbox", {
        "kind": "review_due", "offset": 1, "limit": 1, "obsidian_vault": "vault",
    }))
    builder.assert_called_once_with(indexed.session, duplicate_limit=30)
    item = body["items"][0]
    assert item["id"] == "candidate:1"
    assert item["coverage"] == original[1]["coverage"]
    assert len(item["documents"]) == 20 and len(item["passages"]) == 20
    assert item["delivery"] == {
        "documents_truncated": True, "passages_truncated": True, "reason_truncated": True,
    }
    assert len(item["passages"][0]["text"]) == 3000
    assert item["passages"][0]["complete"] is False
    assert item["passages"][0]["text_truncated"] is True
    assert item["documents"][0]["path"] == indexed.source.name
    assert item["documents"][0]["open_target"]["scope"] == "document"
    assert body["delivery"]["matched"] == 3 and body["delivery"]["next_offset"] == 2
    assert body["delivery"]["has_more"] is True
    assert "history" not in body and "history snapshot" not in str(body)
    assert candidates == original


def test_real_review_inbox_does_not_write_or_call_models(indexed):
    duplicate = indexed.source.parent / "copy.md"
    duplicate.write_text("Copy", encoding="utf-8")
    indexed.db.upsert_note(str(duplicate), "Copy", "v1", file_hash=sha256_file(duplicate))
    with indexed.db._get_connection() as conn:
        before = conn.total_changes
        body = _content_payload(_call(indexed.server, "grimore_review_inbox", {}))
        assert conn.total_changes == before
    item = body["items"][0]
    assert item["kind"] == "duplicate_exact"
    assert item["passages"][0]["source_id"] == _id(indexed.records[0])
    assert indexed.session.oracle.router.mock_calls == []
    assert indexed.session.embedder.mock_calls == []
    assert indexed.db.list_review_decisions() == []
    indexed.db.record_review_decision(
        candidate_id=item["id"], kind=item["kind"], decision="dismiss",
        decided_at="2026-10-07T00:00:00Z", postpone_until=None, snapshot={},
    )
    assert _content_payload(_call(indexed.server, "grimore_review_inbox", {}))["items"] == []


def test_review_paths_outside_vault_or_missing_have_no_open_link(indexed, tmp_path, monkeypatch):
    secret = tmp_path / "external.md"
    candidates = [_candidate(secret), _candidate(indexed.source.parent / "missing.md", index=1)]
    monkeypatch.setattr("grimore.mcp_server.build_review_inbox", lambda *a, **kw: {
        "generated_at": "now", "status": "candidates_only", "counts": {"all": 2},
        "items": candidates,
    })
    body = _content_payload(_call(indexed.server, "grimore_review_inbox", {"obsidian_vault": "vault"}))
    assert body["items"][0]["documents"][0]["path"] is None
    assert all(item["documents"][0]["open_target"]["uri"] is None for item in body["items"])


@pytest.mark.parametrize("arguments", [
    {"limit": True}, {"limit": 0}, {"limit": 51}, {"limit": "20"}, {"offset": -1},
    {"offset": 1000001}, {"offset": False}, {"kind": "unknown"}, {"kind": None},
    {"decision": "confirm"}, {"include_inactive": True}, [], None,
])
def test_invalid_review_inputs_do_not_build_inbox(arguments, monkeypatch):
    builder = MagicMock()
    monkeypatch.setattr("grimore.mcp_server.build_review_inbox", builder)
    response = _call(MCPServer(_stub_session()), "grimore_review_inbox", arguments)
    assert response["error"]["code"] == -32602
    builder.assert_not_called()


def test_readonly_failures_do_not_expose_internal_paths(monkeypatch):
    builder = MagicMock(side_effect=RuntimeError("Private path /home/private/index.db"))
    monkeypatch.setattr("grimore.mcp_server.build_review_inbox", builder)
    response = _call(MCPServer(_stub_session()), "grimore_review_inbox", {})
    assert response["result"]["isError"] is True
    assert "/home/private" not in str(response)


@pytest.mark.parametrize("path", ["../bad.md", "/absolute.md", "a#b.md", "a\\b.md", "a\nb.md"])
def test_obsidian_rejects_ambiguous_or_unsafe_note_paths(path):
    assert open_target(vault="vault", path=path)["uri"] is None


@pytest.mark.parametrize("format", ["pdf", "docx", "html", "epub"])
def test_obsidian_does_not_promise_unsupported_format_or_page_navigation(format):
    target = open_target(vault="vault", path=f"note.{format}", format=format, source_state="current")
    assert target["uri"] is None and target["reason"] == "unsupported_format"


def test_obsidian_only_open_action_no_injected_parameters():
    target = open_target(vault="vault&append=true", path="a&overwrite=true.md", source_state="current")
    split = urlsplit(target["uri"])
    assert split.scheme == "obsidian" and split.netloc == "open"
    assert parse_qs(split.query) == {"vault": ["vault&append=true"], "file": ["a&overwrite=true.md"]}
