"""MCP server (stdio JSON-RPC, read-only RAG tools).

Tests drive the server in-process: we build an :class:`MCPServer` with
a ``MagicMock`` ``Session`` so no Ollama / disk is required, then push
requests through :meth:`MCPServer.handle_request` (which is what
:meth:`serve` calls per stdin line). The transport itself is also
exercised via :meth:`MCPServer.serve` with ``io.StringIO`` pipes so we
confirm the NDJSON framing.

Tests cover:

* the initialize / tools/list handshake;
* each tool's contract (input validation + output shape);
* JSON-RPC error codes for parse errors, unknown methods, unknown
  tools, and missing required params;
* notifications produce no response (per spec).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock


from grimore.mcp_server import MCPServer


# ── helpers ─────────────────────────────────────────────────────────────


def _stub_session(*, hits=None, ask_answer="hello", chunk_vectors=None,
                  note_title="Gothic", note_path="/vault/gothic.md"):
    """Build a MagicMock Session that satisfies every tool handler.

    Keeping the mock here means each test can override one piece without
    having to set up the rest. ``hits`` plays both connector roles
    (find_hybrid + find_similar_notes); when None, both return [].
    """
    session = MagicMock()

    # Derived from note_path so the note is always inside the vault and
    # get_note's containment check passes for the stub by construction.
    session.vault_root = Path(note_path).parent

    # Config knobs the search tool reads via getattr.
    session.config.cognition.hybrid_search = True

    # Embedder returns a deterministic 4-vector.
    session.embedder.embed.return_value = [1.0, 0.0, 0.0, 0.0]

    # Oracle: ask and connector behaviour.
    session.oracle.ask.return_value = {
        "answer": ask_answer,
        "sources": ["Gothic#p.42"],
        "dropped_citations": 0,
    }
    session.oracle.connector.find_hybrid.return_value = hits or []
    session.oracle.connector.find_similar_notes.return_value = hits or []

    # DB: title lookup + category frequency.
    session.db.fts_available = True
    session.db.get_note_title.return_value = note_title
    session.db.get_note_location.return_value = (note_path, note_title)
    session.db.get_category_frequency.return_value = [
        ("philosophy", 5), ("tech/networking", 2),
    ]

    # Connection mock for the title-lookup + connect paths.
    conn = MagicMock()
    if chunk_vectors:
        from grimore.cognition.embedder import Embedder
        rows = [(Embedder.serialize_vector(v),) for v in chunk_vectors]
    else:
        rows = []
    conn.execute.return_value.fetchall.return_value = rows
    conn.execute.return_value.fetchone.return_value = (7, note_path, note_title)
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    session.db._get_connection.return_value = conn

    return session


def _server(session) -> MCPServer:
    return MCPServer(session=session)


def _request(method, *, req_id=1, params=None) -> dict:
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ── handshake + discovery ───────────────────────────────────────────────


class TestHandshake:
    def test_initialize_returns_server_info(self):
        srv = _server(_stub_session())
        resp = srv.handle_request(_request("initialize"))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["serverInfo"]["name"] == "grimore"
        assert "protocolVersion" in result
        assert "tools" in result["capabilities"]

    def test_tools_list_advertises_all_tools(self):
        srv = _server(_stub_session())
        resp = srv.handle_request(_request("tools/list"))
        names = [t["name"] for t in resp["result"]["tools"]]
        assert set(names) == {
            "grimore_ask",
            "grimore_search",
            "grimore_get_note",
            "grimore_connect",
            "grimore_list_categories",
        }
        # Every tool advertises a JSON schema and a description.
        for t in resp["result"]["tools"]:
            assert isinstance(t["inputSchema"], dict)
            assert t["inputSchema"]["type"] == "object"
            assert t["description"]

    def test_initialized_notification_produces_no_response(self):
        srv = _server(_stub_session())
        # No `id` field → JSON-RPC notification.
        resp = srv.handle_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert resp is None

    def test_unknown_method_returns_method_not_found(self):
        srv = _server(_stub_session())
        resp = srv.handle_request(_request("does_not_exist"))
        assert resp["error"]["code"] == -32601

    def test_ping_returns_empty_result(self):
        srv = _server(_stub_session())
        resp = srv.handle_request(_request("ping"))
        assert resp["result"] == {}


# ── tool wire shape ─────────────────────────────────────────────────────


def _call(srv, name, arguments):
    return srv.handle_request(_request(
        "tools/call", params={"name": name, "arguments": arguments},
    ))


def _content_payload(resp) -> dict:
    """Parse the inner content[0].text JSON back into a dict."""
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)


class TestAskTool:
    def test_returns_answer_sources_and_drop_count(self):
        srv = _server(_stub_session(ask_answer="The answer."))
        resp = _call(srv, "grimore_ask", {"question": "What is X?"})
        body = _content_payload(resp)
        assert body == {
            "answer": "The answer.",
            "sources": ["Gothic#p.42"],
            "dropped_citations": 0,
        }
        srv.session.oracle.ask.assert_called_once_with("What is X?", top_k=5)

    def test_top_k_is_forwarded(self):
        srv = _server(_stub_session())
        _call(srv, "grimore_ask", {"question": "q?", "top_k": 12})
        srv.session.oracle.ask.assert_called_once_with("q?", top_k=12)

    def test_top_k_is_clamped_to_schema_maximum(self):
        # The advertised input-schema maximum is 25; an over-large value is
        # clamped instead of being passed through (audit L2).
        srv = _server(_stub_session())
        _call(srv, "grimore_ask", {"question": "q?", "top_k": 9999})
        srv.session.oracle.ask.assert_called_once_with("q?", top_k=25)

    def test_non_numeric_top_k_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_ask", {"question": "q?", "top_k": "lots"})
        assert resp["error"]["code"] == -32602

    def test_missing_question_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_ask", {})
        assert resp["error"]["code"] == -32602

    def test_empty_question_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_ask", {"question": "   "})
        assert resp["error"]["code"] == -32602


class TestSearchTool:
    def test_hybrid_path_returns_normalised_hits(self):
        hits = [
            {"note_id": 3, "text": "a" * 250, "score": 0.91},
            {"note_id": 4, "text": "short", "score": 0.85},
        ]
        srv = _server(_stub_session(hits=hits))
        resp = _call(srv, "grimore_search", {"query": "anything", "top_k": 2})
        rows = _content_payload(resp)["hits"]
        assert [r["note_id"] for r in rows] == [3, 4]
        # First snippet got truncated with the ellipsis sentinel.
        assert rows[0]["snippet"].endswith("…") and len(rows[0]["snippet"]) <= 201
        assert rows[1]["snippet"] == "short"
        assert rows[0]["title"] == "Gothic"

    def test_falls_back_to_dense_when_fts_unavailable(self):
        sess = _stub_session(hits=[{"note_id": 1, "text": "x", "score": 0.5}])
        sess.db.fts_available = False
        srv = _server(sess)
        _call(srv, "grimore_search", {"query": "x"})
        sess.oracle.connector.find_similar_notes.assert_called_once()
        sess.oracle.connector.find_hybrid.assert_not_called()

    def test_empty_query_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_search", {"query": ""})
        assert resp["error"]["code"] == -32602

    def test_top_k_is_clamped_to_schema_maximum(self):
        # Search advertises a maximum of 50; a huge value is clamped before
        # it reaches the connector (audit L2).
        sess = _stub_session(hits=[])
        srv = _server(sess)
        _call(srv, "grimore_search", {"query": "x", "top_k": 9999})
        kwargs = sess.oracle.connector.find_hybrid.call_args.kwargs
        assert kwargs["top_k"] == 50

    def test_non_numeric_top_k_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_search", {"query": "x", "top_k": "lots"})
        assert resp["error"]["code"] == -32602


class TestGetNoteTool:
    def test_by_id_returns_metadata_and_body(self, tmp_path):
        note_file = tmp_path / "gothic.md"
        note_file.write_text("# Gothic\n\nPointed arches.\n", encoding="utf-8")
        sess = _stub_session(note_path=str(note_file))
        srv = _server(sess)
        resp = _call(srv, "grimore_get_note", {"note_id": 7})
        body = _content_payload(resp)
        assert body["found"] is True
        assert body["note_id"] == 7
        assert body["title"] == "Gothic"
        assert "Pointed arches." in body["body"]

    def test_by_title_returns_metadata_and_body(self, tmp_path):
        note_file = tmp_path / "gothic.md"
        note_file.write_text("body\n", encoding="utf-8")
        sess = _stub_session(note_path=str(note_file))
        # Title lookup hits the DB via _get_connection().execute().fetchone()
        srv = _server(sess)
        resp = _call(srv, "grimore_get_note", {"title": "Gothic"})
        body = _content_payload(resp)
        assert body["found"] is True

    def test_path_outside_vault_returns_not_found(self, tmp_path):
        # A DB row whose path escapes the vault (tampered index, or a
        # symlink swapped after scan) must not become an arbitrary-file
        # read for the MCP client — it reads as a plain miss.
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        sess = _stub_session(note_path=str(secret))
        sess.vault_root = tmp_path / "vault"
        srv = _server(sess)
        resp = _call(srv, "grimore_get_note", {"note_id": 7})
        assert _content_payload(resp) == {"found": False}

    def test_missing_id_returns_not_found(self):
        sess = _stub_session()
        sess.db.get_note_location.return_value = None
        srv = _server(sess)
        resp = _call(srv, "grimore_get_note", {"note_id": 999})
        assert _content_payload(resp) == {"found": False}

    def test_no_args_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_get_note", {})
        assert resp["error"]["code"] == -32602


class TestConnectTool:
    def test_averages_chunk_vectors_then_searches(self):
        sess = _stub_session(
            chunk_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            hits=[
                {"note_id": 4, "text": "x", "score": 0.7},
                {"note_id": 5, "text": "y", "score": 0.6},
            ],
        )
        srv = _server(sess)
        resp = _call(srv, "grimore_connect", {"note_id": 3, "top_k": 2})
        rows = _content_payload(resp)["hits"]
        assert [r["note_id"] for r in rows] == [4, 5]
        # The connector was called with exclude=3 + dedupe enabled.
        kwargs = sess.oracle.connector.find_similar_notes.call_args.kwargs
        assert kwargs["exclude_note_id"] == 3
        assert kwargs["dedupe_by_note"] is True

    def test_no_chunks_returns_empty_hits(self):
        srv = _server(_stub_session(chunk_vectors=[]))
        resp = _call(srv, "grimore_connect", {"note_id": 1})
        assert _content_payload(resp) == {"hits": []}

    def test_top_k_is_clamped_to_schema_maximum(self):
        # Connect advertises a maximum of 25 (audit L2).
        sess = _stub_session(
            chunk_vectors=[[1.0, 0.0, 0.0, 0.0]],
            hits=[],
        )
        srv = _server(sess)
        _call(srv, "grimore_connect", {"note_id": 3, "top_k": 9999})
        kwargs = sess.oracle.connector.find_similar_notes.call_args.kwargs
        assert kwargs["top_k"] == 25

    def test_non_numeric_top_k_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_connect", {"note_id": 3, "top_k": "lots"})
        assert resp["error"]["code"] == -32602

    def test_missing_note_id_returns_invalid_params(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_connect", {})
        assert resp["error"]["code"] == -32602


class TestListCategoriesTool:
    def test_returns_counts(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_list_categories", {})
        body = _content_payload(resp)
        assert body == {
            "categories": [
                {"category": "philosophy", "count": 5},
                {"category": "tech/networking", "count": 2},
            ]
        }


# ── error handling ──────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_tool_returns_method_not_found(self):
        srv = _server(_stub_session())
        resp = _call(srv, "grimore_obliterate_universe", {})
        assert resp["error"]["code"] == -32601

    def test_handler_exception_returned_as_isError_payload(self):
        # Force a non-ValueError exception inside a handler — should be
        # wrapped in an MCP "isError: true" payload, not a JSON-RPC error.
        sess = _stub_session()
        sess.oracle.ask.side_effect = RuntimeError("model exploded")
        srv = _server(sess)
        resp = _call(srv, "grimore_ask", {"question": "q?"})
        assert resp["result"]["isError"] is True
        assert "model exploded" in resp["result"]["content"][0]["text"]


# ── stdio transport ─────────────────────────────────────────────────────


class TestTransport:
    def test_serve_processes_ndjson_requests(self):
        srv = _server(_stub_session())
        stdin = io.StringIO(
            json.dumps(_request("initialize")) + "\n"
            + json.dumps(_request("tools/list", req_id=2)) + "\n"
        )
        stdout = io.StringIO()
        srv.serve(stdin=stdin, stdout=stdout)
        lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["id"] == 1 and first["result"]["serverInfo"]["name"] == "grimore"
        second = json.loads(lines[1])
        assert second["id"] == 2 and "tools" in second["result"]

    def test_serve_emits_parse_error_for_invalid_json(self):
        srv = _server(_stub_session())
        stdin = io.StringIO("{not json\n")
        stdout = io.StringIO()
        srv.serve(stdin=stdin, stdout=stdout)
        line = stdout.getvalue().splitlines()[0]
        assert json.loads(line)["error"]["code"] == -32700

    def test_serve_skips_blank_lines(self):
        srv = _server(_stub_session())
        stdin = io.StringIO("\n\n" + json.dumps(_request("ping")) + "\n")
        stdout = io.StringIO()
        srv.serve(stdin=stdin, stdout=stdout)
        out_lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        assert len(out_lines) == 1
        assert json.loads(out_lines[0])["result"] == {}

    def test_serve_swallows_notification_silently(self):
        srv = _server(_stub_session())
        stdin = io.StringIO(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n" + json.dumps(_request("ping", req_id=99)) + "\n")
        stdout = io.StringIO()
        srv.serve(stdin=stdin, stdout=stdout)
        out_lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        # Only the ping should produce a response.
        assert len(out_lines) == 1
        assert json.loads(out_lines[0])["id"] == 99
