"""HTTP API + minimal web UI.

Tests drive the API through Starlette's ``TestClient`` against a
MagicMock Session — no Ollama, no disk required. Auto-skips when the
``serve`` extra (Starlette + httpx) isn't installed so the suite stays
green on default installs.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

starlette = pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402

from grimore.api.app import build_app  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────


def _stub_session(*, hits=None, ask_answer="hello",
                  note_title="Gothic", note_path="/vault/gothic.md"):
    session = MagicMock()
    session.config.cognition.hybrid_search = True
    session.vault_root = Path("/vault")

    session.embedder.embed.return_value = [1.0, 0.0]

    session.oracle.ask.return_value = {
        "answer": ask_answer,
        "sources": ["Gothic#p.42"],
        "dropped_citations": 0,
    }
    # Streaming variant: list of events, terminating with "done".
    session.oracle.ask_stream.return_value = iter([
        {"type": "token", "text": "Hel"},
        {"type": "token", "text": "lo"},
        {"type": "done", "sources": ["Gothic#p.42"], "dropped_citations": 0},
    ])
    session.oracle.connector.find_hybrid.return_value = hits or []
    session.oracle.connector.find_similar_notes.return_value = hits or []

    session.db.fts_available = True
    session.db.vec_available = False
    session.db.get_note_title.return_value = note_title
    session.db.get_note_location.return_value = (note_path, note_title)
    session.db.get_category_frequency.return_value = [
        ("philosophy", 5), ("tech/networking", 2),
    ]
    return session


@pytest.fixture
def session():
    return _stub_session()


@pytest.fixture
def client(session):
    app = build_app(session)
    return TestClient(app)


# ── health + UI ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_reports_version_and_vault(self, client, session):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "version" in data
        assert data["vault"] == "/vault"
        assert data["fts"] is True

    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "Grimore" in r.text


# ── ask ───────────────────────────────────────────────────────────────


class TestAsk:
    def test_ask_returns_answer_payload(self, client, session):
        r = client.post("/api/ask", json={"question": "What?"})
        assert r.status_code == 200
        data = r.json()
        assert data == {
            "answer": "hello",
            "sources": ["Gothic#p.42"],
            "dropped_citations": 0,
        }
        session.oracle.ask.assert_called_once()
        # Oracle.ask is called with top_k=5 (default) and history=None.
        args, kwargs = session.oracle.ask.call_args
        assert args == ("What?",)
        assert kwargs["top_k"] == 5
        assert kwargs["history"] is None

    def test_ask_top_k_and_history_forwarded(self, client, session):
        hist = [{"q": "first", "a": "ans", "sources": []}]
        r = client.post("/api/ask", json={"question": "more?", "top_k": 12, "history": hist})
        assert r.status_code == 200
        kwargs = session.oracle.ask.call_args.kwargs
        assert kwargs["top_k"] == 12
        assert kwargs["history"] == hist

    def test_ask_rejects_empty_question(self, client):
        r = client.post("/api/ask", json={"question": "   "})
        assert r.status_code == 400

    def test_ask_rejects_non_json_body(self, client):
        r = client.post("/api/ask", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400

    def test_ask_stream_uses_sse(self, client, session):
        r = client.post("/api/ask", json={"question": "What?", "stream": True})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # Body is a concatenation of "data: <json>\n\n" chunks.
        events = [
            json.loads(line[len("data: "):])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        assert any(e["type"] == "token" and e["text"] == "Hel" for e in events)
        assert any(e["type"] == "done" for e in events)


# ── search ────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_returns_hybrid_hits(self, client, session):
        session.oracle.connector.find_hybrid.return_value = [
            {"note_id": 3, "text": "x" * 250, "score": 0.91},
            {"note_id": 4, "text": "short", "score": 0.85},
        ]
        r = client.post("/api/search", json={"query": "anything", "top_k": 2})
        assert r.status_code == 200
        hits = r.json()["hits"]
        assert [h["note_id"] for h in hits] == [3, 4]
        assert hits[0]["snippet"].endswith("…")
        assert hits[1]["snippet"] == "short"

    def test_search_falls_back_to_dense_when_fts_off(self, client, session):
        session.db.fts_available = False
        session.oracle.connector.find_similar_notes.return_value = [
            {"note_id": 1, "text": "x", "score": 0.5},
        ]
        client.post("/api/search", json={"query": "x"})
        session.oracle.connector.find_similar_notes.assert_called_once()
        session.oracle.connector.find_hybrid.assert_not_called()

    def test_search_rejects_empty_query(self, client):
        r = client.post("/api/search", json={"query": ""})
        assert r.status_code == 400


# ── notes + categories ────────────────────────────────────────────────


class TestNotes:
    def test_get_note_returns_body(self, tmp_path):
        note_file = tmp_path / "gothic.md"
        note_file.write_text("# Gothic\n\nPointed arches.\n", encoding="utf-8")
        session = _stub_session(note_path=str(note_file))
        client = TestClient(build_app(session))
        r = client.get("/api/notes/7")
        assert r.status_code == 200
        data = r.json()
        assert data["title"] == "Gothic"
        assert "Pointed arches." in data["body"]

    def test_get_missing_note_404(self, session):
        session.db.get_note_location.return_value = None
        client = TestClient(build_app(session))
        r = client.get("/api/notes/9999")
        assert r.status_code == 404

    def test_categories_endpoint(self, client):
        r = client.get("/api/categories")
        assert r.status_code == 200
        assert r.json() == {
            "categories": [
                {"category": "philosophy", "count": 5},
                {"category": "tech/networking", "count": 2},
            ]
        }


# ── auth + CORS ───────────────────────────────────────────────────────


class TestAuth:
    # A non-loopback peer; TestClient lets us spoof the transport address
    # via the `client` kwarg (sets scope["client"]).
    REMOTE = ("192.168.1.50", 51000)
    LOCAL = ("127.0.0.1", 51000)

    def _client(self, session, *, peer, token="secret-token"):
        return TestClient(build_app(session, api_token=token), client=peer)

    # — remote callers must present the token on every /api/* route —

    def test_post_without_token_rejected_when_token_set(self, session):
        client = self._client(session, peer=self.REMOTE)
        r = client.post("/api/ask", json={"question": "q?"})
        assert r.status_code == 401

    def test_post_with_wrong_token_rejected(self, session):
        client = self._client(session, peer=self.REMOTE)
        r = client.post(
            "/api/ask",
            json={"question": "q?"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401

    def test_post_with_correct_token_accepted(self, session):
        client = self._client(session, peer=self.REMOTE)
        r = client.post(
            "/api/ask",
            json={"question": "q?"},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert r.status_code == 200

    def test_remote_get_note_requires_token(self, session):
        # H1 regression: the data-bearing GET must NOT be reachable from a
        # remote host without the bearer. This was the exfiltration path.
        client = self._client(session, peer=self.REMOTE)
        assert client.get("/api/notes/7").status_code == 401
        assert client.get("/api/health").status_code == 401
        assert client.get("/api/categories").status_code == 401

    def test_remote_get_note_with_token_accepted(self, session):
        client = self._client(session, peer=self.REMOTE)
        hdr = {"Authorization": "Bearer secret-token"}
        assert client.get("/api/notes/7", headers=hdr).status_code == 200
        assert client.get("/api/health", headers=hdr).status_code == 200

    # — loopback callers stay open (local browser UI sends no token) —

    def test_loopback_get_open_without_token(self, session):
        client = self._client(session, peer=self.LOCAL)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/categories").status_code == 200
        assert client.get("/api/notes/7").status_code == 200

    def test_loopback_post_open_without_token(self, session):
        # The local browser's ask/search work without threading a token.
        client = self._client(session, peer=self.LOCAL)
        assert client.post("/api/ask", json={"question": "q?"}).status_code == 200

    def test_ui_shell_open_to_remote_without_token(self, session):
        # The static UI shell carries no vault data, so it stays open even
        # to remote callers — only /api/* is gated.
        client = self._client(session, peer=self.REMOTE)
        assert client.get("/").status_code == 200

    def test_no_token_leaves_everything_open(self, session):
        # Default loopback deployment: no token configured → no gate.
        client = TestClient(build_app(session), client=self.REMOTE)
        assert client.get("/api/notes/7").status_code == 200
        assert client.post("/api/ask", json={"question": "q?"}).status_code == 200


class TestCORS:
    def test_cors_off_by_default(self, client):
        # No Access-Control-Allow-Origin should appear when cors_origin
        # wasn't configured.
        r = client.get("/api/health", headers={"Origin": "http://example.com"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_cors_allows_configured_origin_only(self, session):
        client = TestClient(build_app(session, cors_origin="https://app.example"))
        r = client.options(
            "/api/health",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Preflight gets an explicit Access-Control-Allow-Origin echo.
        assert r.headers.get("access-control-allow-origin") == "https://app.example"
