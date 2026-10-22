"""HTTP API + minimal web UI.

Tests drive the API through a direct ASGI transport against a
MagicMock Session — no Ollama, no disk required. Auto-skips when the
``serve`` extra (Starlette + httpx) isn't installed so the suite stays
green on default installs.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

starlette = pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")

from tests.asgi_client import ASGITestClient as TestClient  # noqa: E402

from grimore.api.app import (  # noqa: E402
    _request_has_valid_bearer,
    _TokenAuthMiddleware,
    build_app,
)
from grimore.output.change_journal import ChangeJournal  # noqa: E402
from grimore.ingest.adapters.base import ExtractionIssue  # noqa: E402
from grimore.utils.index_progress import IndexProgressReporter  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────


def _stub_session(*, hits=None, ask_answer="hello",
                  note_title="Gothic", note_path="/vault/gothic.md"):
    session = MagicMock()
    session.config.cognition.hybrid_search = True
    session.config.cognition.allow_remote = False
    session.config.cognition.llm_backend = "ollama"
    session.config.cognition.llm_base_url = ""
    session.config.cognition.model_llm_local = "qwen2.5:3b"
    session.config.cognition.model_embeddings_local = "nomic-embed-text"
    session.config.vault.formats = ["md", "pdf"]
    # Derived from note_path so the note is always inside the vault and
    # get_note's containment check passes for the stub by construction.
    session.vault_root = Path(note_path).parent

    session.embedder.embed.return_value = [1.0, 0.0]
    session.embedder.list_installed_models.return_value = [
        {"name": "nomic-embed-text:latest", "size": 1},
    ]
    session.router.list_installed_models.return_value = [
        {"name": "qwen2.5:3b", "size": 1},
        {"name": "nomic-embed-text:latest", "size": 1},
    ]

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
    session.oracle.rewrite_retrieval_query.side_effect = (
        lambda question, history=None: question
    )

    session.db.fts_available = True
    session.db.vec_available = False
    session.db.get_note_title.return_value = note_title
    session.db.get_note_titles.side_effect = lambda ids: {
        int(note_id): note_title for note_id in ids
    }
    session.db.get_note_location.return_value = (note_path, note_title)
    session.db.get_category_frequency.return_value = [
        ("philosophy", 5), ("tech/networking", 2),
    ]
    session.db.get_tag_frequency.return_value = [
        ("history", 7), ("primary-source", 3),
    ]
    session.db.resolve_note_filter.return_value = None
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
        assert data["cognition"] == {
            "mode": "local",
            "remote_allowed": False,
            "chat": {"backend": "ollama", "mode": "local"},
            "embeddings": {"backend": "ollama", "mode": "local"},
        }

    def test_health_reports_mixed_routes(self, session, monkeypatch):
        session.config.cognition.allow_remote = True
        session.config.cognition.llm_backend = "openai"
        session.config.cognition.llm_base_url = "https://models.example/v1"
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

        data = TestClient(build_app(session)).get("/api/health").json()

        assert data["cognition"]["mode"] == "mixed"
        assert data["cognition"]["chat"]["mode"] == "remote"
        assert data["cognition"]["embeddings"]["mode"] == "local"

    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "Grimore" in r.text
        assert "fonts.googleapis.com" not in r.text
        assert 'id="setup-panel"' in r.text
        assert 'id="setup-toggle"' in r.text
        assert 'id="indexing-panel"' in r.text
        assert 'id="indexing-progress"' in r.text
        assert 'id="review-inbox"' in r.text
        assert 'id="review-filter"' in r.text
        assert 'id="review-operation-status"' in r.text
        assert 'id="review-history-items"' in r.text
        assert 'id="synthesis-form"' in r.text
        assert 'id="synthesis-preview"' in r.text
        assert 'id="synthesis-source-items"' in r.text
        assert 'id="filter-category"' in r.text
        assert 'id="filter-tags"' in r.text
        assert 'id="filter-formats"' in r.text
        assert 'id="conversation-turns"' in r.text
        assert 'id="conversation-reset"' in r.text
        assert 'id="evidence-status"' in r.text
        assert 'id="language-select"' in r.text
        assert 'id="request-error"' in r.text
        assert 'id="operations-token"' in r.text
        assert 'id="setup-apply"' in r.text
        assert 'id="indexing-preview"' in r.text
        assert 'id="indexing-apply"' in r.text
        assert 'id="change-history-items"' in r.text
        assert 'id="change-operation-status"' in r.text
        assert 'data-i18n="invocation"' in r.text

    def test_web_assets_include_keyboard_and_bilingual_support(self, client):
        script = client.get("/static/app.js")
        assert script.status_code == 200
        assert "const I18N" in script.text
        assert "makeKeyboardActivatable" in script.text
        assert "handleDrawerKeydown" in script.text
        assert "showRequestError" in script.text
        assert "ocr_page_limit: 'extractionOcrPageLimit'" in script.text
        assert "OCR page limit reached." in script.text
        assert "Se alcanzó el límite de páginas OCR." in script.text

    def test_bundled_fonts_are_served_locally(self, client):
        css = client.get("/static/grimore.css")
        assert css.status_code == 200
        assert "/static/fonts/NotoSerif-Regular.ttf" in css.text

        font = client.get("/static/fonts/NotoSerif-Regular.ttf")
        assert font.status_code == 200
        assert font.headers["content-type"].startswith("font/ttf")
        assert len(font.content) > 100_000


class TestSetupAssistant:
    def test_reports_current_folder_models_and_scan_effects(self, tmp_path):
        session = _stub_session(note_path=str(tmp_path / "note.md"))
        session.config.vault.path = str(tmp_path)
        client = TestClient(build_app(session))

        response = client.get("/api/setup")

        assert response.status_code == 200
        data = response.json()
        assert data["read_only"] is True
        assert data["ready"] is True
        assert data["vault"] == {
            "path": str(tmp_path),
            "exists": True,
            "is_directory": True,
            "readable": True,
            "valid": True,
            "formats": ["md", "pdf"],
        }
        assert data["models"]["chat"]["selected"] == "qwen2.5:3b"
        assert data["models"]["chat"]["valid"] is True
        assert data["models"]["embeddings"]["valid"] is True
        assert data["scan_effects"]["preview"]
        assert data["scan_effects"]["apply"]
        assert data["operations"] == {
            "write_enabled": False,
            "requires_bearer": True,
            "restart_required": False,
            "config_fields": ["vault_path", "chat_model", "embedding_model"],
            "index_modes": ["preview", "apply"],
            "review_decisions": ["confirm", "dismiss", "postpone"],
            "classification_fields": ["category", "tags"],
            "document_restore": True,
        }
        assert all(check["status"] == "pass" for check in data["checks"][:3])

    def test_reports_invalid_current_folder_without_accepting_candidate_paths(self, tmp_path):
        session = _stub_session(note_path=str(tmp_path / "note.md"))
        session.config.vault.path = str(tmp_path / "missing")
        client = TestClient(build_app(session))

        data = client.get("/api/setup").json()

        assert data["ready"] is False
        assert data["vault"]["valid"] is False
        assert next(c for c in data["checks"] if c["id"] == "vault")["status"] == "error"
        assert client.post("/api/setup", json={"path": "/etc"}).status_code == 405

    def test_openai_chat_lists_embeddings_from_ollama_separately(self, tmp_path):
        session = _stub_session(note_path=str(tmp_path / "note.md"))
        session.config.vault.path = str(tmp_path)
        session.config.cognition.llm_backend = "openai"
        session.router.list_installed_models.return_value = [
            {"name": "remote-chat", "size": 0},
        ]
        session.config.cognition.model_llm_local = "remote-chat"

        data = TestClient(build_app(session)).get("/api/setup").json()

        assert data["models"]["chat"]["available"] == ["remote-chat"]
        assert data["models"]["embeddings"]["available"] == [
            "nomic-embed-text:latest",
        ]
        session.embedder.list_installed_models.assert_called_once_with()


class TestIndexingStatus:
    @staticmethod
    def _redirect_progress_cache(tmp_path, monkeypatch):
        monkeypatch.setattr(
            "grimore.utils.paths.user_cache_dir",
            lambda _name: str(tmp_path / "cache"),
        )

    def test_no_run_returns_safe_recovery_action(self, tmp_path, monkeypatch):
        self._redirect_progress_cache(tmp_path, monkeypatch)
        session = _stub_session(note_path=str(tmp_path / "note.md"))
        session.config.vault.path = str(tmp_path)

        data = TestClient(build_app(session)).get("/api/indexing").json()

        assert data["active"] is False
        assert data["latest"] is None
        assert data["recovery_actions"][0]["command"] == "grimore scan --dry-run"

    def test_reports_live_counts_failures_and_progress(self, tmp_path, monkeypatch):
        self._redirect_progress_cache(tmp_path, monkeypatch)
        session = _stub_session(note_path=str(tmp_path / "note.md"))
        session.config.vault.path = str(tmp_path)
        reporter = IndexProgressReporter(tmp_path, "scan")
        reporter.start(total=4, dry_run=False)
        reporter.record("processed", "a.md", chunks=2)
        reporter.record("failed", "broken.pdf", error="encrypted")

        data = TestClient(build_app(session)).get("/api/indexing").json()

        assert data["active"] is True
        assert data["latest"]["progress"] == 0.5
        assert data["latest"]["counts"]["processed"] == 1
        assert data["latest"]["counts"]["failed"] == 1
        assert data["latest"]["items"][-1]["path"] == "broken.pdf"
        assert data["recovery_actions"][0]["id"] == "retry-failed"


class TestReviewInbox:
    def test_returns_read_only_candidate_payload(self, client, monkeypatch):
        payload = {
            "generated_at": "2026-01-01T00:00:00Z",
            "status": "candidates_only",
            "counts": {
                "all": 1, "contradiction": 1, "duplicate_exact": 0,
                "duplicate_near": 0, "review_due": 0,
            },
            "items": [{
                "id": "contradiction:7", "kind": "contradiction",
                "status": "candidate", "documents": [],
            }],
        }
        monkeypatch.setattr("grimore.api.app.build_review_inbox", lambda _session: payload)

        response = client.get("/api/review-inbox")

        assert response.status_code == 200
        assert response.json() == payload


class TestSynthesisPreview:
    def test_returns_reviewable_dry_run_without_write_token(
        self, session, monkeypatch,
    ):
        calls = []
        payload = {
            "selector": "tag:history",
            "output_path": "/vault/_synthesis/tag-history_2026-09-29.md",
            "sources": ["Chronicle"],
            "notes_considered": 1,
            "notes_excluded_generated": 0,
            "notes_used": 1,
            "title": "History",
            "skipped_reason": None,
            "body": "## Summary\n[[Chronicle]] records the event.",
            "source_details": [{
                "note_id": 7,
                "path": "/vault/chronicle.md",
                "title": "Chronicle",
                "passages": ["The event happened."],
                "passages_selected": 1,
                "passages_available": 4,
                "included_in_prompt": True,
                "omission_reason": None,
            }],
            "coverage": {
                "status": "partial",
                "limitations": ["representative_passages_only"],
            },
            "review_status": "draft",
        }

        class FakeSynthesizer:
            def __init__(self, captured_session):
                assert captured_session is session

            def distill(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(as_dict=lambda: payload)

        monkeypatch.setattr("grimore.api.app.Synthesizer", FakeSynthesizer)
        local = TestClient(
            build_app(session, api_token="configured"),
            client=("127.0.0.1", 51000),
        )

        response = local.post("/api/syntheses/preview", json={
            "kind": "tag", "value": " history ", "passages_per_note": 1,
        })

        assert response.status_code == 200
        assert response.json() == payload
        assert calls == [{
            "tag": "history", "passages_per_note": 1, "dry_run": True,
        }]

    @pytest.mark.parametrize("body,error", [
        ({}, "kind must be"),
        ({"kind": "folder", "value": "x"}, "kind must be"),
        ({"kind": "tag", "value": "   "}, "value is invalid"),
        (
            {"kind": "tag", "value": "x", "passages_per_note": True},
            "passages_per_note",
        ),
        (
            {"kind": "tag", "value": "x", "passages_per_note": 11},
            "passages_per_note",
        ),
        ({"kind": "tag", "value": "x", "extra": 1}, "unsupported"),
    ])
    def test_validates_preview_scope(self, client, body, error):
        response = client.post("/api/syntheses/preview", json=body)

        assert response.status_code == 400
        assert error in response.json()["error"]


class TestChangeHistory:
    @staticmethod
    def _redirect_cache(tmp_path, monkeypatch):
        monkeypatch.setattr(
            "grimore.utils.paths.user_cache_dir",
            lambda _name: str(tmp_path / "cache"),
        )

    def test_lists_and_restores_real_prechange_copy(self, tmp_path, monkeypatch):
        self._redirect_cache(tmp_path, monkeypatch)
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "note.md"
        note.write_text("before\n", encoding="utf-8")
        journal = ChangeJournal(vault)
        change_id = journal.start("scan", details={"source": "test"})
        journal.capture_before(note)
        note.write_text("after\n", encoding="utf-8")
        journal.capture_after(note)
        journal.finish()
        session = _stub_session(note_path=str(note))
        session.config.vault.path = str(vault)
        client = TestClient(
            build_app(session, api_token="secret-token"),
            client=("127.0.0.1", 51000),
        )

        listed = client.get("/api/changes")
        unauthenticated = client.post(
            f"/api/changes/{change_id}/restore",
            json={"confirm": "restore-change-set"},
        )
        restored = client.post(
            f"/api/changes/{change_id}/restore",
            headers={"Authorization": "Bearer secret-token"},
            json={"confirm": "restore-change-set"},
        )

        assert listed.status_code == 200
        assert listed.json()["changes"][0]["files"] == [
            {"path": "note.md", "action": "updated"},
        ]
        assert unauthenticated.status_code == 401
        assert restored.status_code == 200
        assert restored.json()["requires_reindex"] is True
        assert note.read_text(encoding="utf-8") == "before\n"

    def test_restore_requires_exact_confirmation(self, tmp_path, monkeypatch):
        self._redirect_cache(tmp_path, monkeypatch)
        vault = tmp_path / "vault"
        vault.mkdir()
        session = _stub_session(note_path=str(vault / "note.md"))
        session.config.vault.path = str(vault)
        client = TestClient(
            build_app(session, api_token="secret-token"),
            client=("127.0.0.1", 51000),
        )

        response = client.post(
            f"/api/changes/{'a' * 32}/restore",
            headers={"Authorization": "Bearer secret-token"},
            json={"confirm": "yes"},
        )

        assert response.status_code == 400
        assert "restore-change-set" in response.json()["error"]


class TestScopedWriteOperations:
    LOCAL = ("127.0.0.1", 51000)
    AUTH = {"Authorization": "Bearer secret-token"}

    @staticmethod
    def _session_for(tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir(parents=True)
        session = _stub_session(note_path=str(vault / "note.md"))
        session.config.vault.path = str(vault)
        return session, vault

    def test_writes_are_disabled_without_server_token(self, tmp_path):
        session, _ = self._session_for(tmp_path)
        client = TestClient(build_app(session), client=self.LOCAL)

        config = client.post("/api/setup/apply", json={})
        indexing = client.post("/api/indexing/start", json={"mode": "preview"})
        review = client.post(
            "/api/review-inbox/duplicate-near:1:2/decision",
            json={"decision": "confirm"},
        )
        classification = client.post(
            "/api/notes/1/classification", json={"category": None, "tags": []},
        )
        restore = client.post(
            f"/api/changes/{'a' * 32}/restore",
            json={"confirm": "restore-change-set"},
        )

        assert config.status_code == 503
        assert indexing.status_code == 503
        assert review.status_code == 503
        assert classification.status_code == 503
        assert restore.status_code == 503

    def test_loopback_write_requires_correct_bearer(self, tmp_path):
        session, _ = self._session_for(tmp_path)
        client = TestClient(
            build_app(session, api_token="secret-token"), client=self.LOCAL,
        )

        assert client.post("/api/indexing/start", json={"mode": "preview"}).status_code == 401
        wrong = {"Authorization": "Bearer wrong"}
        assert client.post(
            "/api/indexing/start", json={"mode": "preview"}, headers=wrong,
        ).status_code == 401
        assert client.post(
            "/api/review-inbox/duplicate-near:1:2/decision",
            json={"decision": "dismiss"},
        ).status_code == 401
        assert client.post(
            "/api/notes/1/classification",
            json={"category": "manual", "tags": ["curated"]},
        ).status_code == 401
        assert client.post(
            f"/api/changes/{'a' * 32}/restore",
            json={"confirm": "restore-change-set"},
        ).status_code == 401

    def test_config_apply_is_scoped_validated_and_atomic(self, tmp_path):
        session, vault = self._session_for(tmp_path)
        config_path = tmp_path / "grimore.toml"
        config_path.write_text(
            '# keep me\n[memory]\ndb_path = "custom.db"\n', encoding="utf-8",
        )
        client = TestClient(build_app(
            session,
            api_token="secret-token",
            config_path=config_path,
        ), client=self.LOCAL)

        response = client.post("/api/setup/apply", headers=self.AUTH, json={
            "vault_path": str(vault),
            "chat_model": "qwen2.5:3b",
            "embedding_model": "nomic-embed-text:latest",
            "confirm": "apply-config",
        })

        assert response.status_code == 200
        assert response.json()["restart_required"] is True
        text = config_path.read_text(encoding="utf-8")
        assert "# keep me" in text
        assert 'db_path = "custom.db"' in text
        assert f'path = "{vault}"' in text
        blocked_scan = client.post(
            "/api/indexing/start", headers=self.AUTH, json={"mode": "preview"},
        )
        assert blocked_scan.status_code == 409
        assert "restart" in blocked_scan.json()["error"]

    def test_config_apply_rejects_unknown_fields_and_unreported_models(self, tmp_path):
        session, vault = self._session_for(tmp_path)
        client = TestClient(build_app(
            session, api_token="secret-token", config_path=tmp_path / "grimore.toml",
        ), client=self.LOCAL)
        base = {
            "vault_path": str(vault),
            "chat_model": "qwen2.5:3b",
            "embedding_model": "nomic-embed-text:latest",
            "confirm": "apply-config",
        }

        assert client.post(
            "/api/setup/apply", headers=self.AUTH, json={**base, "output": {}},
        ).status_code == 400
        assert client.post(
            "/api/setup/apply", headers=self.AUTH,
            json={**base, "chat_model": "not-reported"},
        ).status_code == 400
        oversized = client.post(
            "/api/setup/apply",
            headers={**self.AUTH, "Content-Type": "application/json"},
            content=b'{' + (b'x' * 17_000),
        )
        assert oversized.status_code == 413

    def test_indexing_modes_are_fixed_and_single_flight(self, tmp_path, monkeypatch):
        TestIndexingStatus._redirect_progress_cache(tmp_path, monkeypatch)
        session, _ = self._session_for(tmp_path)
        calls = []

        class Process:
            returncode = None

            async def wait(self):
                import asyncio
                await asyncio.sleep(60)

        async def launcher(*, dry_run):
            calls.append(dry_run)
            return Process()

        app = build_app(
            session, api_token="secret-token", scan_launcher=launcher,
        )
        client = TestClient(app, client=self.LOCAL)
        preview = client.post(
            "/api/indexing/start", headers=self.AUTH, json={"mode": "preview"},
        )
        duplicate = client.post(
            "/api/indexing/start", headers=self.AUTH,
            json={"mode": "apply", "confirm": "apply-indexing"},
        )

        assert preview.status_code == 202
        assert preview.json()["operation"]["mode"] == "preview"
        assert duplicate.status_code == 409
        assert calls == [True]

    def test_apply_indexing_needs_explicit_confirmation(self, tmp_path, monkeypatch):
        session, _ = self._session_for(tmp_path)
        TestIndexingStatus._redirect_progress_cache(tmp_path, monkeypatch)
        called = False

        async def launcher(*, dry_run):
            nonlocal called
            called = True

        client = TestClient(build_app(
            session, api_token="secret-token", scan_launcher=launcher,
        ), client=self.LOCAL)

        response = client.post(
            "/api/indexing/start", headers=self.AUTH, json={"mode": "apply"},
        )

        assert response.status_code == 400
        assert called is False

    def test_review_decision_is_validated_persisted_and_audited(self, tmp_path, monkeypatch):
        session, _ = self._session_for(tmp_path)
        candidate = {
            "id": "duplicate-near:1:2",
            "kind": "duplicate_near",
            "priority": "medium",
            "documents": [{"note_id": 1, "path": "/v/a.md", "title": "A"}],
            "reason": "similar",
            "coverage": {"status": "complete"},
            "metadata": {"score": 0.95},
        }
        monkeypatch.setattr(
            "grimore.api.app.build_review_inbox",
            lambda _session, **_kwargs: {"items": [candidate]},
        )
        session.db.record_review_decision.return_value = {
            "event_id": 1,
            "candidate_id": candidate["id"],
            "decision": "postpone",
        }
        client = TestClient(
            build_app(session, api_token="secret-token"), client=self.LOCAL,
        )

        response = client.post(
            "/api/review-inbox/duplicate-near%3A1%3A2/decision",
            headers=self.AUTH,
            json={"decision": "postpone", "postpone_days": 30},
        )

        assert response.status_code == 200
        assert response.json()["event"]["event_id"] == 1
        call = session.db.record_review_decision.call_args.kwargs
        assert call["candidate_id"] == candidate["id"]
        assert call["kind"] == "duplicate_near"
        assert call["decision"] == "postpone"
        assert call["postpone_until"] > call["decided_at"]
        assert call["snapshot"]["documents"][0]["title"] == "A"

    def test_review_decision_rejects_bad_shape_and_unknown_candidate(self, tmp_path, monkeypatch):
        session, _ = self._session_for(tmp_path)
        monkeypatch.setattr(
            "grimore.api.app.build_review_inbox",
            lambda _session, **_kwargs: {"items": []},
        )
        client = TestClient(
            build_app(session, api_token="secret-token"), client=self.LOCAL,
        )

        bad_days = client.post(
            "/api/review-inbox/review-due:x/decision",
            headers=self.AUTH,
            json={"decision": "postpone", "postpone_days": 0},
        )
        unknown = client.post(
            "/api/review-inbox/review-due:x/decision",
            headers=self.AUTH,
            json={"decision": "confirm"},
        )

        assert bad_days.status_code == 400
        assert unknown.status_code == 404
        session.db.record_review_decision.assert_not_called()

    def test_classification_override_is_normalized_and_persisted(self, tmp_path):
        session, _ = self._session_for(tmp_path)
        session.db.set_classification_override.return_value = {
            "event_id": 4,
            "note_id": 7,
            "category": "human/reference",
            "tags": ["Curated", "Trusted"],
        }
        client = TestClient(
            build_app(session, api_token="secret-token"), client=self.LOCAL,
        )

        response = client.post(
            "/api/notes/7/classification",
            headers=self.AUTH,
            json={
                "category": "/human/reference/",
                "tags": [" Curated ", "curated", "Trusted", ""],
            },
        )

        assert response.status_code == 200
        call = session.db.set_classification_override.call_args.kwargs
        assert call["note_id"] == 7
        assert call["category"] == "human/reference"
        assert call["tags"] == ["Curated", "Trusted"]
        assert call["source"] == "web"

    def test_classification_override_rejects_partial_or_oversized_values(self, tmp_path):
        session, _ = self._session_for(tmp_path)
        client = TestClient(
            build_app(session, api_token="secret-token"), client=self.LOCAL,
        )

        partial = client.post(
            "/api/notes/7/classification",
            headers=self.AUTH,
            json={"tags": ["only"]},
        )
        oversized = client.post(
            "/api/notes/7/classification",
            headers=self.AUTH,
            json={"category": "ok", "tags": ["x" * 101]},
        )

        assert partial.status_code == 400
        assert oversized.status_code == 400
        session.db.set_classification_override.assert_not_called()


# ── ask ───────────────────────────────────────────────────────────────


class TestAsk:
    @pytest.mark.parametrize("stream", [False, True])
    def test_forwards_optional_sufficiency_assessment(self, client, session, stream):
        assessment = {"decision": "insufficient", "semantic_support": "unverified"}
        payload = {"answer": "Evidencia insuficiente.", "sources": [],
                   "sufficiency": assessment, "answer_status": "insufficient_evidence"}
        session.oracle.ask.return_value = payload
        session.oracle.ask_stream.return_value = iter([{"type": "done", **payload}])
        response = client.post("/api/ask", json={"question": "What?", "stream": stream})
        if stream:
            events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
                      if line.startswith("data: ")]
            assert events[-1]["sufficiency"] == assessment
        else:
            assert response.json()["sufficiency"] == assessment

    def test_ask_returns_answer_payload(self, client, session):
        r = client.post("/api/ask", json={"question": "What?"})
        assert r.status_code == 200
        data = r.json()
        assert data == {
            "answer": "hello",
            "sources": ["Gothic#p.42"],
            "source_details": [],
            "citation_selections": None,
            "answer_status": "uncertain",
            "model_execution": {
                "mode": "local",
                "remote_allowed": False,
                "chat": {"backend": "ollama", "mode": "local"},
                "embeddings": {"backend": "ollama", "mode": "local"},
            },
            "retrieval_scope": {"status": "all", "matched_notes": None},
            "dropped_citations": 0,
        }
        session.oracle.ask.assert_called_once()
        # Oracle.ask is called with top_k=5 (default) and history=None.
        args, kwargs = session.oracle.ask.call_args
        assert args == ("What?",)
        assert kwargs["top_k"] == 5
        assert kwargs["history"] is None
        assert kwargs["filter_note_ids"] is None

    def test_ask_reports_a_model_failure_instead_of_missing_evidence(self, client, session):
        session.oracle.ask.return_value = {
            "answer": "The Oracle is silent.", "sources": [],
            "answer_status": "insufficient_evidence", "generation_error": "no_model_response"}
        data = client.post("/api/ask", json={"question": "What?"}).json()
        assert data["generation_error"] == "no_model_response"

    def test_ask_top_k_and_history_forwarded(self, client, session):
        hist = [{"q": "first", "a": "ans", "sources": []}]
        r = client.post("/api/ask", json={"question": "more?", "top_k": 12, "history": hist})
        assert r.status_code == 200
        kwargs = session.oracle.ask.call_args.kwargs
        assert kwargs["top_k"] == 12
        assert kwargs["history"] == hist

    @pytest.mark.parametrize("stream", [False, True])
    def test_ask_forwards_exact_selection_trace_in_final_payload(self, client, session, stream):
        trace = [{"citation_index": 0, "label": "Gothic#p.4", "form": "short",
                  "citation_ref": "cite:2", "source_detail_indices": [1], "selection": "exact"}]
        payload = {"answer": "Evidence [[Gothic#p.4]]", "sources": ["Gothic#p.4"],
                   "citation_selections": trace}
        session.oracle.ask.return_value = payload
        session.oracle.ask_stream.return_value = iter([{"type": "done", **payload}])
        response = client.post("/api/ask", json={"question": "What?", "stream": stream})
        body = (json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith("data: ")))
                if stream else response.json())
        assert body["citation_selections"] == trace

    def test_ask_clamps_oversized_top_k(self, client, session):
        # Clamp before retrieval so callers cannot amplify backend work.
        r = client.post("/api/ask", json={"question": "q?", "top_k": 9999})
        assert r.status_code == 200
        assert session.oracle.ask.call_args.kwargs["top_k"] == 50

    def test_ask_rejects_non_numeric_top_k(self, client):
        # Non-numeric top_k is a clean 400, not an uncaught ValueError → 500.
        r = client.post("/api/ask", json={"question": "q?", "top_k": "lots"})
        assert r.status_code == 400

    def test_ask_rejects_empty_question(self, client):
        r = client.post("/api/ask", json={"question": "   "})
        assert r.status_code == 400

    def test_ask_rejects_non_json_body(self, client):
        r = client.post("/api/ask", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400

    @pytest.mark.parametrize("body", [["question"], "question", 7, True])
    def test_ask_rejects_non_object_json(self, client, body):
        r = client.post("/api/ask", json=body)
        assert r.status_code == 400
        assert r.json()["error"] == "body must be a JSON object"

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
        done = next(e for e in events if e["type"] == "done")
        assert done["model_execution"]["mode"] == "local"
        assert done["retrieval_scope"] == {"status": "all", "matched_notes": None}

    def test_stream_forwards_inspectable_source_details(self, client, session):
        detail = {
            "id": "note:7:chunk:2:embedding:9",
            "label": "Gothic#p.4",
            "document": {
                "note_id": 7, "title": "Gothic", "path": "gothic.pdf",
                "format": "pdf",
            },
            "locator": {"chunk_index": 2, "page": 4, "heading": "Vaults"},
            "passage": {"text": "Rib vault evidence.", "completeness": "full"},
            "source_state": "current",
            "indexed_extraction": {
                "status": "partial", "missing_pages": [2],
                "missing_page_count": 1,
            },
        }
        session.oracle.ask_stream.return_value = iter([{
            "type": "done",
            "answer": "Grounded [[Gothic#p.4]].",
            "sources": ["Gothic#p.4"],
            "source_details": [detail],
            "answer_status": "supported",
            "dropped_citations": 0,
        }])

        response = client.post("/api/ask", json={"question": "What?", "stream": True})
        done = next(
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        )

        assert done["source_details"] == [detail]
        assert done["answer_status"] == "supported"

    def test_ask_filters_reach_normal_retrieval(self, client, session):
        session.db.resolve_note_filter.return_value = {3, 8}

        response = client.post("/api/ask", json={
            "question": "What?",
            "category": "history",
            "tags": ["primary-source"],
            "formats": ["pdf"],
        })

        assert response.status_code == 200
        session.db.resolve_note_filter.assert_called_once_with(
            category="history", tags=["primary-source"], formats=["pdf"],
        )
        assert session.oracle.ask.call_args.kwargs["filter_note_ids"] == {3, 8}
        assert response.json()["retrieval_scope"] == {
            "status": "filtered", "matched_notes": 2,
        }

    def test_ask_filters_reach_streaming_retrieval(self, client, session):
        session.db.resolve_note_filter.return_value = {9}

        response = client.post("/api/ask", json={
            "question": "What?", "stream": True, "formats": ["md"],
        })

        assert response.status_code == 200
        assert session.oracle.ask_stream.call_args.kwargs["filter_note_ids"] == {9}

    def test_ask_keeps_an_empty_filter_instead_of_widening(self, client, session):
        session.db.resolve_note_filter.return_value = set()

        response = client.post(
            "/api/ask", json={"question": "What?", "tags": ["missing"]},
        )

        assert response.status_code == 200
        assert response.json()["answer_status"] == "insufficient_evidence"
        assert response.json()["retrieval_scope"] == {
            "status": "no_matches", "matched_notes": 0,
        }
        session.oracle.ask.assert_not_called()
        session.embedder.embed.assert_not_called()

    def test_streaming_empty_filter_finishes_without_calling_oracle(self, client, session):
        session.db.resolve_note_filter.return_value = set()

        response = client.post("/api/ask", json={
            "question": "What?", "stream": True, "formats": ["epub"],
        })
        events = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

        assert response.status_code == 200
        assert events == [{
            "type": "done",
            "answer": "No indexed documents match the selected search scope.",
            "sources": [],
            "source_details": [],
            "citation_selections": [],
            "answer_status": "insufficient_evidence",
            "retrieval_scope": {"status": "no_matches", "matched_notes": 0},
            "dropped_citations": 0,
            "model_execution": {
                "mode": "local",
                "remote_allowed": False,
                "chat": {"backend": "ollama", "mode": "local"},
                "embeddings": {"backend": "ollama", "mode": "local"},
            },
        }]
        session.oracle.ask_stream.assert_not_called()
        session.embedder.embed.assert_not_called()

    @pytest.mark.parametrize("bad", [
        {"tags": 5}, {"category": ["history"]}, {"formats": [1]},
    ])
    def test_ask_rejects_malformed_filters(self, client, session, bad):
        response = client.post("/api/ask", json={"question": "What?", **bad})

        assert response.status_code == 400
        session.oracle.ask.assert_not_called()
        session.oracle.ask_stream.assert_not_called()


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

    def test_search_uses_conversation_to_rewrite_follow_up(self, client, session):
        history = [{"q": "Tell me about Gothic vaults", "a": "They use ribs."}]
        session.oracle.rewrite_retrieval_query.side_effect = None
        session.oracle.rewrite_retrieval_query.return_value = "Gothic rib vault materials"

        response = client.post("/api/search", json={
            "query": "What about their materials?",
            "history": history,
        })

        assert response.status_code == 200
        session.oracle.rewrite_retrieval_query.assert_called_once_with(
            "What about their materials?", history=history,
        )
        session.embedder.embed.assert_called_once_with("Gothic rib vault materials")
        assert session.oracle.connector.find_hybrid.call_args.kwargs["query_text"] == \
            "Gothic rib vault materials"

    def test_search_clamps_oversized_top_k(self, client, session):
        session.oracle.connector.find_hybrid.return_value = []
        r = client.post("/api/search", json={"query": "x", "top_k": 9999})
        assert r.status_code == 200
        assert session.oracle.connector.find_hybrid.call_args.kwargs["top_k"] == 50

    def test_search_rejects_non_numeric_top_k(self, client):
        r = client.post("/api/search", json={"query": "x", "top_k": "lots"})
        assert r.status_code == 400

    def test_search_empty_filter_short_circuits_before_embedding(self, client, session):
        session.db.resolve_note_filter.return_value = set()

        response = client.post(
            "/api/search", json={"query": "x", "tags": ["missing"]},
        )

        assert response.json() == {
            "hits": [],
            "retrieval_scope": {"status": "no_matches", "matched_notes": 0},
        }
        session.embedder.embed.assert_not_called()
        session.oracle.connector.find_hybrid.assert_not_called()
        session.oracle.connector.find_similar_notes.assert_not_called()

    def test_search_rejects_empty_query(self, client):
        r = client.post("/api/search", json={"query": ""})
        assert r.status_code == 400

    @pytest.mark.parametrize("body", [["query"], "query", 7, True])
    def test_search_rejects_non_object_json(self, client, body):
        r = client.post("/api/search", json=body)
        assert r.status_code == 400
        assert r.json()["error"] == "body must be a JSON object"


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
        assert data["timeline"] == {
            "published": {"value": None, "source": None},
            "modified": {"value": None, "source": None},
            "indexed": {"value": None, "source": None},
            "verified": {"value": None, "source": None},
        }

    def test_get_note_returns_page_and_section_navigation(self, tmp_path, monkeypatch):
        note_file = tmp_path / "gothic.pdf"
        note_file.write_bytes(b"placeholder")
        session = _stub_session(note_path=str(note_file))
        content = "Page one text.\n\nRib vault evidence."
        parsed = SimpleNamespace(
            content=content,
            sections=[
                SimpleNamespace(
                    text="Page one text.", page=1, heading="Introduction", order=0,
                ),
                SimpleNamespace(
                    text="Rib vault evidence.", page=2, heading="Vaults", order=1,
                ),
            ],
        )
        parser = MagicMock()
        parser.parse_file.return_value = parsed
        monkeypatch.setattr("grimore.api.app.MarkdownParser", lambda: parser)

        data = TestClient(build_app(session)).get("/api/notes/7").json()

        assert data["body"] == content
        assert data["sections"] == [
            {"order": 0, "page": 1, "heading": "Introduction", "start": 0, "end": 14},
            {"order": 1, "page": 2, "heading": "Vaults", "start": 16, "end": 35},
        ]

    @pytest.mark.parametrize("reason", ["ocr_unavailable", "ocr_page_limit"])
    def test_get_note_exposes_incomplete_pdf_extraction(self, tmp_path, monkeypatch, reason):
        note_file = tmp_path / "incomplete.pdf"
        note_file.write_bytes(b"placeholder")
        session = _stub_session(note_path=str(note_file))
        parsed = SimpleNamespace(
            content="Only page two was read.", sections=[], format="pdf",
            issues=[ExtractionIssue(
                "page_text_unavailable", page=1, engine="pypdf",
                ocr_requested=True, reason=reason,
            )],
        )
        parser = MagicMock()
        parser.parse_file.return_value = parsed
        monkeypatch.setattr("grimore.api.app.MarkdownParser", lambda: parser)

        data = TestClient(build_app(session)).get("/api/notes/7").json()

        assert data["extraction"]["status"] == "partial"
        assert data["extraction"]["issues"][0]["page"] == 1
        assert data["extraction"]["issues"][0]["reason"] == reason

    def test_get_note_reports_current_file_read_failure(self, tmp_path, monkeypatch):
        note_file = tmp_path / "unreadable.pdf"
        note_file.write_bytes(b"placeholder")
        session = _stub_session(note_path=str(note_file))
        parser = MagicMock()
        parser.parse_file.side_effect = ValueError("invalid PDF")
        monkeypatch.setattr("grimore.api.app.MarkdownParser", lambda: parser)

        data = TestClient(build_app(session)).get("/api/notes/7").json()

        assert data["body"] == ""
        assert data["extraction"] == {
            "status": "failed",
            "issues": [{"code": "document_read_failed", "page": None}],
        }

    def test_get_note_path_outside_vault_404(self, tmp_path):
        # A DB row whose path escapes the vault (tampered index, or a
        # symlink swapped after scan) must not become an arbitrary-file
        # read for an API caller — it 404s like a missing note.
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        session = _stub_session(note_path=str(secret))
        session.vault_root = tmp_path / "vault"
        client = TestClient(build_app(session))
        r = client.get("/api/notes/7")
        assert r.status_code == 404
        assert "top secret" not in r.text

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

    def test_filter_options_endpoint(self, client, session):
        response = client.get("/api/filter-options")

        assert response.status_code == 200
        assert response.json() == {
            "categories": [
                {"category": "philosophy", "count": 5},
                {"category": "tech/networking", "count": 2},
            ],
            "tags": [
                {"tag": "history", "count": 7},
                {"tag": "primary-source", "count": 3},
            ],
            "formats": ["md", "pdf"],
        }
        session.db.get_tag_frequency.assert_called_once_with(limit=100)


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
        # Data-bearing routes require the bearer when accessed remotely.
        client = self._client(session, peer=self.REMOTE)
        assert client.get("/api/notes/7").status_code == 401
        assert client.get("/api/health").status_code == 401
        assert client.get("/api/categories").status_code == 401
        assert client.get("/api/review-inbox").status_code == 401

    def test_non_ascii_bearer_does_not_crash(self):
        # A bearer carrying high bytes (>= 0x80) arrives raw in the ASGI
        # scope — an httpx client can't even send it, so we exercise the
        # check directly. secrets.compare_digest raises TypeError on
        # non-ASCII *str*; comparing raw *bytes* makes a malformed token
        # return False (→ 401) instead of raising (→ 500).
        scope = {"headers": [(b"authorization", b"Bearer \xff\xfe-nope")]}
        assert _request_has_valid_bearer(scope, "secret-token") is False

    def test_bytes_bearer_matches_token(self):
        # Positive control: the raw-bytes path still accepts a good token.
        scope = {"headers": [(b"authorization", b"Bearer secret-token")]}
        assert _request_has_valid_bearer(scope, "secret-token") is True

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


class TestStrictToken:
    # On Android/Termux any app on the device can reach localhost ports,
    # so --strict-token drops the loopback exemption.
    LOCAL = ("127.0.0.1", 51000)

    def _client(self, session):
        return TestClient(
            build_app(session, api_token="secret-token", strict_token=True),
            client=self.LOCAL,
        )

    def test_loopback_requires_token(self, session):
        client = self._client(session)
        assert client.get("/api/health").status_code == 401
        assert client.get("/api/notes/7").status_code == 401
        hdr = {"Authorization": "Bearer secret-token"}
        assert client.get("/api/health", headers=hdr).status_code == 200

    def test_ui_shell_stays_open(self, session):
        # Only /api/* is gated; the static shell carries no vault data.
        assert self._client(session).get("/").status_code == 200

    def test_strict_without_token_raises(self, session):
        # Nothing to enforce — refuse to build rather than serve open.
        with pytest.raises(ValueError):
            build_app(session, strict_token=True)


class TestAuthThrottle:
    REMOTE = ("192.168.1.50", 51000)

    def _client(self, session):
        return TestClient(
            build_app(session, api_token="secret-token"), client=self.REMOTE,
        )

    def test_throttles_after_repeated_failures(self, session):
        client = self._client(session)
        bad = {"Authorization": "Bearer wrong"}
        for _ in range(_TokenAuthMiddleware.MAX_FAILURES):
            assert client.get("/api/health", headers=bad).status_code == 401
        r = client.get("/api/health", headers=bad)
        assert r.status_code == 429
        assert r.headers.get("retry-after") == str(
            int(_TokenAuthMiddleware.WINDOW_SECONDS)
        )
        # While throttled, even the right token is refused — the lockout
        # is what bounds an online brute force.
        good = {"Authorization": "Bearer secret-token"}
        assert client.get("/api/health", headers=good).status_code == 429

    def test_success_resets_failure_count(self, session):
        client = self._client(session)
        bad = {"Authorization": "Bearer wrong"}
        good = {"Authorization": "Bearer secret-token"}
        for _ in range(_TokenAuthMiddleware.MAX_FAILURES - 1):
            client.get("/api/health", headers=bad)
        assert client.get("/api/health", headers=good).status_code == 200
        # Counter cleared: a full window of failures is available again.
        for _ in range(_TokenAuthMiddleware.MAX_FAILURES - 1):
            assert client.get("/api/health", headers=bad).status_code == 401

    def test_window_expiry_unthrottles(self):
        # Unit-level: drive the clock by hand instead of sleeping.
        mw = _TokenAuthMiddleware(None, api_token="t")
        for _ in range(mw.MAX_FAILURES):
            mw._record_failure("1.2.3.4", 100.0)
        assert mw._is_throttled("1.2.3.4", 100.0) is True
        assert mw._is_throttled("1.2.3.4", 100.0 + mw.WINDOW_SECONDS) is False

    def test_peer_map_stays_bounded(self):
        mw = _TokenAuthMiddleware(None, api_token="t")
        for i in range(mw._MAX_TRACKED_PEERS + 200):
            mw._record_failure(f"10.0.{i // 256}.{i % 256}", 100.0 + i * 0.001)
        assert len(mw._failures) <= mw._MAX_TRACKED_PEERS


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
