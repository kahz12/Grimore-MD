"""Pluggable LLM backends.

Covers the backend factory, both shipped backends (Ollama + OpenAI-
compatible), and the router-level circuit-breaker / json-extraction
wrap. The HTTP layer is fully mocked — every test runs against a
``MagicMock`` ``requests.Session`` so the suite stays offline.

Tests don't go through the router for low-level shape checks; they hit
the backend directly. Then the router-level tests confirm that:

* The dispatcher picks the right backend from ``config.llm_backend``.
* JSON extraction runs *after* the backend returns raw text (so
  servers that don't honour ``response_format`` still parse).
* The circuit breaker advances on backend failures.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from grimore.cognition.llm_backends import build_backend
from grimore.cognition.llm_backends.ollama import OllamaBackend
from grimore.cognition.llm_backends.openai import OpenAICompatibleBackend
from grimore.cognition.llm_router import LLMRouter
from grimore.utils.config import CognitionConfig, Config


# ── helpers ─────────────────────────────────────────────────────────────


def _cfg(**cognition) -> Config:
    """Build a Config with cognition overrides."""
    cfg = Config()
    cfg.cognition = CognitionConfig(**cognition)
    return cfg


def _ollama(**cognition) -> OllamaBackend:
    b = OllamaBackend(_cfg(**cognition))
    b.session = MagicMock()
    return b


def _openai(**cognition) -> OpenAICompatibleBackend:
    b = OpenAICompatibleBackend(_cfg(**cognition))
    b.session = MagicMock()
    return b


# ── OllamaBackend ───────────────────────────────────────────────────────


class TestOllamaBackend:
    """The v2.x default, lifted verbatim. Confirm wire shape didn't drift."""

    def test_complete_posts_generate_with_json_format(self):
        b = _ollama()
        resp = MagicMock()
        resp.json.return_value = {"response": '{"k": "v"}'}
        b.session.post.return_value = resp

        out = b.complete("prompt", system_prompt="sys", json_format=True)
        assert out == '{"k": "v"}'

        call = b.session.post.call_args
        url = call.args[0] if call.args else call.kwargs["url"]
        assert url.endswith("/api/generate")
        payload = call.kwargs["json"]
        assert payload["model"] == CognitionConfig().model_llm_local
        assert payload["prompt"] == "prompt"
        assert payload["system"] == "sys"
        assert payload["stream"] is False
        assert payload["format"] == "json"

    def test_complete_omits_format_when_json_false(self):
        b = _ollama()
        resp = MagicMock()
        resp.json.return_value = {"response": "plain text answer"}
        b.session.post.return_value = resp

        b.complete("prompt", json_format=False)
        payload = b.session.post.call_args.kwargs["json"]
        assert "format" not in payload

    def test_complete_returns_none_on_error(self):
        b = _ollama()
        b.session.post.side_effect = RuntimeError("network down")
        assert b.complete("prompt") is None

    def test_streaming_yields_response_pieces(self):
        b = _ollama()
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.iter_lines.return_value = iter([
            json.dumps({"response": "Hel"}),
            json.dumps({"response": "lo"}),
            json.dumps({"response": "", "done": True}),
        ])
        b.session.post.return_value = resp

        pieces = list(b.complete_streaming("q"))
        assert "".join(pieces) == "Hello"

    def test_streaming_skips_malformed_lines(self):
        b = _ollama()
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.iter_lines.return_value = iter([
            "not json",
            "",
            json.dumps({"response": "ok"}),
            json.dumps({"done": True}),
        ])
        b.session.post.return_value = resp
        assert list(b.complete_streaming("q")) == ["ok"]

    def test_streaming_returns_empty_on_error(self):
        b = _ollama()
        b.session.post.side_effect = RuntimeError("boom")
        assert list(b.complete_streaming("q")) == []

    def test_list_models_normalises_shape(self):
        b = _ollama()
        resp = MagicMock()
        resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:3b", "size": 2_000_000_000},
                {"size": 1},  # no name — must be skipped
                {"name": "nomic-embed-text", "size": "274000000"},  # str size coerces
            ]
        }
        b.session.get.return_value = resp
        out = b.list_installed_models()
        assert [m["name"] for m in out] == ["qwen2.5:3b", "nomic-embed-text"]
        assert out[1]["size"] == 274_000_000


# ── OpenAICompatibleBackend ─────────────────────────────────────────────


class TestOpenAICompatibleBackend:
    """Targets the ``/v1/chat/completions`` wire shape."""

    def test_complete_posts_chat_completions_with_response_format(self):
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": '{"k": "v"}'}}]
        }
        b.session.post.return_value = resp

        out = b.complete("prompt", system_prompt="sys", json_format=True)
        assert out == '{"k": "v"}'

        call = b.session.post.call_args
        url = call.args[0] if call.args else call.kwargs["url"]
        assert url.endswith("/v1/chat/completions")
        payload = call.kwargs["json"]
        assert payload["model"] == CognitionConfig().model_llm_local
        assert payload["stream"] is False
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ]

    def test_complete_omits_response_format_when_json_false(self):
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "plain"}}]
        }
        b.session.post.return_value = resp

        b.complete("p", json_format=False)
        payload = b.session.post.call_args.kwargs["json"]
        assert "response_format" not in payload

    def test_complete_omits_empty_system_message(self):
        # An empty system prompt would otherwise eat into the model's
        # context budget on some hosted providers.
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        b.session.post.return_value = resp

        b.complete("p", system_prompt="")
        msgs = b.session.post.call_args.kwargs["json"]["messages"]
        assert msgs == [{"role": "user", "content": "p"}]

    def test_complete_returns_none_on_empty_choices(self):
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {"choices": []}
        b.session.post.return_value = resp
        assert b.complete("p") is None

    def test_complete_returns_none_on_http_error(self):
        b = _openai(llm_base_url="http://localhost:8080")
        b.session.post.side_effect = RuntimeError("boom")
        assert b.complete("p") is None

    def test_bearer_token_attached_when_env_set(self, monkeypatch):
        monkeypatch.setenv("GRIMORE_LLM_API_KEY", "sk-test")
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        b.session.post.return_value = resp

        b.complete("p")
        headers = b.session.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test"

    def test_no_authorization_header_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("GRIMORE_LLM_API_KEY", raising=False)
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        b.session.post.return_value = resp

        b.complete("p")
        headers = b.session.post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_custom_api_key_env_is_honoured(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "abc123")
        b = _openai(
            llm_base_url="http://localhost:8080",
            llm_api_key_env="MY_KEY",
        )
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        b.session.post.return_value = resp

        b.complete("p")
        headers = b.session.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer abc123"

    def test_streaming_parses_sse(self):
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            "",  # SSE blank line between events
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        resp.iter_lines.return_value = iter(chunks)
        b.session.post.return_value = resp

        out = "".join(b.complete_streaming("q"))
        assert out == "Hello"

    def test_streaming_tolerates_missing_prefix(self):
        # Some servers (LM Studio in certain modes) drop the "data: "
        # prefix. The backend should still parse JSON lines.
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.iter_lines.return_value = iter([
            '{"choices":[{"delta":{"content":"X"}}]}',
            'data: [DONE]',
        ])
        b.session.post.return_value = resp
        assert "".join(b.complete_streaming("q")) == "X"

    def test_streaming_returns_empty_on_error(self):
        b = _openai(llm_base_url="http://localhost:8080")
        b.session.post.side_effect = RuntimeError("boom")
        assert list(b.complete_streaming("q")) == []

    def test_list_models_maps_id_to_name(self):
        b = _openai(llm_base_url="http://localhost:8080")
        resp = MagicMock()
        resp.json.return_value = {
            "data": [
                {"id": "qwen2.5-3b"},
                {"id": ""},  # empty id — skipped
                {"id": "nomic-embed-text"},
            ]
        }
        b.session.get.return_value = resp
        out = b.list_installed_models()
        assert [m["name"] for m in out] == ["qwen2.5-3b", "nomic-embed-text"]
        assert all(m["size"] == 0 for m in out)

    def test_default_base_url_is_loopback_llama_cpp(self, monkeypatch):
        # No env, no config — defaults to llama.cpp's stock port.
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        b = OpenAICompatibleBackend(_cfg())
        assert b.base_url == "http://localhost:8080"

    def test_env_overrides_default_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9000")
        b = OpenAICompatibleBackend(_cfg())
        assert b.base_url == "http://127.0.0.1:9000"


# ── Security gate ───────────────────────────────────────────────────────


class TestBackendSecurityGate:
    """Both backends route through SecurityGuard.validate_llm_host."""

    def test_openai_rejects_non_loopback_without_allow_remote(self):
        with pytest.raises(ValueError, match="non-loopback"):
            OpenAICompatibleBackend(_cfg(
                llm_base_url="http://203.0.113.5:8080",
            ))

    def test_openai_requires_https_for_remote(self):
        with pytest.raises(ValueError, match="https"):
            OpenAICompatibleBackend(_cfg(
                allow_remote=True,
                llm_base_url="http://203.0.113.5:8080",
            ))

    def test_openai_accepts_remote_https(self):
        b = OpenAICompatibleBackend(_cfg(
            allow_remote=True,
            llm_base_url="https://api.openai.com",
        ))
        assert b.base_url == "https://api.openai.com"


# ── Factory + router dispatch ───────────────────────────────────────────


class TestBuildBackend:
    def test_default_picks_ollama(self):
        b = build_backend(_cfg())
        assert isinstance(b, OllamaBackend)

    def test_openai_engine_picks_openai(self):
        b = build_backend(_cfg(llm_backend="openai", llm_base_url="http://localhost:8080"))
        assert isinstance(b, OpenAICompatibleBackend)

    def test_unknown_backend_falls_back_to_ollama(self):
        b = build_backend(_cfg(llm_backend="moonbeams"))
        assert isinstance(b, OllamaBackend)


class TestRouterDispatch:
    def test_router_complete_runs_json_extraction_on_backend_text(self):
        # The OpenAI backend returns raw text; the router extracts JSON
        # so the rest of the codebase can call `router.complete(..., json_format=True)`
        # and get a dict back regardless of which backend is active.
        cfg = _cfg(llm_backend="openai", llm_base_url="http://localhost:8080")
        router = LLMRouter(cfg)
        router.backend.session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": 'noise {"answer": "42"} trailing'}}]
        }
        router.backend.session.post.return_value = resp

        out = router.complete("p", json_format=True)
        assert out == {"answer": "42"}

    def test_router_failure_advances_circuit_breaker(self):
        cfg = _cfg()
        router = LLMRouter(cfg)
        router.backend.session = MagicMock()
        router.backend.session.post.side_effect = RuntimeError("boom")
        for _ in range(LLMRouter._FAILURE_THRESHOLD):
            router.complete("p")
        assert router._circuit_open()

    def test_router_streaming_routes_to_backend(self):
        cfg = _cfg(llm_backend="openai", llm_base_url="http://localhost:8080")
        router = LLMRouter(cfg)
        router.backend.session = MagicMock()
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: [DONE]',
        ])
        router.backend.session.post.return_value = resp

        out = "".join(router.complete_streaming("q"))
        assert out == "A"

    def test_router_circuit_open_short_circuits_complete(self):
        cfg = _cfg()
        router = LLMRouter(cfg)
        router._open_until = float("inf")
        # Backend mock would error if reached — it shouldn't be.
        router.backend.session = MagicMock(side_effect=AssertionError("backend reached"))
        assert router.complete("p") is None

    def test_router_circuit_open_short_circuits_streaming(self):
        cfg = _cfg()
        router = LLMRouter(cfg)
        router._open_until = float("inf")
        router.backend.session = MagicMock(side_effect=AssertionError("backend reached"))
        assert list(router.complete_streaming("p")) == []

    def test_router_explicit_backend_wins_over_config(self):
        # Injection point used by tests + e2e harnesses.
        stub = MagicMock()
        stub.complete.return_value = "raw"
        stub.complete_streaming.return_value = iter(["x"])
        cfg = _cfg(llm_backend="openai", llm_base_url="http://localhost:8080")
        router = LLMRouter(cfg, backend=stub)
        assert router.backend is stub
        assert router.complete("p", json_format=False) == "raw"
