from unittest.mock import MagicMock

import pytest

from grimore.cognition.llm_router import LLMRouter, _extract_json_object
from grimore.utils.config import CognitionConfig, Config


class TestExtractJsonObject:
    def test_direct_parse(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_block(self):
        text = 'some prose\n```json\n{"tags": ["a", "b"]}\n```\nmore prose'
        assert _extract_json_object(text) == {"tags": ["a", "b"]}

    def test_unlabeled_fence(self):
        text = '```\n{"x": 2}\n```'
        assert _extract_json_object(text) == {"x": 2}

    def test_bracket_balanced_fallback(self):
        text = 'here is the answer: {"a": 1, "b": {"c": 2}} and then trailing noise'
        assert _extract_json_object(text) == {"a": 1, "b": {"c": 2}}

    def test_braces_inside_strings_do_not_confuse(self):
        text = '{"key": "value with } brace inside"}'
        assert _extract_json_object(text) == {"key": "value with } brace inside"}

    def test_escaped_quotes_respected(self):
        text = '{"k": "he said \\"hi\\""}'
        assert _extract_json_object(text) == {"k": 'he said "hi"'}

    def test_returns_none_on_empty(self):
        assert _extract_json_object("") is None

    def test_returns_none_on_no_braces(self):
        assert _extract_json_object("just plain prose here") is None

    def test_returns_none_for_non_dict_top_level(self):
        # A JSON array is not a dict — this function only returns dicts
        assert _extract_json_object("[1, 2, 3]") is None

    def test_returns_none_on_malformed(self):
        assert _extract_json_object("{not valid json at all") is None

    def test_prefers_direct_parse_when_clean(self):
        text = '{"outer": {"inner": 1}}'
        assert _extract_json_object(text) == {"outer": {"inner": 1}}


def _router_with_config(cognition: CognitionConfig) -> LLMRouter:
    cfg = Config()
    cfg.cognition = cognition
    router = LLMRouter(cfg)
    router.session = MagicMock()
    return router


class TestRequestTimeoutWiring:
    """Per-call timeouts must come from CognitionConfig, not be hard-coded."""

    def test_complete_uses_request_timeout_from_config(self):
        router = _router_with_config(CognitionConfig(request_timeout_s=181))
        resp = MagicMock()
        resp.json.return_value = {"response": '{"answer": "ok"}'}
        router.session.post.return_value = resp

        router.complete("hi", system_prompt="sys")

        kwargs = router.session.post.call_args.kwargs
        assert kwargs["timeout"] == 181

    def test_complete_streaming_uses_stream_timeout_from_config(self):
        router = _router_with_config(CognitionConfig(stream_timeout_s=240))
        # context manager protocol over a streaming response
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.iter_lines.return_value = iter([])
        router.session.post.return_value = resp

        list(router.complete_streaming("hi", system_prompt="sys"))

        kwargs = router.session.post.call_args.kwargs
        assert kwargs["timeout"] == 240

    def test_defaults_match_pre_config_behavior(self):
        cfg = CognitionConfig()
        assert cfg.request_timeout_s == 60
        assert cfg.stream_timeout_s == 120
        assert cfg.embed_timeout_s == 30


class TestListInstalledModels:
    def test_returns_models_in_response_order(self):
        router = _router_with_config(CognitionConfig())
        resp = MagicMock()
        resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:3b", "size": 2_000_000_000},
                {"name": "nomic-embed-text", "size": 274_000_000},
            ]
        }
        router.session.get.return_value = resp

        out = router.list_installed_models()

        assert [m["name"] for m in out] == ["qwen2.5:3b", "nomic-embed-text"]
        assert out[0]["size"] == 2_000_000_000

    def test_empty_list_on_http_failure(self):
        router = _router_with_config(CognitionConfig())
        router.session.get.side_effect = RuntimeError("boom")
        assert router.list_installed_models() == []

    def test_skips_entries_without_name(self):
        router = _router_with_config(CognitionConfig())
        resp = MagicMock()
        resp.json.return_value = {"models": [{"size": 1}, {"name": "ok"}]}
        router.session.get.return_value = resp
        assert [m["name"] for m in router.list_installed_models()] == ["ok"]
