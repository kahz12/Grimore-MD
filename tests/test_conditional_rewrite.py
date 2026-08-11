"""A follow-up is only rewritten when it actually points at the previous turn.

Rewriting costs a full LLM round-trip, and most follow-ups do not need one: a
question that names its own subject retrieves the same documents whether or not
the history is folded into it. These tests pin which questions the heuristic
classifies as needing resolution.

The bias is deliberate and asymmetric: a false positive wastes one round-trip,
a false negative retrieves against an unresolved pronoun and can lose the
answer. Anything borderline should therefore come out True, and the tests below
encode that rather than treating the two errors as equal.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from grimore.cognition.oracle import Oracle

_HISTORY = [{"q": "What is Kotlin?", "a": "A JVM language by JetBrains."}]


def _oracle(conditional=True):
    o = Oracle.__new__(Oracle)
    o.router = MagicMock()
    o.conditional_rewrite = conditional
    return o


class TestNeedsRewrite:
    def test_no_history_never_needs_rewriting(self):
        assert Oracle._needs_rewrite("Which company created it?", None) is False
        assert Oracle._needs_rewrite("Which company created it?", []) is False

    @pytest.mark.parametrize("question", [
        # English pronouns and demonstratives.
        "Which company created it?",
        "And what else is in it?",
        "Which one carried the lateral thrust outward?",
        "Which early warlord generals concentrated military power in that period?",
        "Which psychological principle does it exploit when posing as a manager?",
        "How do they differ from the earlier ones?",
        "What made him change his mind about that?",
        # Spanish.
        "¿En qué guerra sirvió él como hoplita?",
        "¿Y dónde nació él?",
        "¿Y con qué se observa?",
        "¿Y cuáles son sus desventajas?",
        "¿Y qué distingue su piel de la de los reptiles?",
        "¿Qué pasó entonces con ese imperio?",
        # Too short to retrieve on regardless of vocabulary.
        "¿Por qué?",
        "Expand on that",
        "And when?",
    ])
    def test_referential_questions_need_rewriting(self, question):
        assert Oracle._needs_rewrite(question, _HISTORY) is True

    @pytest.mark.parametrize("question", [
        # Self-contained: they name their own subject.
        "What does a Kotlin data class generate for you?",
        "What is pretexting in social engineering?",
        "¿Qué es el método mayéutico de Sócrates?",
        "¿Qué etapas atraviesa la metamorfosis de los anuros?",
        "How did Rome transition from the Republic to the Empire?",
        "What three structural innovations defined Gothic cathedrals?",
        "¿Qué caracteriza a un monolito tradicional en arquitectura de software?",
    ])
    def test_self_contained_questions_are_left_alone(self, question):
        assert Oracle._needs_rewrite(question, _HISTORY) is False

    def test_the_spanish_article_is_not_a_pronoun(self):
        """Accent folding turns "él" into "el", the definite article, which
        appears in almost any Spanish sentence. Folding both together made a
        fully self-contained question look like a follow-up.
        """
        assert Oracle._needs_rewrite(
            "¿Qué es el método mayéutico de Sócrates?", _HISTORY) is False
        assert Oracle._needs_rewrite(
            "¿En qué guerra sirvió él como hoplita?", _HISTORY) is True

    def test_first_and_second_person_are_not_referential(self):
        """"you" and "I" point at the speaker, not at an earlier turn."""
        assert Oracle._needs_rewrite(
            "What does a Kotlin data class generate for you?", _HISTORY) is False
        assert Oracle._needs_rewrite(
            "Can I use coroutines with Kotlin Multiplatform?", _HISTORY) is False

    def test_spanish_articles_do_not_trigger(self):
        # la / los / las are articles far more often than pronouns.
        assert Oracle._needs_rewrite(
            "¿Cuáles son las fases de la metamorfosis en los anuros?",
            _HISTORY) is False

    def test_a_question_with_no_words_is_rewritten(self):
        assert Oracle._needs_rewrite("???", _HISTORY) is True
        assert Oracle._needs_rewrite("", _HISTORY) is True

    def test_unaccented_spanish_still_matches(self):
        # Users type without accents constantly.
        assert Oracle._needs_rewrite(
            "¿Que paso entonces con ese imperio?", _HISTORY) is True


class TestRewriteWiring:
    def test_a_self_contained_follow_up_does_not_call_the_llm(self):
        o = _oracle()
        out = o._rewrite_query("What is pretexting in social engineering?", _HISTORY)
        assert out == "What is pretexting in social engineering?"
        o.router.complete.assert_not_called()

    def test_a_referential_follow_up_still_calls_the_llm(self):
        o = _oracle()
        o.router.complete.return_value = {"query": "Which company created Kotlin?"}
        out = o._rewrite_query("Which company created it?", _HISTORY)
        assert out == "Which company created Kotlin?"
        o.router.complete.assert_called_once()

    def test_disabling_the_flag_restores_rewrite_everywhere(self):
        o = _oracle(conditional=False)
        o.router.complete.return_value = {"query": "rewritten"}
        out = o._rewrite_query("What is pretexting in social engineering?", _HISTORY)
        assert out == "rewritten"
        o.router.complete.assert_called_once()

    def test_no_history_short_circuits_before_the_heuristic(self):
        o = _oracle()
        assert o._rewrite_query("Which company created it?", None) == \
            "Which company created it?"
        o.router.complete.assert_not_called()

    def test_a_failed_rewrite_still_degrades_to_the_original(self):
        o = _oracle()
        o.router.complete.side_effect = RuntimeError("ollama down")
        assert o._rewrite_query("Which company created it?", _HISTORY) == \
            "Which company created it?"


class TestConfigWiring:
    def _config(self, **kw):
        return SimpleNamespace(cognition=SimpleNamespace(**kw))

    def test_default_is_on(self):
        oracle = Oracle(
            self._config(context_max_chars=16000),
            MagicMock(), MagicMock(), MagicMock(),
        )
        assert oracle.conditional_rewrite is True

    def test_config_can_turn_it_off(self):
        oracle = Oracle(
            self._config(context_max_chars=16000, conditional_rewrite=False),
            MagicMock(), MagicMock(), MagicMock(),
        )
        assert oracle.conditional_rewrite is False

    def test_bare_instance_falls_back_to_the_class_default(self):
        assert Oracle.__new__(Oracle).conditional_rewrite is True


class TestRewriteTimeout:
    """The rewrite blocks retrieval with nothing on screen, so it gets its own
    deadline rather than the generation-sized request timeout."""

    def test_the_rewrite_passes_its_own_budget(self):
        o = _oracle()
        o.rewrite_timeout_s = 7
        o.router.complete.return_value = {"query": "resolved"}
        o._rewrite_query("Which company created it?", _HISTORY)
        assert o.router.complete.call_args.kwargs["timeout_s"] == 7

    def test_default_budget_is_far_below_the_request_timeout(self):
        from grimore.cognition.oracle import _REWRITE_TIMEOUT_S
        oracle = Oracle(
            SimpleNamespace(cognition=SimpleNamespace(
                context_max_chars=16000, request_timeout_s=600)),
            MagicMock(), MagicMock(), MagicMock(),
        )
        assert oracle.rewrite_timeout_s == _REWRITE_TIMEOUT_S
        assert oracle.rewrite_timeout_s < 600

    def test_config_overrides_the_budget(self):
        oracle = Oracle(
            SimpleNamespace(cognition=SimpleNamespace(
                context_max_chars=16000, rewrite_timeout_s=3)),
            MagicMock(), MagicMock(), MagicMock(),
        )
        assert oracle.rewrite_timeout_s == 3

    def test_a_timeout_degrades_to_the_original_question(self):
        o = _oracle()
        o.rewrite_timeout_s = 1
        # A backend timeout surfaces as None from the router, not an exception.
        o.router.complete.return_value = None
        assert o._rewrite_query("Which company created it?", _HISTORY) == \
            "Which company created it?"


class TestBackendTimeoutPassthrough:
    def test_ollama_uses_the_per_call_budget(self):
        from grimore.cognition.llm_backends.ollama import OllamaBackend
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.config = SimpleNamespace(
            cognition=SimpleNamespace(model_llm_local="m", request_timeout_s=600))
        backend.host = "http://127.0.0.1:11434"
        backend.session = MagicMock()
        backend.session.post.return_value.json.return_value = {"response": "{}"}
        backend.complete("p", timeout_s=5)
        assert backend.session.post.call_args.kwargs["timeout"] == 5

    def test_ollama_falls_back_to_the_configured_timeout(self):
        from grimore.cognition.llm_backends.ollama import OllamaBackend
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.config = SimpleNamespace(
            cognition=SimpleNamespace(model_llm_local="m", request_timeout_s=600))
        backend.host = "http://127.0.0.1:11434"
        backend.session = MagicMock()
        backend.session.post.return_value.json.return_value = {"response": "{}"}
        backend.complete("p")
        assert backend.session.post.call_args.kwargs["timeout"] == 600


class TestOptionalCallIsolation:
    """A best-effort call on a tight deadline must not disable the LLM for
    everything else. Rewrite timeouts opened the shared circuit breaker, which
    then cancelled both further rewrites AND answer generation."""

    def _router(self):
        from grimore.cognition.llm_router import LLMRouter
        from grimore.utils.config import (
            CognitionConfig, Config, MemoryConfig, VaultConfig,
        )
        cfg = Config(vault=VaultConfig(path="./v"), cognition=CognitionConfig(),
                     memory=MemoryConfig(db_path=":memory:"))
        router = LLMRouter(cfg)
        router.backend = MagicMock()
        return router

    def test_optional_failures_do_not_open_the_breaker(self):
        router = self._router()
        router.backend.complete.return_value = None
        for _ in range(router.failure_threshold + 3):
            router.complete("p", optional=True)
        assert not router._circuit_open()

    def test_normal_failures_still_open_the_breaker(self):
        router = self._router()
        router.backend.complete.return_value = None
        for _ in range(router.failure_threshold):
            router.complete("p")
        assert router._circuit_open()

    def test_an_optional_call_runs_even_with_the_breaker_open(self):
        router = self._router()
        router.backend.complete.return_value = None
        for _ in range(router.failure_threshold):
            router.complete("p")
        assert router._circuit_open()

        router.backend.complete.reset_mock()
        router.backend.complete.return_value = '{"query": "ok"}'
        assert router.complete("p", optional=True) == {"query": "ok"}
        router.backend.complete.assert_called_once()

    def test_the_rewrite_marks_itself_optional(self):
        o = _oracle()
        o.rewrite_timeout_s = 60
        o.router.complete.return_value = {"query": "resolved"}
        o._rewrite_query("Which company created it?", _HISTORY)
        assert o.router.complete.call_args.kwargs["optional"] is True
