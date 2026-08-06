"""Regression: the promoted config keys actually reach their consumers.

tests/test_config.py proves the keys load and default correctly. That is not
enough on its own: it would pass unchanged if a consumer were never wired up
and kept reading its old module constant. These tests assert the observable
behaviour each key controls, so removing a wiring line fails a test rather
than silently reverting the feature.
"""
from unittest.mock import MagicMock

from grimore.cognition.embedder import _EMBED_BATCH_SIZE, Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.oracle import _ORACLE_CONTEXT_MAX_CHARS, Oracle
from grimore.session import Session
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MemoryConfig,
    ShellConfig,
    VaultConfig,
)


def _config(**cognition_kwargs) -> Config:
    return Config(
        vault=VaultConfig(path="./vault"),
        cognition=CognitionConfig(**cognition_kwargs),
        memory=MemoryConfig(db_path=":memory:"),
    )


class TestOracleContextCap:
    def _oracle(self, config) -> Oracle:
        # Mirrors tests/test_oracle.py: build the instance without __init__ so
        # _build_context is exercised without a live DB, router or embedder,
        # then apply only the attribute under test.
        o = Oracle.__new__(Oracle)
        o.db = MagicMock()
        o.router = MagicMock()
        o.embedder = MagicMock()
        o.connector = MagicMock()
        o.config = config
        o.system_prompt_template = "TEMPLATE: {context}"
        o.db.fts_available = False
        o.db.get_note_titles.side_effect = lambda nids: {
            nid: f"Note {nid}" for nid in nids
        }
        o.db.get_chunk_anchors_bulk.side_effect = lambda pairs: dict.fromkeys(pairs, (None, None))
        o.embedder.embed.return_value = [0.0] * 16
        o.context_max_chars = config.cognition.context_max_chars
        return o

    def _context_for(self, cap: int) -> str:
        config = _config(context_max_chars=cap, hybrid_search=False)
        oracle = self._oracle(config)
        oracle.connector.find_similar_notes.return_value = [
            {"note_id": i, "text": "z" * 2_000, "score": 1.0} for i in range(20)
        ]
        context, _sources, _dropped = oracle._build_context("q", top_k=20)[:3]
        return context

    def test_smaller_cap_truncates_context_further(self):
        wide = self._context_for(16_000)
        narrow = self._context_for(4_000)
        assert len(narrow) <= 4_000
        assert len(narrow) < len(wide)

    def test_real_init_reads_the_cap_from_config(self):
        # Deliberately a fully constructed Oracle, not the __new__ stand-in
        # above: this is the assertion that fails if __init__ stops reading
        # the config key. Asserting it on a hand-populated instance would be
        # tautological -- it would only re-check the value the test just set.
        oracle = Oracle(
            _config(context_max_chars=4_321), MagicMock(), MagicMock(), MagicMock()
        )
        assert oracle.context_max_chars == 4_321

    def test_real_init_defaults_to_the_historical_cap(self):
        oracle = Oracle(_config(), MagicMock(), MagicMock(), MagicMock())
        assert oracle.context_max_chars == _ORACLE_CONTEXT_MAX_CHARS

    def test_bare_instance_falls_back_to_the_class_default(self):
        # Oracle.__new__ with no attributes set at all: the class-level
        # default is what keeps _build_context working for the isolation
        # tests that predate this key.
        assert Oracle.__new__(Oracle).context_max_chars == _ORACLE_CONTEXT_MAX_CHARS


class TestEmbedderBatchSize:
    def test_batch_size_comes_from_config(self):
        assert Embedder(_config(embed_batch_size=8)).batch_size == 8

    def test_default_matches_the_module_constant(self):
        assert Embedder(_config()).batch_size == _EMBED_BATCH_SIZE

    def test_non_positive_batch_size_is_floored_to_one(self):
        # A zero step would make embed_batch's range() loop never advance.
        assert Embedder(_config(embed_batch_size=0)).batch_size == 1

    def test_embed_batch_splits_requests_at_the_configured_size(self):
        # Asserting the attribute alone is not enough: it passes even if
        # embed_batch keeps stepping by the old module constant. Counting the
        # sub-batches is what proves the value is actually consumed.
        embedder = Embedder(_config(embed_batch_size=4))
        seen: list[int] = []

        def fake_remote(texts):
            seen.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

        embedder._embed_many_remote = fake_remote
        embedder.embed_batch([f"text {i}" for i in range(10)])
        assert seen == [4, 4, 2]


class TestCircuitBreakerTunables:
    def test_threshold_and_cooldown_come_from_config(self):
        router = LLMRouter(_config(circuit_failure_threshold=2, circuit_cooldown_s=30))
        assert router.failure_threshold == 2
        assert router.cooldown_s == 30

    def test_breaker_opens_at_the_configured_threshold(self):
        router = LLMRouter(_config(circuit_failure_threshold=2))
        router._record_failure()
        assert not router._circuit_open()
        router._record_failure()
        assert router._circuit_open()

    def test_defaults_preserve_the_historical_behaviour(self):
        router = LLMRouter(_config())
        assert router.failure_threshold == LLMRouter._FAILURE_THRESHOLD
        assert router.cooldown_s == LLMRouter._COOLDOWN_SECONDS
        for _ in range(LLMRouter._FAILURE_THRESHOLD - 1):
            router._record_failure()
        assert not router._circuit_open()
        router._record_failure()
        assert router._circuit_open()


class TestSessionTurnWindow:
    def _session(self, max_turns: int) -> Session:
        config = _config()
        config.shell = ShellConfig(max_turns=max_turns)
        return Session(config)

    def test_window_comes_from_config(self):
        session = self._session(5)
        for i in range(8):
            session.record_turn(f"q{i}", f"a{i}", [])
        assert len(session.turns) == 5
        assert session.turns[-1]["q"] == "q7"

    def test_default_keeps_three_turns(self):
        session = Session(_config())
        for i in range(6):
            session.record_turn(f"q{i}", f"a{i}", [])
        assert len(session.turns) == Session.MAX_TURNS

    def test_zero_disables_memory_rather_than_keeping_everything(self):
        # Guards the slicing edge case: turns[-0:] is turns[0:], so folding
        # zero into the slice would retain the whole history instead of
        # dropping it.
        session = self._session(0)
        for i in range(4):
            session.record_turn(f"q{i}", f"a{i}", [])
        assert session.turns == []

    def test_zero_window_also_empties_a_loaded_thread(self, tmp_path):
        source = self._session(3)
        for i in range(3):
            source.record_turn(f"q{i}", f"a{i}", [])
        path = source.save_turns(tmp_path / "thread.jsonl")

        session = self._session(0)
        assert session.load_turns(path) == 0
        assert session.turns == []
