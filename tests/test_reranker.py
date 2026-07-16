"""
Re-ranker module + Connector dispatch.

Two backends share the :class:`Reranker` protocol so the Connector can
swap implementations without ripple. These tests pin down:

* ``LLMReranker``: prompt+parse logic, soft failure modes (no router,
  empty / malformed JSON, partial scoring) — returns ``[]`` on failure
  so the Connector keeps the upstream RRF order.
* ``CrossEncoderReranker``: import-time guard when the extra is missing
  (raises a clean ImportError naming the install command); real-model
  scoring is marked ``reranker`` and auto-skips when the extra isn't
  installed.
* ``build_reranker``: engine selection + silent fallback chain.
* ``Connector._rerank``: head/tail slicing, stable order on ties,
  reorder by scores, no-op when fewer than 2 head items or when the
  reranker returns ``[]``.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from grimore.cognition.connector import Connector
from grimore.cognition.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    build_reranker,
)


# ── LLMReranker ─────────────────────────────────────────────────────────


class TestLLMReranker:
    def _router(self, payload):
        r = MagicMock()
        r.complete.return_value = payload
        return r

    def test_parses_scores_aligned_to_input(self):
        router = self._router(
            {"scores": [{"index": 0, "score": 7}, {"index": 2, "score": 9}]}
        )
        r = LLMReranker(router)
        scores = r.score("q", ["a", "b", "c"])
        assert scores[0] == 7.0
        assert scores[2] == 9.0
        # Unscored index stays at -inf so a stable sort sinks it.
        assert scores[1] == float("-inf")

    def test_empty_on_no_router(self):
        assert LLMReranker(None).score("q", ["a", "b"]) == []

    def test_empty_on_single_passage(self):
        # No work to do — caller should keep candidates as-is.
        assert LLMReranker(self._router({"scores": []})).score("q", ["only"]) == []

    @pytest.mark.parametrize("bad", [None, "garbage", {"scores": "nope"}, {}, {"scores": []}])
    def test_empty_on_garbage_response(self, bad):
        r = LLMReranker(self._router(bad))
        assert r.score("q", ["a", "b", "c"]) == []

    def test_router_exception_returns_empty(self):
        router = MagicMock()
        router.complete.side_effect = RuntimeError("network down")
        assert LLMReranker(router).score("q", ["a", "b"]) == []

    def test_out_of_range_index_ignored(self):
        # A model that hallucinates an index >= len(passages) shouldn't
        # IndexError or corrupt the scores list.
        router = self._router({"scores": [{"index": 99, "score": 5}]})
        assert LLMReranker(router).score("q", ["a", "b"]) == []


# ── CrossEncoderReranker ────────────────────────────────────────────────


class TestCrossEncoderReranker:
    def test_missing_extra_raises_clean_import_error(self, monkeypatch):
        # Simulate sentence-transformers not being installed.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match=r"reranker.*extra"):
            CrossEncoderReranker()

    @pytest.mark.reranker
    def test_real_model_scores(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        r = CrossEncoderReranker()
        scores = r.score(
            "What is the capital of France?",
            ["Paris is the capital of France.", "Bananas are yellow."],
        )
        assert len(scores) == 2
        # The relevant passage should outrank the irrelevant one.
        assert scores[0] > scores[1]


# ── build_reranker ──────────────────────────────────────────────────────


class TestBuildReranker:
    def test_llm_engine_returns_llm_reranker_when_router_present(self):
        r = build_reranker("llm", router=MagicMock())
        assert isinstance(r, LLMReranker)

    def test_llm_engine_returns_none_when_no_router(self):
        # No router + no cross-encoder → re-rank simply not available.
        assert build_reranker("llm", router=None) is None

    def test_cross_encoder_engine_falls_back_to_llm_when_extra_missing(
        self, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        r = build_reranker("cross-encoder", router=MagicMock())
        # Cross-encoder extra is gone, but the router can still rerank
        # via the LLM path — Connector shouldn't blow up here.
        assert isinstance(r, LLMReranker)

    def test_cross_encoder_with_no_router_returns_none_when_extra_missing(
        self, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        assert build_reranker("cross-encoder", router=None) is None


# ── Connector._rerank dispatch ──────────────────────────────────────────


class _StubReranker:
    """Deterministic stand-in: returns whatever scores it was handed."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        # Trim/pad so we exercise both the strict alignment branch (when
        # len matches) and the fall-through (when it doesn't).
        return self.scores[: len(passages)]


def _connector_with(reranker):
    c = Connector.__new__(Connector)
    c.router = None
    c._reranker = reranker
    return c


class TestConnectorRerank:
    def _cands(self, n):
        return [{"note_id": i, "text": f"p{i}", "score": 1.0} for i in range(n)]

    def test_reorders_by_reranker_scores(self):
        reranker = _StubReranker([1.0, 9.0, 5.0, 2.0])
        c = _connector_with(reranker)
        out = c._rerank("q", self._cands(4), pool=4)
        assert [d["note_id"] for d in out] == [1, 2, 3, 0]
        assert reranker.calls[0] == ("q", ["p0", "p1", "p2", "p3"])

    def test_stable_on_ties(self):
        # All equal → original order preserved.
        c = _connector_with(_StubReranker([5.0, 5.0, 5.0, 5.0]))
        out = c._rerank("q", self._cands(4), pool=4)
        assert [d["note_id"] for d in out] == [0, 1, 2, 3]

    def test_tail_preserved_when_pool_below_total(self):
        # pool=2 → only the first two get reranked; rest passthrough.
        c = _connector_with(_StubReranker([0.0, 10.0]))
        out = c._rerank("q", self._cands(5), pool=2)
        assert [d["note_id"] for d in out] == [1, 0, 2, 3, 4]

    def test_no_reranker_is_no_op(self):
        c = _connector_with(None)
        cands = self._cands(3)
        assert c._rerank("q", cands, pool=3) is cands

    def test_empty_scores_keep_original_order(self):
        c = _connector_with(_StubReranker([]))
        cands = self._cands(3)
        assert c._rerank("q", cands, pool=3) == cands

    def test_single_head_item_no_op(self):
        c = _connector_with(_StubReranker([1.0]))
        cands = self._cands(1)
        assert c._rerank("q", cands, pool=1) is cands

    def test_partial_scores_keep_original_order(self):
        # Length mismatch → keep order, don't IndexError or partial-sort.
        c = _connector_with(_StubReranker([99.0]))  # only 1 of 3
        cands = self._cands(3)
        assert c._rerank("q", cands, pool=3) == cands


# ── Connector construction picks the right backend ─────────────────────


class TestConnectorBackendPick:
    def _build(self, **kw):
        db = MagicMock()
        db.vec_available = False
        emb = MagicMock()
        return Connector(db, emb, router=MagicMock(), **kw)

    def test_default_engine_picks_llm(self):
        c = self._build()
        assert isinstance(c._reranker, LLMReranker)

    def test_cross_encoder_falls_back_to_llm_when_extra_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        c = self._build(rerank_engine="cross-encoder")
        assert isinstance(c._reranker, LLMReranker)

    def test_explicit_reranker_injection_wins(self):
        sentinel = _StubReranker([1.0])
        db = MagicMock()
        db.vec_available = False
        c = Connector(db, MagicMock(), router=MagicMock(), reranker=sentinel)
        assert c._reranker is sentinel

    def test_no_router_no_extra_yields_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        db = MagicMock()
        db.vec_available = False
        c = Connector(db, MagicMock(), router=None, rerank_engine="cross-encoder")
        assert c._reranker is None


# ── Reranker protocol shape ─────────────────────────────────────────────


def test_protocol_shape_satisfied_structurally():
    # The Reranker Protocol isn't @runtime_checkable (keeps it cheap),
    # so we duck-check by calling .score(...) on each concrete backend
    # we ship. A real CrossEncoderReranker can't be constructed without
    # the extra; its shape is exercised by the stub here.
    router = MagicMock()
    router.complete.return_value = {"scores": []}
    assert LLMReranker(router).score("q", ["a"]) == []
    assert _StubReranker([1.0]).score("q", ["a"]) == [1.0]
