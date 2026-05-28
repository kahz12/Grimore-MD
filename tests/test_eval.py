"""Unit tests for the eval harness (mocked router/embedder; no live model).

Live coverage of the harness end-to-end against real Ollama lives in
``tests/test_e2e_oracle.py`` so this file can run in milliseconds and gate CI.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from grimore.cognition.eval import (
    EvalCase,
    EvalReport,
    TurnResult,
    export_report,
    faithfulness,
    judge_relevance,
    keyword_recall,
    load_golden,
    mrr,
    recall_at_k,
    run_eval,
)


# ── metric formulas ────────────────────────────────────────────────────────


class TestMetrics:
    def test_recall_at_k_full_partial_miss(self):
        # Two of three expected found, one missing.
        assert recall_at_k(["A", "B", "C#p.1"], ["A", "C", "Z"]) == pytest.approx(2 / 3)

    def test_recall_at_k_strips_anchors(self):
        assert recall_at_k(["Gothic#p.3"], ["Gothic"]) == 1.0

    def test_recall_at_k_empty_expected_is_vacuously_full(self):
        assert recall_at_k(["A"], []) == 1.0

    def test_mrr_first_hit_at_rank_2(self):
        # MRR rewards earlier hits — the first expected here is at rank 2.
        assert mrr(["X", "A#p.4", "B"], ["A", "B"]) == 0.5

    def test_mrr_no_hit_is_zero(self):
        assert mrr(["X", "Y"], ["A"]) == 0.0

    def test_mrr_empty_expected_is_full(self):
        assert mrr(["X"], []) == 1.0

    def test_faithfulness_no_citations_is_full(self):
        # Citation-free answer can't be unfaithful — score is 1.0.
        score, total = faithfulness("plain prose, no links", 0)
        assert score == 1.0 and total == 0

    def test_faithfulness_partial_drop(self):
        # Two citations kept in the cleaned text + one dropped = 1/3 ungrounded.
        score, total = faithfulness("Per [[A]] and [[B#p.1]].", 1)
        assert total == 3
        assert score == pytest.approx(2 / 3)

    def test_keyword_recall_case_insensitive(self):
        assert keyword_recall("Pointed Arches matter.", ["pointed", "ARCHES"]) == 1.0
        assert keyword_recall("Only one term.", ["one", "missing"]) == 0.5


# ── judge ───────────────────────────────────────────────────────────────────


class TestJudge:
    def test_returns_normalised_score(self):
        router = MagicMock()
        router.complete.return_value = {"score": 8, "reason": "thorough"}
        assert judge_relevance(router, "q", "a") == pytest.approx(0.8)

    def test_returns_none_when_router_returns_none(self):
        router = MagicMock()
        router.complete.return_value = None  # circuit open / Ollama down
        assert judge_relevance(router, "q", "a") is None

    def test_returns_none_when_score_missing_or_garbage(self):
        router = MagicMock()
        for resp in ({"nope": 1}, {"score": "high"}, {"score": None}, "not a dict"):
            router.complete.return_value = resp
            assert judge_relevance(router, "q", "a") is None

    def test_clamps_out_of_range(self):
        router = MagicMock()
        router.complete.return_value = {"score": 11.0}
        assert judge_relevance(router, "q", "a") == 1.0
        router.complete.return_value = {"score": -2.0}
        assert judge_relevance(router, "q", "a") == 0.0

    def test_returns_none_on_router_exception(self):
        router = MagicMock()
        router.complete.side_effect = RuntimeError("boom")
        assert judge_relevance(router, "q", "a") is None


# ── golden loader ──────────────────────────────────────────────────────────


class TestGoldenLoader:
    def _write(self, tmp_path, text: str) -> Path:
        p = tmp_path / "g.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_loads_minimal_case(self, tmp_path):
        cases = load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: q1
    question: "What?"
"""))
        assert len(cases) == 1
        assert cases[0].id == "q1" and cases[0].expected_sources == []

    def test_walks_followups(self, tmp_path):
        cases = load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: q1
    question: "What?"
    follow_ups:
      - id: q1a
        question: "Tell me more."
"""))
        assert cases[0].follow_ups[0].id == "q1a"

    def test_rejects_unknown_key(self, tmp_path):
        with pytest.raises(ValueError, match="unknown key"):
            load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: q1
    question: "What?"
    typo_field: 42
"""))

    def test_rejects_wrong_version(self, tmp_path):
        with pytest.raises(ValueError, match="schema version"):
            load_golden(self._write(tmp_path, """\
version: 999
questions: []
"""))

    def test_rejects_missing_id(self, tmp_path):
        with pytest.raises(ValueError, match="required 'id'"):
            load_golden(self._write(tmp_path, """\
version: 1
questions:
  - question: "Where's the id?"
"""))


# ── runner ──────────────────────────────────────────────────────────────────


class _FakeSession:
    """Minimum surface the runner pokes — keeps the test fast and isolated
    from Session's many lazy-built services."""

    def __init__(self, oracle_responses: list[dict]):
        self.oracle = MagicMock()
        self.oracle.ask.side_effect = list(oracle_responses)
        self.router = MagicMock()
        self.router.complete.return_value = {"score": 9}
        self.turns: list[dict] = []

    def forget(self):
        self.turns = []

    def record_turn(self, q, a, sources):
        self.turns.append({"q": q, "a": a, "sources": list(sources)})


class TestRunner:
    def test_walks_followups_with_history(self):
        session = _FakeSession([
            {"answer": "Pointed arch, ribbed vault, flying buttress.",
             "sources": ["gothic_architecture"], "dropped_citations": 0},
            {"answer": "The flying buttress transferred thrust outward.",
             "sources": ["gothic_architecture"], "dropped_citations": 0},
        ])
        cases = [EvalCase(
            id="g", question="What three?",
            expected_sources=["gothic_architecture"],
            expected_keywords=["pointed arch", "flying buttress"],
            follow_ups=[EvalCase(
                id="g-fu", question="Which carried thrust?",
                expected_sources=["gothic_architecture"],
                expected_keywords=["flying buttress"],
            )],
        )]
        report = run_eval(session, cases, top_k=3, judge=False)
        assert len(report.turns) == 2
        # Second ask must have received history (record_turn fed turn 0 in).
        second_call_kwargs = session.oracle.ask.call_args_list[1].kwargs
        assert "history" in second_call_kwargs
        assert second_call_kwargs["history"][0]["q"] == "What three?"
        # Recall + MRR are both perfect on this trivial set.
        agg = report.summary()["aggregate"]
        assert agg["recall_at_k"] == 1.0 and agg["mrr"] == 1.0

    def test_judge_off_means_relevance_none(self):
        session = _FakeSession([
            {"answer": "stub", "sources": ["A"], "dropped_citations": 0},
        ])
        cases = [EvalCase(id="x", question="q", expected_sources=["A"], expected_keywords=[])]
        report = run_eval(session, cases, top_k=1, judge=False)
        assert report.turns[0].answer_relevance is None
        assert session.router.complete.call_count == 0

    def test_oracle_failure_becomes_zero_recall_not_crash(self):
        session = _FakeSession([])  # exhausts immediately → StopIteration on ask
        session.oracle.ask.side_effect = RuntimeError("oracle exploded")
        cases = [EvalCase(id="x", question="q", expected_sources=["A"], expected_keywords=["x"])]
        report = run_eval(session, cases, top_k=1, judge=False)
        # The harness must report the failure as a graceful zero, not raise.
        assert report.turns[0].recall_at_k == 0.0
        assert report.turns[0].answer == ""

    def test_forget_runs_between_top_level_cases(self):
        session = _FakeSession([
            {"answer": "a1", "sources": [], "dropped_citations": 0},
            {"answer": "a2", "sources": [], "dropped_citations": 0},
        ])
        cases = [
            EvalCase(id="c1", question="q1"),
            EvalCase(id="c2", question="q2"),
        ]
        run_eval(session, cases, top_k=1, judge=False)
        # Second ask must not see c1's turn in history (forget cleared it).
        second_call_kwargs = session.oracle.ask.call_args_list[1].kwargs
        assert "history" not in second_call_kwargs


# ── report export ──────────────────────────────────────────────────────────


def test_export_creates_parent_dirs(tmp_path):
    report = EvalReport(top_k=5, turns=[
        TurnResult(
            case_id="x", turn_index=0, question="q", answer="a",
            sources=["S"], dropped_citations=0, total_citations=0,
            recall_at_k=1.0, mrr=1.0, faithfulness=1.0,
            keyword_recall=1.0, answer_relevance=0.8, latency_s=0.1,
        )
    ])
    out = tmp_path / "deep" / "tree" / "report.json"
    export_report(report, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert data["n"] == 1 and data["aggregate"]["recall_at_k"] == 1.0
