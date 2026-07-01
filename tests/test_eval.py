"""Unit tests for the eval harness (mocked router/embedder; no live model).

Live coverage of the harness end-to-end against real Ollama lives in
``tests/test_e2e_oracle.py`` so this file can run in milliseconds and gate CI.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from grimore.cognition.eval import (
    EvalCase,
    EvalReport,
    TurnResult,
    answer_abstained,
    append_history,
    classify_outcome,
    compare_runs,
    comparison_summary,
    export_report,
    run_provenance,
    faithfulness,
    judge_relevance,
    keyword_recall,
    _keyword_hits,
    load_golden,
    mrr,
    ranked_sources,
    recall_at_k,
    run_baseline,
    run_eval,
    source_hit_at_k,
    _stage_latency_means,
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

    def test_keyword_recall_is_accent_insensitive_both_ways(self):
        # LLM accenting is non-deterministic; match must survive either drift.
        assert keyword_recall("La energía del Sol.", ["energia"]) == 1.0
        assert keyword_recall("The Socrates method.", ["Sócrates"]) == 1.0

    def test_keyword_recall_no_midword_false_positive(self):
        # The substring test used to count 'art' inside 'Sparta'/'start'; the
        # leading word boundary now rejects both.
        assert keyword_recall("Sparta had a strong start.", ["art"]) == 0.0
        # …but a real occurrence at a word boundary still counts.
        assert keyword_recall("Gothic art endured.", ["art"]) == 1.0

    def test_keyword_recall_tolerates_simple_inflection(self):
        # No trailing boundary, so plurals/derivations still register.
        assert keyword_recall("Frogs breathe through skin.", ["frog"]) == 1.0
        assert keyword_recall("An orbital resonance.", ["orbit"]) == 1.0

    def test_keyword_recall_multiword_separator_flexible(self):
        for text in ("machine learning", "machine-learning", "machine  learning"):
            assert keyword_recall(f"About {text} today.", ["machine learning"]) == 1.0

    def test_keyword_hits_present_and_missing_agree_with_recall(self):
        present, missing = _keyword_hits(
            "Frogs and pointed arches.", ["frog", "pointed", "buttress"]
        )
        assert present == ["frog", "pointed"]
        assert missing == ["buttress"]

    def test_hit_at_k_respects_rank(self):
        assert source_hit_at_k(["A", "B", "C"], ["A"], 1) is True
        assert source_hit_at_k(["X", "A", "B"], ["A"], 1) is False
        assert source_hit_at_k(["X", "A", "B"], ["A"], 3) is True

    def test_hit_at_k_strips_anchors(self):
        assert source_hit_at_k(["Gothic#p.3"], ["Gothic"], 1) is True

    def test_hit_at_k_empty_expected_is_vacuously_true(self):
        assert source_hit_at_k([], [], 1) is True

    def test_ranked_sources_prefers_retrieved_over_flat_sources(self):
        # ``sources`` is set-order junk; ``retrieved`` carries the real rank.
        result = {
            "retrieved": [{"title": "First"}, {"title": "Second"}],
            "sources": ["Second", "First"],
        }
        assert ranked_sources(result) == ["First", "Second"]

    def test_ranked_sources_falls_back_when_retrieved_absent(self):
        assert ranked_sources({"sources": ["Only"]}) == ["Only"]


# ── token-normalised title matching ──────────────────────────────────────────


class TestTitleMatching:
    """Short golden titles must match a note's decorated H1, accent- and
    emoji-insensitively, without colliding across distinct notes."""

    def test_short_title_matches_decorated_h1(self):
        retrieved = ["🏛️ The Roman Empire: An Exhaustive Historical and Structural Analysis"]
        assert source_hit_at_k(retrieved, ["Roman Empire"], 1) is True
        assert mrr(retrieved, ["Roman Empire"]) == 1.0
        assert recall_at_k(retrieved, ["Roman Empire"]) == 1.0

    def test_accent_and_emoji_insensitive(self):
        assert source_hit_at_k(["🌌 Astronomía General"], ["astronomia"], 1) is True

    def test_stem_with_underscore_tokenises(self):
        assert source_hit_at_k(["gothic_architecture"], ["Gothic Architecture"], 1) is True

    def test_distinct_titles_do_not_collide_on_shared_token(self):
        # Both H1s contain "Guide"; a 2-token expected must not match the wrong note.
        kotlin = ["☕ Kotlin Fundamentals: A Comprehensive Reference Guide"]
        assert source_hit_at_k(kotlin, ["Social Engineering"], 1) is False

    def test_subset_rule_requires_all_expected_tokens(self):
        assert source_hit_at_k(["Roman Empire"], ["Roman Republic"], 1) is False


# ── abstention (negative cases) ──────────────────────────────────────────────


class TestAbstention:
    def test_empty_or_blank_answer_abstains(self):
        assert answer_abstained("") is True
        assert answer_abstained("   ") is True

    def test_refusal_phrasings_abstain(self):
        assert answer_abstained("Your vault seems empty of relevant whispers.") is True
        assert answer_abstained("I don't know — there's no mention of that here.") is True

    def test_substantive_answer_does_not_abstain(self):
        assert answer_abstained("Rome fell in 476 CE after repeated invasions.") is False


# ── failure taxonomy ─────────────────────────────────────────────────────────


class TestClassifyOutcome:
    def _c(self, **over):
        base = {
            "negative": False, "generate": True, "ranked": ["A"],
            "expected_sources": ["A"], "top_k": 3,
            "keyword_recall": 1.0, "faithfulness": 1.0, "abstained": None,
        }
        base.update(over)
        return classify_outcome(**base)

    def test_ok_when_everything_passes(self):
        assert self._c() == "ok"

    def test_retrieval_miss_when_not_in_pool(self):
        assert self._c(ranked=["X", "Y"]) == "retrieval_miss"

    def test_ranking_miss_when_below_context_cut(self):
        assert self._c(ranked=["X", "Y", "Z", "A"], top_k=3) == "ranking_miss"

    def test_generation_miss_when_keywords_low(self):
        assert self._c(keyword_recall=0.0) == "generation_miss"

    def test_citation_miss_when_unfaithful(self):
        assert self._c(faithfulness=0.5) == "citation_miss"

    def test_retrieval_only_passes_at_context_without_generation(self):
        assert self._c(generate=False, keyword_recall=None, faithfulness=None) == "ok"

    def test_retrieval_only_still_detects_ranking_miss(self):
        assert self._c(generate=False, ranked=["X", "Y", "Z", "A"], top_k=3,
                       keyword_recall=None, faithfulness=None) == "ranking_miss"

    def test_negative_ok_when_abstained_else_hallucinated(self):
        kw = {
            "negative": True, "generate": True, "ranked": ["X"],
            "expected_sources": [], "top_k": 3,
            "keyword_recall": None, "faithfulness": None,
        }
        assert classify_outcome(abstained=True, **kw) == "ok"
        assert classify_outcome(abstained=False, **kw) == "hallucinated"

    def test_negative_na_in_retrieval_only(self):
        assert classify_outcome(
            negative=True, generate=False, ranked=["X"], expected_sources=[],
            top_k=3, keyword_recall=None, faithfulness=None, abstained=None,
        ) == "na"


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

    def test_parses_category_and_negative(self, tmp_path):
        cases = load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: q1
    question: "What?"
    category: single-hop
    expected_sources: ["A"]
  - id: nq
    question: "Not in vault?"
    category: negative
    negative: true
"""))
        assert cases[0].category == "single-hop" and cases[0].negative is False
        assert cases[1].negative is True and cases[1].category == "negative"

    def test_followup_inherits_parent_category(self, tmp_path):
        cases = load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: q1
    question: "What?"
    category: multi-hop
    follow_ups:
      - id: q1a
        question: "More?"
"""))
        assert cases[0].follow_ups[0].category == "multi-hop"

    def test_rejects_negative_with_expected_sources(self, tmp_path):
        with pytest.raises(ValueError, match="negative case"):
            load_golden(self._write(tmp_path, """\
version: 1
questions:
  - id: bad
    question: "x"
    negative: true
    expected_sources: ["A"]
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
        # Mirror the real Session: the Oracle reads hybrid_search off this
        # same object, so --baseline can toggle it.
        self.config = SimpleNamespace(cognition=SimpleNamespace(hybrid_search=True))

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

    def test_ranking_metrics_read_rank_order_not_flattened_sources(self):
        # The whole point of #1: ``sources`` lists the right note but in junk
        # order (Right first by luck); ``retrieved`` says it was actually
        # ranked 2nd. Hit@1 must be False, Hit@3 True, MRR 0.5 — proving the
        # harness scores ranking off ``retrieved``, not ``sources``.
        session = _FakeSession([
            {"answer": "ans",
             "sources": ["Right", "Wrong"],
             "retrieved": [{"title": "Wrong", "rank": 1},
                           {"title": "Right", "rank": 2}],
             "dropped_citations": 0},
        ])
        cases = [EvalCase(id="r", question="q", expected_sources=["Right"])]
        report = run_eval(session, cases, top_k=3, judge=False)
        t = report.turns[0]
        assert t.source_hit_at_1 is False
        assert t.source_hit_at_3 is True
        assert t.mrr == 0.5
        assert t.retrieved == ["Wrong", "Right"]
        agg = report.summary()["aggregate"]
        assert agg["hit_at_1"] == 0.0 and agg["hit_at_3"] == 1.0

    def test_full_mode_threads_retrieval_k_into_ask(self):
        session = _FakeSession([
            {"answer": "a", "sources": ["A"],
             "retrieved": [{"title": "A", "rank": 1}], "dropped_citations": 0},
        ])
        cases = [EvalCase(id="x", question="q", expected_sources=["A"])]
        run_eval(session, cases, top_k=3, retrieval_k=7, judge=False)
        assert session.oracle.ask.call_args.kwargs["retrieval_k"] == 7

    def test_retrieval_only_mode_ranks_without_generation(self):
        # generate=False must use oracle.retrieve (no ask), score ranking from
        # it, and report generation metrics as None — not a misleading 0.0.
        session = _FakeSession([])  # ask must never fire
        session.oracle.retrieve.return_value = [
            {"title": "Right", "rank": 1},
            {"title": "Other", "rank": 2},
        ]
        cases = [EvalCase(
            id="r", question="q",
            expected_sources=["Right"], expected_keywords=["ignored"],
        )]
        report = run_eval(
            session, cases, top_k=3, retrieval_k=5, generate=False, judge=False,
        )
        t = report.turns[0]
        assert t.source_hit_at_1 is True and t.mrr == 1.0
        assert t.keyword_recall is None
        assert t.faithfulness is None
        assert t.answer_relevance is None
        session.oracle.ask.assert_not_called()
        assert session.oracle.retrieve.call_args.kwargs["top_k"] == 5
        # Aggregate carries the absence through, not a zero.
        agg = report.summary()["aggregate"]
        assert agg["keyword_recall"] is None and agg["faithfulness"] is None
        assert agg["hit_at_1"] == 1.0

    def test_negative_scored_by_abstention_and_excluded_from_ranking(self):
        session = _FakeSession([
            {"answer": "Pointed arch and flying buttress.",
             "sources": ["gothic_architecture"],
             "retrieved": [{"title": "gothic_architecture", "rank": 1}],
             "dropped_citations": 0},
            {"answer": "Your vault seems empty of relevant whispers on this subject.",
             "sources": [],
             "retrieved": [{"title": "gothic_architecture", "rank": 1}],
             "dropped_citations": 0},
        ])
        cases = [
            EvalCase(id="pos", question="q", category="single-hop",
                     expected_sources=["gothic_architecture"],
                     expected_keywords=["pointed arch"]),
            EvalCase(id="neg", question="not in vault?", category="negative", negative=True),
        ]
        report = run_eval(session, cases, top_k=3, judge=False)
        s = report.summary()
        assert s["positives_n"] == 1 and s["negatives_n"] == 1
        # Ranking aggregate is over the positive turn only — the negative's
        # empty expectation does not inflate it.
        assert s["aggregate"]["hit_at_1"] == 1.0
        assert s["aggregate"]["abstention_rate"] == 1.0
        neg_turn = report.turns[1]
        assert neg_turn.negative is True and neg_turn.abstained is True
        # by_category covers positive turns only.
        assert set(s["by_category"]) == {"single-hop"}
        assert s["by_category"]["single-hop"]["n"] == 1

    def test_negative_hallucination_drops_abstention_rate(self):
        session = _FakeSession([
            {"answer": "The starter needs flour, water, and five days of fermentation.",
             "sources": ["x"], "retrieved": [{"title": "x", "rank": 1}],
             "dropped_citations": 0},
        ])
        cases = [EvalCase(id="neg", question="sourdough?", category="negative", negative=True)]
        report = run_eval(session, cases, top_k=3, judge=False)
        assert report.summary()["aggregate"]["abstention_rate"] == 0.0
        assert report.turns[0].abstained is False

    def test_keywordless_turn_is_none_not_vacuous_one(self):
        session = _FakeSession([
            {"answer": "some answer", "sources": ["A"],
             "retrieved": [{"title": "A", "rank": 1}], "dropped_citations": 0},
        ])
        cases = [EvalCase(id="x", question="q", expected_sources=["A"], expected_keywords=[])]
        report = run_eval(session, cases, top_k=3, judge=False)
        assert report.turns[0].keyword_recall is None
        assert report.summary()["aggregate"]["keyword_recall"] is None

    def test_baseline_runs_both_arms_and_restores_config(self):
        session = _FakeSession([])
        seen: list[bool] = []

        def ask(question, **kwargs):
            # The arm is selected by the live config flag the override flips.
            hybrid = session.config.cognition.hybrid_search
            seen.append(hybrid)
            if hybrid:
                return {"answer": "a", "sources": ["A"],
                        "retrieved": [{"title": "A", "rank": 1}], "dropped_citations": 0}
            # Dense-only ranks the right note lower → a worse hit.
            return {"answer": "a", "sources": ["A"],
                    "retrieved": [{"title": "B", "rank": 1}, {"title": "A", "rank": 2}],
                    "dropped_citations": 0}

        session.oracle.ask.side_effect = ask
        cases = [EvalCase(id="x", question="q", expected_sources=["A"])]

        hybrid, base = run_baseline(session, cases, top_k=3, judge=False)

        # Hybrid arm first, then dense-only; original config restored.
        assert seen == [True, False]
        assert session.config.cognition.hybrid_search is True
        # Fusion wins on this case.
        assert hybrid.summary()["aggregate"]["hit_at_1"] == 1.0
        assert base.summary()["aggregate"]["hit_at_1"] == 0.0
        cmp = comparison_summary(hybrid, base)
        assert cmp["metrics"]["hit_at_1"]["delta"] == 1.0
        assert cmp["metrics"]["mrr"]["delta"] == pytest.approx(0.5)  # 1.0 vs 0.5
        assert "RRF" in cmp["arms"]["hybrid"]

    def test_outcomes_failures_and_pass_rate(self):
        session = _FakeSession([
            {"answer": "Pointed arch here.", "sources": ["A"],
             "retrieved": [{"title": "A", "rank": 1}], "dropped_citations": 0},
            {"answer": "unrelated text", "sources": ["B"],
             "retrieved": [{"title": "B", "rank": 1}], "dropped_citations": 0},
            {"answer": "whatever", "sources": ["Z"],
             "retrieved": [{"title": "Z", "rank": 1}], "dropped_citations": 0},
        ])
        cases = [
            EvalCase(id="ok", question="q", expected_sources=["A"],
                     expected_keywords=["pointed arch"]),
            EvalCase(id="gen", question="q", expected_sources=["B"],
                     expected_keywords=["needle"]),
            EvalCase(id="ret", question="q", expected_sources=["C"],
                     expected_keywords=["x"]),
        ]
        report = run_eval(session, cases, top_k=3, judge=False)
        assert [t.outcome for t in report.turns] == [
            "ok", "generation_miss", "retrieval_miss",
        ]
        s = report.summary()
        assert s["aggregate"]["pass_rate"] == pytest.approx(1 / 3)
        assert s["aggregate"]["outcomes"]["ok"] == 1
        fails = {f["case_id"]: f for f in s["failures"]}
        assert set(fails) == {"gen", "ret"}
        assert fails["gen"]["missing_keywords"] == ["needle"]
        assert fails["ret"]["outcome"] == "retrieval_miss"
        assert fails["ret"]["expected_sources"] == ["C"]

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


# ── per-stage latency ────────────────────────────────────────────────────────


def _turn_with_timings(timings: dict) -> TurnResult:
    return TurnResult(
        case_id="x", turn_index=0, question="q", answer="a",
        sources=["S"], dropped_citations=0, total_citations=0,
        recall_at_k=1.0, mrr=1.0, faithfulness=1.0,
        keyword_recall=1.0, answer_relevance=None, latency_s=0.1,
        timings=timings,
    )


class TestStageLatency:
    def test_means_average_stages_independently(self):
        turns = [
            _turn_with_timings({"embed_s": 0.1, "retrieve_s": 0.2, "generate_s": 2.0}),
            _turn_with_timings({"embed_s": 0.3, "retrieve_s": 0.4}),  # no generate
        ]
        means = _stage_latency_means(turns)
        assert means["embed_s"] == pytest.approx(0.2)
        assert means["retrieve_s"] == pytest.approx(0.3)
        # generate seen once → averaged only over the turn that reported it.
        assert means["generate_s"] == pytest.approx(2.0)
        # a stage no turn reported never appears.
        assert "rerank_s" not in means

    def test_keys_follow_pipeline_order(self):
        means = _stage_latency_means([
            _turn_with_timings({"generate_s": 1.0, "embed_s": 0.1, "retrieve_s": 0.2}),
        ])
        assert list(means) == ["embed_s", "retrieve_s", "generate_s"]

    def test_run_captures_oracle_timings_into_aggregate(self):
        session = _FakeSession([
            {"answer": "a", "sources": ["A"], "dropped_citations": 0,
             "timings": {"embed_s": 0.05, "retrieve_s": 0.1, "generate_s": 1.5}},
        ])
        cases = [EvalCase(id="x", question="q", expected_sources=["A"])]
        report = run_eval(session, cases, top_k=1, judge=False)
        assert report.turns[0].timings["generate_s"] == pytest.approx(1.5)
        stages = report.summary()["aggregate"]["stage_latency"]
        assert stages["embed_s"] == pytest.approx(0.05)
        assert stages["generate_s"] == pytest.approx(1.5)


# ── history ledger + run-over-run comparison ─────────────────────────────────


class TestHistoryAndCompare:
    def test_append_history_appends_one_jsonl_row_per_run(self, tmp_path):
        ledger = tmp_path / "deep" / "hist.jsonl"
        summary = {
            "n": 3, "positives_n": 2, "negatives_n": 1,
            "aggregate": {"hit_at_1": 0.5, "mrr": 0.75, "pass_rate": 0.66,
                          "outcomes": {"ok": 2, "ranking_miss": 1}},
        }
        prov = {"git_sha": "abc123", "config_hash": "deadbeef"}
        meta = {"mode": "full", "top_k": 5, "retrieval_k": 10}
        append_history(ledger, summary, prov, run_meta=meta)
        append_history(ledger, summary, prov, run_meta=meta)
        import json
        lines = ledger.read_text().strip().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["git_sha"] == "abc123" and row["config_hash"] == "deadbeef"
        assert row["mode"] == "full" and row["metrics"]["hit_at_1"] == 0.5
        assert row["outcomes"]["ok"] == 2

    def test_compare_runs_flags_regression_and_improvement(self):
        prev = {
            "aggregate": {"hit_at_1": 1.0, "mrr": 1.0, "pass_rate": 1.0},
            "turns": [
                {"case_id": "a", "turn_index": 0, "outcome": "ok", "mrr": 1.0, "source_hit_at_1": True},
                {"case_id": "b", "turn_index": 0, "outcome": "retrieval_miss", "mrr": 0.0, "source_hit_at_1": False},
            ],
        }
        cur = {
            "aggregate": {"hit_at_1": 0.5, "mrr": 0.75, "pass_rate": 0.5},
            "turns": [
                {"case_id": "a", "turn_index": 0, "outcome": "ranking_miss", "mrr": 0.5, "source_hit_at_1": False},
                {"case_id": "b", "turn_index": 0, "outcome": "ok", "mrr": 1.0, "source_hit_at_1": True},
            ],
        }
        cmp = compare_runs(cur, prev)
        assert cmp["n_regressions"] == 1 and cmp["n_improvements"] == 1
        assert cmp["regressions"][0]["case_id"] == "a"
        assert cmp["improvements"][0]["case_id"] == "b"
        assert cmp["aggregate_delta"]["hit_at_1"]["delta"] == pytest.approx(-0.5)

    def test_compare_runs_marks_new_and_dropped_not_regressions(self):
        prev = {"aggregate": {}, "turns": [
            {"case_id": "old", "turn_index": 0, "outcome": "ok", "mrr": 1.0, "source_hit_at_1": True}]}
        cur = {"aggregate": {}, "turns": [
            {"case_id": "new", "turn_index": 0, "outcome": "ok", "mrr": 1.0, "source_hit_at_1": True}]}
        cmp = compare_runs(cur, prev)
        statuses = {r["case_id"]: r["status"] for r in cmp["turns"]}
        assert statuses["new"] == "new"
        assert cmp["dropped"] == ["old#0"]
        assert cmp["n_regressions"] == 0

    def test_run_provenance_is_stable_and_well_formed(self, tmp_path):
        golden = tmp_path / "g.yaml"
        golden.write_text("version: 1\nquestions: []\n", encoding="utf-8")
        session = SimpleNamespace(config=SimpleNamespace(cognition=SimpleNamespace(
            model_embeddings_local="emb", hybrid_search=True, rrf_k=60)))
        prov = run_provenance(session, golden)
        assert prov["config"]["hybrid_search"] is True
        assert prov["golden_hash"] is not None and "timestamp" in prov
        # Same fingerprint → same config_hash (so genuine regressions are
        # distinguishable from config changes).
        assert run_provenance(session, golden)["config_hash"] == prov["config_hash"]


# ── shipped golden ───────────────────────────────────────────────────────────


def test_shipped_golden_loads_and_is_stratified():
    """The checked-in golden parses and spans the expected strata. Guards
    against a malformed edit landing in the repo (titles/keywords are
    validated separately against the live vault)."""
    golden = Path(__file__).resolve().parents[1] / "eval" / "grimore_golden.yaml"
    cases = load_golden(golden)
    assert len(cases) >= 10
    categories = {c.category for c in cases}
    assert {"single-hop", "multi-hop", "negative"} <= categories
    assert any(c.negative for c in cases)
    assert any(c.follow_ups for c in cases)
    # Every non-negative case names at least one expected source.
    assert all(c.expected_sources for c in cases if not c.negative)
