"""Fixed unanswerable tuning cases; cues never substitute for human review."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.benchmark_answers import negative_control
from grimore.cognition.eval import EvalCase, load_golden, run_eval


MANIFEST = Path(__file__).resolve().parents[1] / "eval/phase4_unanswerable_answers.yaml"
DETAILS = [{"label": "Humedal#p.1", "citation_ref": "cite:1",
            "passage": {"text": "La calidad del agua se mide el primer martes de cada mes."},
            "locator": {"page": 1}}]


def evaluate(answer="No se menciona la hora exacta.", **overrides):
    session = MagicMock()
    session.turns = []
    session.oracle.ask.return_value = {
        "answer": answer, "answer_status": "insufficient_evidence",
        "sources": ["Humedal#p.1"], "source_details": DETAILS,
        "citation_selections": [], "retrieved": [{"title": "Humedal", "rank": 1}],
        **overrides,
    }
    case = EvalCase(id="neg", question="Hora?", negative=True,
                    review_criteria="No inventar una hora.")
    return run_eval(session, [case], judge=False)


def test_fixed_negative_manifest_is_separate_pdf_tuning_not_held_out():
    cases = load_golden(MANIFEST)
    assert [c.id for c in cases] == ["phase4-missing-sampling-hour",
                                   "phase4-missing-cooperative-deadline",
                                   "phase4-missing-actual-restored-area"]
    for case in cases:
        assert case.negative and case.split == "tuning"
        assert case.case_type == "unanswerable" and case.document_formats == ["pdf"]
        assert case.expected_answer == "abstain" and case.review_criteria
        assert not case.expected_sources and not case.expected_evidence and not case.expected_keywords


def test_negative_queue_preserves_context_and_does_not_inflate_positive_evidence():
    report = evaluate()
    summary = report.summary()
    assert summary["manual_review"] == []
    review = summary["negative_manual_review"][0]
    assert review["abstention_verdict"] == "pending"
    assert review["abstention_detected"] is True
    assert review["source_details"] == DETAILS
    assert review["citation_selections"] == []
    assert review["review_criteria"] == "No inventar una hora."
    assert "expected_evidence" not in review and "claim_support" not in review
    assert summary["aggregate"]["expected_evidence_n"] == 0
    assert summary["aggregate"]["context_evidence_recall"] is None
    assert summary["aggregate"]["automated_claim_support"] is None


def test_cue_diagnostic_keeps_semantic_verdict_pending_even_with_refusal():
    diagnostic = negative_control(evaluate().turns)
    assert diagnostic["cue_assessed_n"] == 1 and diagnostic["unassessed_n"] == 0
    assert diagnostic["abstention_cue_rate"] == 1.0
    assert diagnostic["semantic_abstention_rate"] is None
    assert diagnostic["observations"][0]["delivered_context_nonempty"]
    assert diagnostic["human_review"] == "pending"


@pytest.mark.parametrize("answer,overrides", [
    ("", {}), ("  ", {}), ("The Oracle is silent.", {}),
    ("Your vault seems empty of relevant whispers on this subject.", {}),
    ("No se menciona la hora.", {"source_details": []}),
    ("No se menciona la hora.", {"source_details": [{"passage": {"text": ""}}]}),
    ("No se menciona la hora.", {"answer_status": "unassessed"}),
])
def test_empty_silent_no_context_and_unassessed_are_not_successful_abstentions(answer, overrides):
    report = evaluate(answer, **overrides)
    diagnostic = negative_control(report.turns)
    assert diagnostic["cue_assessed_n"] == 0 and diagnostic["unassessed_n"] == 1
    assert diagnostic["abstention_cue_rate"] is None
    assert diagnostic["observations"][0]["abstention_detected"] is None
    assert report.summary()["negative_manual_review"][0]["abstention_verdict"] == "pending"


def test_transport_exception_remains_unassessed_in_control_not_vacuous_success():
    session = MagicMock()
    session.turns = []
    session.oracle.ask.side_effect = RuntimeError("connection failed")
    report = run_eval(session, [EvalCase(id="error", question="Hora?", negative=True)], judge=False)
    assert negative_control(report.turns)["unassessed_n"] == 1
    assert negative_control(report.turns)["abstention_cue_rate"] is None


def test_supported_status_does_not_override_cue_and_cue_does_not_prove_no_hallucination():
    # The historic detector's known mixed-answer limitation is made visible,
    # not 'fixed' by adding a phrase after observing actual model responses.
    report = evaluate("No se menciona la hora, pero el muestreo será a las 09:00.",
                      answer_status="supported")
    assert negative_control(report.turns)["abstention_cue_rate"] == 1.0
    assert report.summary()["negative_manual_review"][0]["abstention_verdict"] == "pending"
    report = evaluate("El muestreo será a las 09:00.", answer_status="insufficient_evidence")
    assert negative_control(report.turns)["abstention_cue_rate"] == 0.0


def test_non_generated_or_positive_turns_are_not_negative_cue_successes():
    turn = evaluate().turns[0]
    assert negative_control([replace(turn, generated=False)])["cue_assessed_n"] == 0
    assert negative_control([replace(turn, negative=False)])["negative_turns_n"] == 0


def test_retrieval_only_negative_not_queued_for_human_generated_answer_review():
    session = MagicMock()
    session.turns = []
    session.oracle.retrieve.return_value = []
    case = EvalCase(id="neg", question="Hora?", negative=True, review_criteria="No inventar.")
    summary = run_eval(session, [case], generate=False, judge=False).summary()
    assert summary["negative_manual_review"] == [] and summary["manual_review"] == []
