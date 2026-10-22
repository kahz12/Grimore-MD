"""No models: explicit selection/rendering, failure isolation and bounded audit."""
import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from grimore.cognition.answer_protocol import RESPONSE_LIMIT, render_claims, response_snapshot
from grimore.cognition.eval import EvalCase, run_eval
from grimore.cognition.oracle import _coverage_context_note
from grimore.cognition.sufficiency import context_blocks
from eval.benchmark_answers import negative_control
from tests.test_answer_sufficiency import GOOD, details, make_oracle


VALID = {"claims": [{"text": "Explicit responsibility.", "refs": ["cite:3"]}], "status": "supported"}


def test_refs_are_explicit_model_choices_not_inferred_from_checker():
    text, status, error = render_claims(VALID, details())
    assert text == "Explicit responsibility. [cite:3]"
    assert status == "supported" and error is None
    response = deepcopy(VALID)
    response["claims"][0]["refs"] = []
    assert render_claims(response, details())[2] == "missing_or_invalid_claim_refs"


@pytest.mark.parametrize("response", [None, {}, [],
    {"answer": "Uncited fact", "status": "supported"},
    {"claims": [], "status": "supported"},
    {"claims": [], "status": "new-status"},
    {"claims": VALID["claims"], "status": "insufficient_evidence"},
    {"claims": [{"text": "Fact", "refs": "cite:3"}], "status": "supported"},
    {"claims": [{"text": "Fact", "refs": ["cite:8"]}], "status": "supported"},
    {"claims": [{"text": "Fact", "refs": ["cite:3", "cite:3"]}], "status": "supported"},
    {"claims": [{"text": "Fact [cite:3]", "refs": ["cite:3"]}], "status": "supported"},
    {"claims": [{"text": "Fact [[Document]]", "refs": ["cite:3"]}], "status": "supported"},
    {"claims": [{"text": "...", "refs": ["cite:3"]}], "status": "supported"},
    {"claims": [{"text": 1, "refs": ["cite:3"]}], "status": "supported"},
    {"claims": [{"text": "Fact", "refs": ["cite:3"], "quote": "Extra"}], "status": "supported"},
    {"claims": VALID["claims"] * 65, "status": "supported"},
    {"claims": [{"text": "x" * 16001, "refs": ["cite:3"]}], "status": "supported"},
])
@pytest.mark.parametrize("streaming", [False, True])
def test_invalid_answers_never_escape_as_supported_or_provisional_facts(response, streaming):
    oracle = make_oracle(GOOD)
    oracle.router.complete.side_effect = [GOOD, response]
    events = list(oracle.ask_stream("¿Qué responsabilidad tiene?")) if streaming else [oracle.ask("¿Qué responsabilidad tiene?")]
    assert len(events) == 1
    result = events[0]
    assert result["answer_status"] == "uncertain"
    assert result["citation_selections"] == []
    assert result["sufficiency"]["generation"]["decision"] == "unassessed"
    assert result["sufficiency"]["generation"]["error"]
    assert result["sufficiency"]["generation"]["model_response"]["value"] == response
    assert "Explicit responsibility." not in result["answer"]
    assert "no se muestra una respuesta factual" in result["answer"]
    assert oracle.router.complete.call_count == 2
    oracle.router.complete_streaming.assert_not_called()


def test_a_bad_second_claim_invalidates_the_entire_response():
    response = deepcopy(VALID)
    response["claims"].append({"text": "Another fact", "refs": []})
    assert render_claims(response, details())[0] == ""


@pytest.mark.parametrize("status", ["supported", "uncertain", "conflicting_evidence"])
def test_valid_multiple_claims_preserve_explicit_choice_order_and_state(status):
    response = {"claims": [{"text": "First fact.", "refs": ["cite:3", "cite:1"]},
                           {"text": "Second fact.", "refs": ["cite:1"]}], "status": status}
    assert render_claims(response, details()) == (
        "First fact. [cite:3] [cite:1]\n\nSecond fact. [cite:1]", status, None)


def test_generation_cannot_cite_a_candidate_not_delivered_to_it():
    oracle = make_oracle(GOOD)
    response = deepcopy(VALID)
    response["claims"][0]["refs"] = ["cite:1"]
    oracle.router.complete.side_effect = [GOOD, response]
    result = oracle.ask("Question?")
    assert result["sufficiency"]["generation"]["error"] == "missing_or_invalid_claim_refs"


def test_same_page_labels_still_resolve_each_explicit_token_to_its_fragment():
    check = {"answerable": True, "evidence": [GOOD["evidence"][0],
             {"ref": "cite:1", "quote": "Unrelated fact."}]}
    oracle = make_oracle(check)
    records = details()
    records[0]["label"] = records[1]["label"]
    context = "\n\n".join(context_blocks(records, _coverage_context_note))
    oracle._build_context_payload.return_value = (context, [records[0]["label"]], [{"title": "Document"}], records)
    answer = {"claims": [{"text": "Comparison.", "refs": ["cite:3", "cite:1"]}], "status": "supported"}
    oracle.router.complete.side_effect = [check, answer]
    result = oracle.ask("Comparison?")
    assert result["answer"] == "Comparison. [[Document#p.2]] [[Document#p.2]]"
    assert [s["source_detail_indices"] for s in result["citation_selections"]] == [[1], [0]]
    assert [s["citation_ref"] for s in result["citation_selections"]] == ["cite:3", "cite:1"]


def test_generation_can_decline_without_exporting_a_factual_explanation():
    oracle = make_oracle(GOOD)
    oracle.router.complete.side_effect = [GOOD, {"claims": [], "status": "insufficient_evidence"}]
    result = oracle.ask("¿Qué responsabilidad tiene?")
    assert result["answer_status"] == "insufficient_evidence"
    assert result["sufficiency"]["generation"]["decision"] == "insufficient"


def test_generation_exception_is_not_negative_credit_or_a_private_error_leak():
    oracle = make_oracle(GOOD)
    oracle.router.complete.side_effect = [GOOD, RuntimeError("private endpoint information")]
    session = MagicMock(oracle=oracle, turns=[])
    report = run_eval(session, [EvalCase(id="failed-generation", question="¿Qué falta?", negative=True)], judge=False)
    assert negative_control(report.turns)["unassessed_n"] == 1
    assert report.summary()["aggregate"]["evidence_control_failure_n"] == 1
    assert "private" not in report.turns[0].answer
    assert "sufficiency_s" in report.summary()["aggregate"]["stage_latency"]


@pytest.mark.parametrize("failed_stage", ["checker", "generation"])
def test_technical_failures_skip_judges_and_cannot_receive_claim_support_credit(failed_stage):
    oracle = make_oracle(GOOD)
    oracle.router.complete.side_effect = [None] if failed_stage == "checker" else [GOOD, None]
    session = MagicMock(oracle=oracle, router=oracle.router, turns=[])
    report = run_eval(session, [EvalCase(id="failure", question="Question?", expected_sources=["Document"])], judge=True)
    assert oracle.router.complete.call_count == (1 if failed_stage == "checker" else 2)
    assert report.turns[0].automated_claim_support is None
    assert report.turns[0].answer_relevance is None


def test_audit_snapshot_is_bounded_explicitly_incomplete_and_independent():
    response = {"evidence": "x" * RESPONSE_LIMIT}
    snapshot = response_snapshot(response)
    assert snapshot["truncated"] and snapshot["value"] is None
    assert len(snapshot["json_prefix"]) == RESPONSE_LIMIT
    assert snapshot["sha256"] and snapshot["chars"] > RESPONSE_LIMIT
    small = response_snapshot(VALID)
    VALID_COPY = deepcopy(VALID)
    VALID_COPY["claims"][0]["text"] = "Changed"
    assert small["value"] == VALID and small["value"] != VALID_COPY
    json.dumps(snapshot, allow_nan=False)


def test_unexpected_non_json_audit_is_unavailable_not_repr_of_private_object():
    assert response_snapshot(object()) == {"value": None, "truncated": False, "unavailable": True}
