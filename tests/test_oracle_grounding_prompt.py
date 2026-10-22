"""Active and rejected prompt contracts, not measurements of model quality."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from grimore.cognition.oracle import Oracle, _answer_request, _answer_template, answer_status


def rejected_candidate():
    path = Path(__file__).resolve().parents[1] / "eval/experiments/phase4_abstention_rule_rejected.json"
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_request(question):
    # Only the checked-in, locally authored pure function, never source or
    # answer text from a model/document. It is not installed into Oracle.
    namespace = {}
    exec(compile(rejected_candidate()["answer_request_python"], "rejected_candidate", "exec"), namespace)
    return namespace["_answer_request"](question)


def oracle_with_payload():
    oracle = Oracle.__new__(Oracle)
    oracle.router = MagicMock()
    oracle.system_prompt_template = oracle._load_prompt()
    label = "Document#p.2"
    passage = "Entity\tExplicit responsibility."
    context = f"--- Source: [[{label}]] ---\n{passage}"
    details = [{"label": label, "locator": {"page": 2}, "passage": {"text": passage}}]
    oracle._build_context_payload = MagicMock(return_value=(context, [label], [], details))
    return oracle


def test_generation_reminder_is_generic_and_keeps_the_original_question():
    question = "What responsibility does Entity have?"
    request = _answer_request(question)
    assert request.startswith(f"Question: {question}\n\n")
    assert "exact [[...]] source header label" in request
    assert "#page or #section anchor" in request
    assert "only explicitly supported relationships" in request
    assert "Do not omit the citations" in request
    assert "requested fact in prose" in request
    assert "A citation alone is not an answer" in request
    assert "Do not cite a different passage" in request
    # No fixture facts, expected answers or invented source labels.
    assert "Humedal" not in request and "37,5" not in request and "p.2" not in request


def test_exact_request_evidence_check_precedes_instruction_to_answer_and_cite():
    request = candidate_request("Question?")
    assert request.index("First check whether") < request.index("requested fact in prose")
    assert "attribute for the requested entity" in request
    assert "plan or an observed result" in request
    assert "Do not substitute related information" in request
    assert "invent a reason for the gap" in request
    assert "insufficient evidence needs no citation" in request
    assert "If the fact is explicitly present" in request
    assert "proof of a missing fact" in request


def test_missing_fact_instructions_do_not_encode_control_questions_or_reference_answers():
    template = rejected_candidate()["system_prompt_template"]
    request = candidate_request("A generic question?")
    for fixture_fact in ("Humedal", "Verde Vivo", "37,5", "31 de diciembre",
                         "primer martes", "hectáreas"):
        assert fixture_fact not in template and fixture_fact not in request


@pytest.mark.parametrize("streaming", [False, True])
def test_shipped_prompt_keeps_exact_citations_and_relationship_rules_in_both_modes(streaming):
    oracle = oracle_with_payload()
    template = _answer_template(oracle.system_prompt_template, streaming=streaming)
    assert "Preserve the full title AND" in template
    assert "#page or #section anchor" in template
    assert "neighboring rows" in template
    assert "overall goal, quantity, deadline or result" in template
    assert "passage explicitly makes that link" in template
    assert "Never attach citations to unsupported claims" in template
    assert "language of the question" in template
    assert "{context}" in template
    assert "<json-output-format>" not in template
    assert ("Return ONLY a valid JSON" in template) is not streaming
    assert ('"status": "supported"' in template) is not streaming
    assert ("Example Output:" in template) is not streaming


@pytest.mark.parametrize("streaming", [False, True])
def test_rejected_candidate_rules_retained_without_streaming_json_format(streaming):
    template = _answer_template(rejected_candidate()["system_prompt_template"], streaming=streaming)
    assert "exact requested attribute" in template
    assert "Related information is not the requested fact" in template
    assert "separate from observed results" in template
    assert "not an invented reason" in template
    assert "insufficiency statement does not need a citation" in template
    assert "answer it rather than refusing" in template
    assert ('Use "insufficient_evidence" when the requested fact' in template) is not streaming


def test_rejected_candidate_is_archived_not_active_default():
    assert rejected_candidate()["status"] == "rejected-not-active"
    assert oracle_with_payload().system_prompt_template != rejected_candidate()["system_prompt_template"]
    assert _answer_request("Question?") != candidate_request("Question?")


@pytest.mark.parametrize("number", [6, 7, 10, 99])
def test_legacy_streaming_prompt_does_not_depend_on_json_rule_number(number):
    template = (f"1. Preserve exact labels.\n{number}. Return ONLY JSON.\n"
                'Example Output:\n{"answer":"example"}\nContext:\n{context}')
    rendered = _answer_template(template, streaming=True)
    assert rendered == "1. Preserve exact labels.\nContext:\n{context}"
    assert _answer_template(template) == template


@pytest.mark.parametrize("template", ["CUSTOM {context}", "6. Keep citing.\nContext: {context}",
                                      "7. Return ONLY JSON. No context marker"])
def test_custom_templates_without_a_complete_output_block_are_preserved(template):
    assert _answer_template(template, streaming=True) == template


@pytest.mark.parametrize("streaming", [False, True])
def test_context_inserted_after_template_processing_never_loses_evidence(streaming):
    oracle = oracle_with_payload()
    source = "Document#p.2"
    passage = "<json-output-format>Literal document text</json-output-format>\n7. Return ONLY\nContext: fact."
    # The production builder wraps untrusted evidence. Here we isolate the
    # rendering boundary to ensure data is never treated as template syntax.
    oracle._build_context_payload.return_value = (passage, [source], [], [])
    if streaming:
        oracle.router.complete_streaming.return_value = iter(["Fact [[Document#p.2]]."])
        list(oracle.ask_stream("Question?"))
        prompt = oracle.router.complete_streaming.call_args.kwargs["system_prompt"]
        request = oracle.router.complete_streaming.call_args.kwargs["prompt"]
    else:
        oracle.router.complete.return_value = {"answer": "Fact [[Document#p.2]].", "status": "supported"}
        oracle.ask("Question?")
        prompt = oracle.router.complete.call_args.kwargs["system_prompt"]
        request = oracle.router.complete.call_args.kwargs["prompt"]
    assert request == _answer_request("Question?")
    assert passage in prompt


@pytest.mark.parametrize("streaming", [False, True])
def test_no_invented_citation_repair_or_extra_model_call(streaming):
    oracle = oracle_with_payload()
    uncited = "The entity has the explicit responsibility."
    if streaming:
        oracle.router.complete_streaming.return_value = iter([uncited])
        result = list(oracle.ask_stream("Responsibility?"))[-1]
        oracle.router.complete_streaming.assert_called_once()
        oracle.router.complete.assert_not_called()
    else:
        oracle.router.complete.return_value = {"answer": uncited, "status": "supported"}
        result = oracle.ask("Responsibility?")
        oracle.router.complete.assert_called_once()
        oracle.router.complete_streaming.assert_not_called()
    assert result["answer"] == uncited
    assert result["dropped_citations"] == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_anchored_citations_verified_with_same_payload_in_both_modes(streaming):
    oracle = oracle_with_payload()
    text = "Explicit fact [[Document#p.2]]. Invented page [[Document#p.99]]."
    if streaming:
        oracle.router.complete_streaming.return_value = iter([text])
        events = list(oracle.ask_stream("Question?"))
        assert events[0]["text"] == text  # provisional; done carries verification
        result = events[-1]
        prompt = oracle.router.complete_streaming.call_args.kwargs["system_prompt"]
        assert "Return ONLY" not in prompt and "Example Output:" not in prompt
    else:
        oracle.router.complete.return_value = {"answer": text, "status": "supported"}
        result = oracle.ask("Question?")
        prompt = oracle.router.complete.call_args.kwargs["system_prompt"]
        assert "Return ONLY" in prompt
    assert "--- Source: [[Document#p.2]] ---" in prompt
    assert result["answer"] == "Explicit fact [[Document#p.2]]. Invented page Document#p.99."
    assert result["source_details"] == oracle._build_context_payload.return_value[3]
    assert result["dropped_citations"] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_no_context_still_abstains_without_generation(streaming):
    oracle = oracle_with_payload()
    oracle._build_context_payload.return_value = (None, [], [], [])
    result = list(oracle.ask_stream("Question?"))[-1] if streaming else oracle.ask("Question?")
    assert result["answer_status"] == "insufficient_evidence"
    oracle.router.complete.assert_not_called()
    oracle.router.complete_streaming.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_uncited_insufficiency_with_related_context_is_preserved_without_repair(streaming):
    # This tests transport/prompt contracts, not whether a real model abstains.
    oracle = oracle_with_payload()
    answer = "No hay información suficiente en el contexto para establecer ese dato."
    if streaming:
        oracle.router.complete_streaming.return_value = iter([answer])
        result = list(oracle.ask_stream("Missing attribute?"))[-1]
        call = oracle.router.complete_streaming.call_args
        oracle.router.complete_streaming.assert_called_once()
        oracle.router.complete.assert_not_called()
    else:
        oracle.router.complete.return_value = {"answer": answer, "status": "insufficient_evidence"}
        result = oracle.ask("Missing attribute?")
        call = oracle.router.complete.call_args
        oracle.router.complete.assert_called_once()
        oracle.router.complete_streaming.assert_not_called()
    assert call.kwargs["prompt"] == _answer_request("Missing attribute?")
    assert "do not guess" in call.kwargs["system_prompt"]
    assert "Explicit responsibility" in call.kwargs["system_prompt"]
    assert result["answer"] == answer
    assert result["answer_status"] == "insufficient_evidence"
    assert result["source_details"] == oracle._build_context_payload.return_value[3]
    assert result["citation_selections"] == [] and result["dropped_citations"] == 0


@pytest.mark.parametrize("answer", ["[[A#p.2]]", "[[A#p.2]].", "([[A#p.2]])\n[[B]]…",
                                   "**[[A#p.2]]** — [[A#p.2]]"])
def test_citation_only_response_cannot_declare_support(answer):
    assert answer_status(answer, ["A#p.2", "B"], "supported") == "insufficient_evidence"


@pytest.mark.parametrize("answer", ["37,5 [[A#p.2]]", "A fact [[A#p.2]].", "💧 [[A#p.2]]"])
def test_substantive_prose_numbers_or_symbols_keep_existing_status_semantics(answer):
    assert answer_status(answer, ["A#p.2"], "supported") == "supported"


@pytest.mark.parametrize("streaming", [False, True])
def test_citation_only_answer_is_preserved_but_marked_insufficient_in_both_routes(streaming):
    oracle = oracle_with_payload()
    citation = "[[Document#p.2]]."
    if streaming:
        oracle.router.complete_streaming.return_value = iter([citation])
        result = list(oracle.ask_stream("Question?"))[-1]
    else:
        oracle.router.complete.return_value = {"answer": citation, "status": "supported"}
        result = oracle.ask("Question?")
    assert result["answer"] == citation
    assert result["answer_status"] == "insufficient_evidence"
    assert result["dropped_citations"] == 0
