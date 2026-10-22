"""Exact selected-passage coverage must not borrow evidence from its neighbor."""
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from grimore.cognition.eval import EvalCase, expected_evidence_checks, run_eval
from grimore.cognition.oracle import Oracle


REFERENCE = {"source": "Humedal", "passage": "El área es de 37,5 hectáreas", "locator": "p.3"}


def details():
    return [{"label": "Humedal#p.3", "citation_ref": f"cite:{i+1}",
             "document": {"title": "Humedal"}, "locator": {"page": 3},
             "passage": {"text": text}}
            for i, text in enumerate(["Un distractor en la misma página.", REFERENCE["passage"]])]


def resolve(raw, sources=None):
    sources = details() if sources is None else sources
    answer, _, trace = Oracle.resolve_citations(raw, [s["label"] for s in sources], sources)
    return answer, trace, expected_evidence_checks([REFERENCE], sources, answer, trace)[0]


@pytest.mark.parametrize("raw,selected", [("Fact [cite:1]", False), ("Fact [cite:2]", True),
                                        ("Fact [[cite:2]]", True), ("Fact [[Humedal#p.3]]", False),
                                        ("Fact [cite:99]", False), ("Fact without citation", False)])
def test_same_page_distractor_cannot_earn_exact_selection_credit(raw, selected):
    answer, trace, check = resolve(raw)
    assert check["in_context"]
    assert check["selected"] is selected
    assert check["selected_detail_indices"] == ([1] if selected else [])
    if "[[Humedal#p.3]]" in answer:
        assert check["cited"]  # Historical page metric intentionally unchanged.
    if trace and raw.endswith("[[Humedal#p.3]]"):
        assert trace[0]["selection"] == "ambiguous"


def test_explicit_unique_canonical_passage_is_exact_without_short_token():
    _, trace, check = resolve("Fact [[Humedal#p.3]]", [details()[1]])
    assert check["selected"] and trace[0]["form"] == "wikilink"


@pytest.mark.parametrize("trace", [None, []])
def test_missing_trace_is_unassessed_but_empty_trace_has_zero_hits(trace):
    check = expected_evidence_checks([REFERENCE], details(), "[[Humedal#p.3]]", trace)[0]
    assert check["cited"]
    assert check["selected"] is (None if trace is None else False)


@pytest.mark.parametrize("change", [
    {"source_detail_indices": [True]}, {"source_detail_indices": [-1]},
    {"source_detail_indices": [2]}, {"source_detail_indices": [0, 1]},
    {"citation_index": True}, {"citation_index": -1}, {"citation_index": 1},
    {"label": "Humedal#p.1"}, {"citation_ref": "cite:99"}, {"citation_ref": "cite:01"},
    {"form": "invented"}, {"selection": "ambiguous"},
    {"form": "wikilink", "citation_ref": None},
])
def test_malformed_or_inconsistent_trace_cannot_earn_credit(change):
    answer, trace, _ = resolve("Fact [cite:2]")
    trace[0].update(change)
    assert not expected_evidence_checks([REFERENCE], details(), answer, trace)[0]["selected"]


def test_duplicate_reference_or_wrong_page_never_counts_as_selected_evidence():
    sources = details()
    sources[0]["citation_ref"] = "cite:2"
    assert not resolve("Fact [cite:2]", sources)[2]["selected"]
    sources = details()
    sources[1]["locator"]["page"] = 1
    assert not resolve("Fact [cite:2]", sources)[2]["selected"]


def test_trace_indices_are_local_to_delivered_order_not_persistent_ids():
    sources = list(reversed(details()))
    sources[0]["chunk_id"] = 9999
    assert resolve("Fact [cite:2]", sources)[2]["selected_detail_indices"] == [0]


@pytest.mark.parametrize("trace_available,generate", [(False, True), (True, True), (True, False)])
def test_report_exposes_selection_gaps_and_preserves_historical_metrics(trace_available, generate):
    answer, trace, _ = resolve("37,5 hectáreas [cite:1]")
    session = MagicMock()
    session.turns = []
    payload = {"answer": answer, "sources": ["Humedal#p.3"], "source_details": details()}
    if trace_available:
        payload["citation_selections"] = deepcopy(trace)
    session.oracle.ask.return_value = payload
    session.oracle.retrieve.return_value = [{"title": "Humedal"}]
    case = EvalCase(id="same-page-distractor", question="Área?", expected_sources=["Humedal"],
                    expected_keywords=["37,5"], expected_evidence=[REFERENCE], review_criteria="Review")
    report = run_eval(session, [case], generate=generate, judge=False).summary()
    assessed = trace_available and generate
    assert report["aggregate"]["selection_assessed_n"] == int(assessed)
    assert report["aggregate"]["selected_evidence_recall"] == (0 if assessed else None)
    assert report["aggregate"]["selected_evidence_hits"] == 0
    assert bool(report["selection_gaps"]) is assessed
    if generate:
        assert report["aggregate"]["cited_evidence_recall"] == 1
        assert report["aggregate"]["pass_rate"] == 1
        assert report["manual_review"][0]["claim_support"] == "pending"
        assert report["manual_review"][0]["citation_selections"] == (trace if trace_available else None)
    else:
        assert report["aggregate"]["cited_evidence_recall"] is None
