"""Literal, locator-specific coverage is separate from human claim support."""
from unittest.mock import MagicMock

import pytest

from grimore.cognition.eval import EvalCase, expected_evidence_checks, run_eval
from grimore.cognition.oracle import Oracle


def detail(text="El área es de 37,5 hectáreas", page=3, label="Humedal#p.3"):
    return {"label": label, "document": {"title": "Humedal"},
            "locator": {"page": page}, "passage": {"text": text}}


REFERENCE = {"source": "Humedal", "passage": "El área es de 37,5 hectáreas", "locator": "p.3"}


@pytest.mark.parametrize("page", [1, 0, True, "3", None])
def test_wrong_physical_page_does_not_match(page):
    check = expected_evidence_checks([REFERENCE], [detail(page=page)], "[[Humedal#p.3]]")[0]
    assert not check["in_context"] and not check["cited"]


@pytest.mark.parametrize("citation", ["[[Humedal]]", "[[Humedal#p.1]]", "Humedal#p.3", ""])
def test_correct_context_without_correct_citation(citation):
    check = expected_evidence_checks([REFERENCE], [detail()], citation)[0]
    assert check["in_context"] and not check["cited"]


def test_exact_reference_tolerates_only_case_whitespace_and_unicode_composition():
    check = expected_evidence_checks([REFERENCE], [detail("EL A\u0301REA\tes de\n37,5 hectáreas")],
                                     "37,5 [[Humedal#p.3]]")[0]
    assert check["in_context"] and check["cited"]


@pytest.mark.parametrize("text", ["El area es de 37,5 hectáreas", "El área es de 375 hectáreas",
                                   "El área es de 37.5 hectáreas", "El área es de 37,5"])
def test_no_fuzzy_numeric_or_accent_match(text):
    assert not expected_evidence_checks([REFERENCE], [detail(text)], "[[Humedal#p.3]]")[0]["cited"]


def test_cannot_stitch_evidence_across_passages():
    details = [detail("El área es de"), detail("37,5 hectáreas")]
    assert not expected_evidence_checks([REFERENCE], details, "[[Humedal#p.3]]")[0]["in_context"]


def test_label_inconsistent_with_page_does_not_earn_citation_credit():
    check = expected_evidence_checks([REFERENCE], [detail(label="Humedal#p.1")], "[[Humedal#p.1]]")[0]
    assert check["in_context"] and not check["cited"]


def test_heading_and_legacy_detail_without_locator():
    reference = {"source": "A", "passage": "Exact passage", "locator": "Sampling"}
    source = {"label": "A#Sampling", "locator": {"heading": "Sampling"},
              "passage": {"text": "Exact passage"}}
    assert expected_evidence_checks([reference], [source], "[[A#Sampling]]")[0]["cited"]
    source["locator"]["heading"] = "Other"
    assert not expected_evidence_checks([reference], [source], "[[A#Sampling]]")[0]["in_context"]
    reference["locator"] = ""
    assert expected_evidence_checks([reference], [source], "[[A#Sampling]]")[0]["cited"]


@pytest.mark.parametrize("payload,generate,negative,assessed", [
    ({}, True, False, 0), ({"source_details": []}, True, False, 1),
    ({"source_details": [detail()]}, False, False, 0),
    ({"source_details": [detail()]}, True, True, 0),
    ({"source_details": [detail()]}, True, False, 1),
])
def test_report_unassessed_vs_explicit_empty_context_and_manual_review(payload, generate, negative, assessed):
    session = MagicMock()
    session.turns = []
    session.oracle.ask.return_value = {"answer": "37,5 [[Humedal#p.3]]", "sources": ["Humedal#p.3"], **payload}
    session.oracle.retrieve.return_value = []
    case = EvalCase(id="coverage", question="Área?", expected_sources=[] if negative else ["Humedal"],
                    negative=negative, expected_evidence=[REFERENCE], review_criteria="Review the claim")
    report = run_eval(session, [case], generate=generate, judge=False).summary()
    assert report["aggregate"]["evidence_assessed_n"] == assessed
    if not assessed:
        assert report["aggregate"]["context_evidence_recall"] is None
    elif not payload["source_details"]:
        assert report["aggregate"]["context_evidence_recall"] == 0
        assert report["evidence_gaps"]
    else:
        assert report["aggregate"]["cited_evidence_hits"] == 1
    assert report["aggregate"]["automated_claim_support"] is None
    if generate and not negative:
        assert report["manual_review"][0]["claim_support"] == "pending"


def test_citation_verifier_rejects_wrong_page_and_heading_but_keeps_bare_title():
    text = "[[A#p.99]] [[A#Wrong]] [[A#p.2]] [[A]] [[B#p.2]] [[C#Made up]]"
    cleaned, dropped = Oracle.verify_citations(text, ["A#p.2", "B#Section", "C"])
    assert dropped == 4
    assert cleaned == "A#p.99 A#Wrong [[A#p.2]] [[A]] B#p.2 C#Made up"
