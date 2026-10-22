"""Paired opt-in/off diagnostics, never semantic acceptance or hardware attribution.

Reads existing reports only. Preserves failed/unassessed cases and their
denominators; never runs models, rewrites reports or fills missing citations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct-script execution needs the repository package, as in the answer bank.
# ruff: noqa: E402
from grimore.cognition.answer_protocol import technical_failure


def load_report(path: Path) -> dict:
    if path.stat().st_size > 8_000_000:
        raise ValueError("answer report exceeds 8 MB")
    def reject_nonfinite(value):
        raise ValueError(f"non-finite JSON value: {value}")
    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    if not isinstance(report, dict):
        raise ValueError("requires an answer report object")
    if (report.get("controlled_run") or {}).get("version") != 3:
        raise ValueError("requires a version-3 controlled answer report")
    _turns(report)
    return report


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _turns(report):
    rows = report.get("turns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("requires nonempty answer turns")
    mapping = {}
    for turn in rows:
        if (not isinstance(turn, dict) or not isinstance(turn.get("case_id"), str)
                or type(turn.get("turn_index")) is not int or type(turn.get("negative")) is not bool):
            raise ValueError("invalid answer turn identity")
        key = (turn["case_id"], turn["turn_index"])
        if key in mapping:
            raise ValueError("duplicate answer turn")
        mapping[key] = turn
    return mapping


def _statistics(report):
    turns = list(_turns(report).values())
    pos = [t for t in turns if not t["negative"]]
    assessed = [t for t in pos if t.get("context_evidence_recall") is not None]
    latencies = [t["latency_s"] for t in turns if _number(t.get("latency_s"))]
    return {"turns_n": len(turns), "positive_n": len(pos), "negative_n": len(turns) - len(pos),
            "positive_full_keyword_hits_n": sum(t.get("keyword_recall") == 1
                                                and not technical_failure(t.get("sufficiency")) for t in pos),
            "positive_cited_n": sum(bool(t.get("citation_selections")) for t in pos),
            "positive_evidence_assessed_n": len(assessed),
            "positive_evidence_unassessed_n": len(pos) - len(assessed),
            "positive_full_selected_evidence_n": sum(t.get("selected_evidence_recall") == 1 for t in pos),
            "evidence_control_failure_n": sum(technical_failure(t.get("sufficiency")) for t in turns),
            "latency_assessed_n": len(latencies),
            "latency_median_s": statistics.median(latencies) if latencies else None,
            "memory": report["controlled_run"].get("memory"),
            "negative_control": report["controlled_run"].get("negative_control"),
            "human_claim_support": "pending", "human_abstention": "pending"}


def compare(reference: dict, candidate: dict) -> dict:
    reasons = []
    runs = []
    for label, report in (("reference", reference), ("candidate", candidate)):
        run = report.get("controlled_run")
        if not isinstance(run, dict) or run.get("version") != 3:
            raise ValueError("requires version-3 controlled answer reports; historical reports remain separate")
        runs.append(run)
        inputs = run.get("inputs") or {}
        if (not all(inputs.get(k) for k in ("cases_sha256", "pdf_sha256", "config_file_sha256", "code_sha256"))
                or run.get("inputs_unchanged") is not True or run.get("copied_pdf_unchanged") is not True):
            reasons.append(f"{label}: missing or unstable input provenance")
        if not run.get("models") or any(not m.get("digest") for m in run["models"].values()):
            reasons.append(f"{label}: missing model digests")
    a, b = runs
    for key in ("inputs", "models", "ollama_version", "ingest", "retrieval_k",
                "client_hardware", "endpoint", "generation_samples_per_case", "judge_enabled"):
        if key not in a or key not in b or a[key] != b[key]:
            reasons.append(f"paired setup differs or is unknown: {key}")
    settings = []
    for run in runs:
        cognition = dict(run.get("cognition") or {})
        cognition.pop("answer_sufficiency_check", None)
        settings.append(cognition)
    if not settings[0] or settings[0] != settings[1]:
        reasons.append("cognition differs beyond the explicit sufficiency switch")
    if ((a.get("sufficiency_control") or {}).get("enabled") is not False
            or (b.get("sufficiency_control") or {}).get("enabled") is not True
            or (a.get("cognition") or {}).get("answer_sufficiency_check") is not False
            or (b.get("cognition") or {}).get("answer_sufficiency_check") is not True):
        reasons.append("requires an off reference and on candidate")
    if (b.get("sufficiency_control") or {}).get("answer_protocol_version") != 1:
        reasons.append("candidate answer protocol is unknown")
    presentation = [run.get("presentation_control") or {} for run in runs]
    if (not presentation[0].get("strategy") or presentation[0] != presentation[1]):
        reasons.append("accepted presentation/position/maps differ or are unknown")
    left, right = _turns(reference), _turns(candidate)
    if set(left) != set(right):
        reasons.append("turn identities differ")
    observations = []
    for key in sorted(set(left) | set(right)):
        before, after = left.get(key), right.get(key)
        if before is None or after is None:
            observations.append({"case_id": key[0], "turn_index": key[1], "missing_from_one_report": True})
            continue
        for field in ("question", "negative", "retrieved", "expected_sources", "expected_evidence", "review_criteria"):
            if field not in before or field not in after or before[field] != after[field]:
                reasons.append(f"{key[0]}/{key[1]}: turn data/ranking differs or is unknown: {field}")
        assessment = after.get("sufficiency")
        if not isinstance(assessment, dict):
            reasons.append(f"{key[0]}/{key[1]}: missing candidate assessment")
        else:
            if assessment.get("version") != 2:
                reasons.append(f"{key[0]}/{key[1]}: candidate assessment protocol is unknown")
            if assessment.get("decision") == "answerable" and not isinstance(assessment.get("generation"), dict):
                reasons.append(f"{key[0]}/{key[1]}: missing generation assessment")
            if before.get("source_details") != assessment.get("candidate_source_details"):
                reasons.append(f"{key[0]}/{key[1]}: accepted candidate passages differ")
        observations.append({"case_id": key[0], "turn_index": key[1], "negative": before["negative"],
                             "reference": {k: before.get(k) for k in ("answer", "answer_status", "keyword_recall", "context_evidence_recall", "cited_evidence_recall", "selected_evidence_recall")},
                             "candidate": {k: after.get(k) for k in ("answer", "answer_status", "keyword_recall", "context_evidence_recall", "cited_evidence_recall", "selected_evidence_recall")},
                             "candidate_technical_failure": technical_failure(assessment)})
    comparable = not reasons
    return {"version": 1, "comparable": comparable, "reasons": sorted(set(reasons)),
            "reference": _statistics(reference), "candidate": _statistics(candidate),
            "paired_observations": observations if comparable else None,
            "semantic_acceptance": "pending-human-review",
            "limitations": ["Keyword hits, reference validity and quote provenance are not semantic support.",
                            "Failures stay in the case counts; unassessed cases are not successful refusals.",
                            "Paired runs are sequential; cache/residency and background load are not controlled.",
                            "RSS excludes the inference server; do not add separate peak RSS values.",
                            "No speedup or generalization is inferred from a single pair."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = compare(load_report(args.reference), load_report(args.candidate))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
