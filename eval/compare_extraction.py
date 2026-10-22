"""Compare extraction reports conservatively without attributing setup changes to hardware.

Inputs are existing benchmark JSON files; this command never reruns a model,
reads corpus documents or writes to the source reports.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

HARDWARE_FIELDS = ("system", "architecture", "cpu_model", "logical_cpus", "ram_gib")
SETTING_DEFAULTS = {"ocr_language": "eng", "ocr_max_pages": 0}
SETTINGS = ("timeout_s", "include_ocr", "include_legacy", "engines", "ocr_language", "ocr_max_pages")
QUALITY_KINDS = {"reading_order", "table_rows", "page_anchors"}


def load_report(path: Path) -> dict:
    if path.stat().st_size > 8_000_000:
        raise ValueError("benchmark report exceeds 8 MB")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("version") != 2:
        raise ValueError("requires a version-2 extraction benchmark report")
    for field in ("inputs", "profile", "settings"):
        if not isinstance(report.get(field), dict):
            raise ValueError(f"missing benchmark {field}")
    if not isinstance(report.get("results"), list) or not report["results"]:
        raise ValueError("missing benchmark results")
    return report


def _tool_versions(profile: dict) -> dict:
    # Installation paths and the --list-langs header are machine-local.
    # Version lines and actual language identifiers must still match.
    tools = profile.get("external_tools") or {}
    if not isinstance(tools, dict):
        raise ValueError("invalid external tool provenance")
    result = {}
    for name, tool in tools.items():
        if not isinstance(tool, dict) or not isinstance(tool.get("output") or "", str):
            raise ValueError("invalid external tool record")
        output = (tool.get("output") or "").splitlines()
        value = sorted(output[1:]) if name == "tesseract_languages" else [
            line for line in output if not re.fullmatch(
                r"\s*Found (?:AVX\w*|FMA|SSE[\d.]*|NEON)\s*", line,
            )
        ]
        result[name] = {"status": tool.get("status"), "identity": value}
    return result


def _finite_number(value, *, positive=False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def compare(reference: dict, candidate: dict) -> dict:
    """Return comparability, quality changes and ratios only when trustworthy."""
    reasons = []
    for report in (reference, candidate):
        if report.get("version") != 2 or any(
            not isinstance(report.get(key), dict) for key in ("inputs", "settings", "profile")
        ) or not isinstance(report.get("results"), list) or not report["results"]:
            raise ValueError("requires non-empty version-2 benchmark reports")
    for label, report in (("reference", reference), ("candidate", candidate)):
        if report.get("inputs_unchanged") is not True:
            reasons.append(f"{label}: inputs were not stable during measurement")
        inputs = report.get("inputs", {})
        if not all(inputs.get(key) for key in ("manifest_sha256", "documents", "code_sha256")):
            reasons.append(f"{label}: missing corpus/code provenance")
    if reference["inputs"] != candidate["inputs"]:
        reasons.append("corpus or measured code fingerprints differ")
    for key in SETTINGS:
        a = reference["settings"].get(key, SETTING_DEFAULTS.get(key))
        b = candidate["settings"].get(key, SETTING_DEFAULTS.get(key))
        if a != b or (a is None and key not in SETTING_DEFAULTS):
            reasons.append(f"settings differ or are unknown: {key}")
    for key in ("system", "release", "python", "packages", "ocr_environment"):
        a, b = reference["profile"].get(key), candidate["profile"].get(key)
        if a is None or b is None or a != b or (key != "ocr_environment" and not a):
            reasons.append(f"software/environment differs or is unknown: {key}")
    if _tool_versions(reference["profile"]) != _tool_versions(candidate["profile"]):
        reasons.append("external tool versions or OCR languages differ")
    if not reference["profile"].get("external_tools") or not candidate["profile"].get("external_tools"):
        reasons.append("external tool provenance is missing")
    if reference["settings"].get("include_ocr") or candidate["settings"].get("include_ocr"):
        a = reference["profile"].get("ocr_language_data_sha256")
        b = candidate["profile"].get("ocr_language_data_sha256")
        language_setting = reference["settings"].get("ocr_language", "eng")
        if not isinstance(language_setting, str) or not language_setting:
            raise ValueError("invalid OCR language setting")
        if (a is not None and not isinstance(a, dict)) or (b is not None and not isinstance(b, dict)):
            raise ValueError("invalid OCR language data fingerprints")
        languages = language_setting.split("+")
        if not a or not b or any(not a.get(lang) or not b.get(lang) for lang in languages):
            reasons.append("requested OCR language data fingerprints are missing")
        elif any(a[lang] != b[lang] for lang in languages):
            reasons.append("OCR language data fingerprints differ")

    hardware = {
        key: {"reference": reference["profile"].get(key), "candidate": candidate["profile"].get(key)}
        for key in HARDWARE_FIELDS
        if reference["profile"].get(key) != candidate["profile"].get(key)
    }
    maps = []
    for report in (reference, candidate):
        rows = report["results"]
        if any(not isinstance(row, dict) or not isinstance(row.get("variant"), str)
               or not row["variant"] for row in rows):
            raise ValueError("invalid benchmark variant")
        mapping = {row["variant"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("duplicate benchmark variants")
        maps.append(mapping)
    left, right = maps
    comparisons = []
    for variant in sorted(set(left) | set(right)):
        a, b = left.get(variant), right.get(variant)
        problems = list(reasons)
        if a is None or b is None:
            problems.append("variant missing from one report")
        elif a.get("status") != "ok" or b.get("status") != "ok":
            problems.append("variant was skipped or failed")
        else:
            for label, row in (("reference", a), ("candidate", b)):
                if row.get("quality_consistent") is not True:
                    problems.append(f"{label}: quality differs between samples")
                repeats = row.get("repeats")
                if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
                    problems.append(f"{label}: requires at least three samples")
                if not _finite_number(row.get("wall_s_median"), positive=True):
                    problems.append(f"{label}: invalid elapsed time")
                samples = row.get("samples")
                if (not isinstance(samples, list) or len(samples) != row.get("repeats")
                        or not samples or any(not isinstance(sample, dict)
                               or sample.get("quality") != row.get("quality")
                               or not _finite_number(sample.get("wall_s"), positive=True)
                               for sample in samples)):
                    problems.append(f"{label}: samples do not substantiate summary")
                else:
                    if _finite_number(row.get("wall_s_median"), positive=True) and not math.isclose(
                        statistics.median(sample["wall_s"] for sample in samples),
                        row["wall_s_median"], abs_tol=0.000001, rel_tol=0,
                    ):
                        problems.append(f"{label}: elapsed median does not match samples")
                    peaks = [sample.get("peak_rss_mb") for sample in samples]
                    if any(value is not None and not _finite_number(value) for value in peaks):
                        problems.append(f"{label}: invalid RSS samples")
                    else:
                        median = round(statistics.median(peaks), 2) if all(
                            value is not None for value in peaks
                        ) else None
                        if row.get("peak_rss_mb_median") != median:
                            problems.append(f"{label}: RSS median does not match samples")
                quality = row.get("quality", {})
                if not isinstance(quality, dict) or not isinstance(quality.get("totals"), dict):
                    raise ValueError("invalid quality summary")
                if quality.get("evaluation_semantics") != "fixed-expectations-v2":
                    problems.append(f"{label}: unsupported quality semantics")
                if set(quality["totals"]) != QUALITY_KINDS:
                    problems.append(f"{label}: missing or unknown quality dimensions")
                for score in quality.get("totals", {}).values():
                    if not isinstance(score, dict):
                        raise ValueError("invalid expectation score")
                    if (isinstance(score.get("total"), bool) or not isinstance(score.get("total"), int)
                            or score["total"] < 1 or isinstance(score.get("passed"), bool)
                            or not isinstance(score.get("passed"), int)
                            or not 0 <= score["passed"] <= score["total"]):
                        problems.append(f"{label}: invalid expectation counts")
            totals_a, totals_b = a["quality"]["totals"], b["quality"]["totals"]
            if set(totals_a) != set(totals_b) or any(
                totals_a[k].get("total") != totals_b[k].get("total") for k in totals_a
            ):
                problems.append("expectation denominators differ")
        row_result = {"variant": variant, "comparable": not problems, "reasons": problems}
        if not problems:
            deltas = {kind: b["quality"]["totals"][kind]["passed"] - score["passed"]
                      for kind, score in a["quality"]["totals"].items()}
            memory_a, memory_b = a.get("peak_rss_mb_median"), b.get("peak_rss_mb_median")
            # Equal aggregate counts can still hide a newly broken fixed case.
            failed_a = {json.dumps(check, sort_keys=True) for check in a["quality"].get("failed_checks", [])}
            failed_b = {json.dumps(check, sort_keys=True) for check in b["quality"].get("failed_checks", [])}
            row_result.update({
                "candidate_over_reference_wall_ratio": round(b["wall_s_median"] / a["wall_s_median"], 6),
                "candidate_over_reference_rss_ratio": (
                    round(memory_b / memory_a, 6)
                    if _finite_number(memory_a, positive=True) and _finite_number(memory_b) else None
                ),
                "passed_expectations_delta": deltas,
                "quality_regression": bool(failed_b - failed_a) or any(delta < 0 for delta in deltas.values()) or bool(
                    set(b["quality"].get("document_errors", []))
                    - set(a["quality"].get("document_errors", []))
                ),
            })
        comparisons.append(row_result)
    comparable = all(row["comparable"] for row in comparisons)
    regression = any(row.get("quality_regression", False) for row in comparisons)
    return {
        "version": 1, "comparable": comparable,
        "hardware_profile_differences": hardware,
        "distinct_hardware_profiles": bool(hardware),
        "cross_hardware_comparison_ready": comparable and bool(hardware) and all(
            report["profile"].get(key) not in (None, "")
            for report in (reference, candidate) for key in HARDWARE_FIELDS
        ),
        "quality_regression": regression if comparable or regression else None,
        "results": comparisons,
        "limitations": [
            "Hardware descriptors do not prove two physical devices were measured.",
            "Ratios describe these runs only, not general hardware performance.",
            "Incomparable variants have no performance ratios.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        report = compare(load_report(args.reference), load_report(args.candidate))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["comparable"] and not report["quality_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
