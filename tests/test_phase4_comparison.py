"""Comparison policy, using fabricated unit-test reports, not hardware results."""
from copy import deepcopy
import hashlib
import json

import pytest

from eval.benchmark_extraction import _language_data
from eval.compare_extraction import compare, load_report, main


def report():
    quality = {
        "evaluation_semantics": "fixed-expectations-v2",
        "totals": {key: {"passed": 2, "total": 3} for key in
                   ("reading_order", "table_rows", "page_anchors")},
        "document_errors": [],
    }
    return {
        "version": 2, "inputs_unchanged": True,
        "inputs": {"manifest_sha256": "unit-manifest", "documents": ["unit-doc"],
                   "code_sha256": {"unit-file": "unit-code"}},
        "settings": {"timeout_s": 300, "include_ocr": True, "include_legacy": False,
                     "engines": ["pypdf"], "ocr_language": "spa", "ocr_max_pages": 0},
        "profile": {
            "label": "unit-host", "system": "Linux", "release": "unit-kernel",
            "architecture": "x86_64", "cpu_model": "unit-cpu", "logical_cpus": 4,
            "ram_gib": 16, "python": "unit-python", "packages": {"pypdf": "unit-version"},
            "ocr_environment": {"OMP_THREAD_LIMIT": "1"},
            "ocr_language_data_sha256": {"spa": "unit-weights"},
            "external_tools": {
                "tesseract": {"status": "ok", "path": "/one/bin", "output": "unit-version\nunit-library"},
                "tesseract_languages": {"status": "ok", "output": 'List of available languages in "/one" (2):\nspa\neng'},
            },
        },
        "results": [{
            "variant": "pypdf+ocr", "status": "ok", "repeats": 3,
            "wall_s_median": 1.0, "peak_rss_mb_median": 100,
            "quality_consistent": True, "quality": quality,
            "samples": [{"wall_s": 1.0, "peak_rss_mb": 100, "quality": deepcopy(quality)}
                        for _ in range(3)],
        }],
    }


def sync_quality(value):
    for sample in value["results"][0]["samples"]:
        sample["quality"] = deepcopy(value["results"][0]["quality"])


def test_self_comparison_is_not_another_hardware_profile():
    baseline = report()
    candidate = deepcopy(baseline)
    candidate["profile"]["label"] = "different-label"
    candidate["profile"]["external_tools"]["tesseract"]["path"] = "/another/bin"
    candidate["profile"]["external_tools"]["tesseract_languages"]["output"] = (
        'List of available languages in "/another" (2):\neng\nspa'
    )
    result = compare(baseline, candidate)
    assert result["comparable"] is True
    assert result["distinct_hardware_profiles"] is False
    assert result["cross_hardware_comparison_ready"] is False
    assert result["results"][0]["candidate_over_reference_wall_ratio"] == 1


def test_changed_hardware_and_costs_are_reported_without_claiming_a_physical_device():
    baseline, candidate = report(), report()
    candidate["profile"]["cpu_model"] = "unit-other-cpu"
    row = candidate["results"][0]
    row["wall_s_median"], row["peak_rss_mb_median"] = 2, 50
    for sample in row["samples"]:
        sample["wall_s"], sample["peak_rss_mb"] = 2, 50
    result = compare(baseline, candidate)
    assert result["cross_hardware_comparison_ready"] is True
    assert result["results"][0]["candidate_over_reference_wall_ratio"] == 2
    assert result["results"][0]["candidate_over_reference_rss_ratio"] == .5
    assert "do not prove" in result["limitations"][0]


@pytest.mark.parametrize("field,value", [
    ("ocr_max_pages", 1), ("ocr_language", "eng"), ("timeout_s", 30),
    ("engines", ["pdfplumber"]), ("include_ocr", False),
])
def test_different_settings_have_no_ratios(field, value):
    baseline, candidate = report(), report()
    candidate["settings"][field] = value
    result = compare(baseline, candidate)
    assert result["comparable"] is False
    assert result["quality_regression"] is None
    assert "candidate_over_reference_wall_ratio" not in result["results"][0]


@pytest.mark.parametrize("field,value", [
    ("python", "another-python"), ("packages", {}), ("ocr_environment", {}),
    ("release", "another-kernel"), ("ocr_language_data_sha256", {}),
    ("ocr_language_data_sha256", {"spa": "different-weights"}),
])
def test_different_or_missing_software_blocks_comparison(field, value):
    baseline, candidate = report(), report()
    candidate["profile"][field] = value
    assert compare(baseline, candidate)["comparable"] is False


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "different"), ("documents", ["different"]),
    ("code_sha256", {"unit-file": "different"}),
])
def test_changed_corpus_or_code_blocks_comparison(field, value):
    baseline, candidate = report(), report()
    candidate["inputs"][field] = value
    assert compare(baseline, candidate)["comparable"] is False


def test_mutated_inputs_and_external_library_versions_block_comparison():
    baseline, candidate = report(), report()
    candidate["inputs_unchanged"] = False
    assert compare(baseline, candidate)["comparable"] is False
    candidate = report()
    candidate["profile"]["external_tools"]["tesseract"]["output"] = "unit-version\nother-library"
    assert compare(baseline, candidate)["comparable"] is False


def test_cpu_instruction_detection_does_not_count_as_software_change():
    baseline, candidate = report(), report()
    candidate["profile"]["external_tools"]["tesseract"]["output"] += "\n Found AVX2\n Found FMA\n Found SSE4.1"
    assert compare(baseline, candidate)["comparable"] is True


@pytest.mark.parametrize("field,value", [
    ("repeats", 2), ("repeats", True), ("repeats", None), ("quality_consistent", False),
    ("wall_s_median", float("nan")), ("wall_s_median", None),
    ("wall_s_median", 0), ("wall_s_median", True), ("wall_s_median", 2),
    ("peak_rss_mb_median", 2), ("samples", []), ("samples", [None] * 3),
    ("status", "skipped"),
])
def test_inconsistent_measurements_have_no_ratios(field, value):
    baseline, candidate = report(), report()
    candidate["results"][0][field] = value
    result = compare(baseline, candidate)
    assert result["comparable"] is False
    assert "candidate_over_reference_rss_ratio" not in result["results"][0]


def test_different_sample_quality_is_not_hidden_by_summary():
    baseline, candidate = report(), report()
    candidate["results"][0]["samples"][1]["quality"]["totals"]["reading_order"]["passed"] = 0
    assert compare(baseline, candidate)["comparable"] is False


@pytest.mark.parametrize("score", [
    {"passed": 0, "total": 0}, {"passed": 4, "total": 3},
    {"passed": True, "total": 3}, {"passed": 1, "total": "3"}, {},
])
def test_invalid_counts_are_not_comparable(score):
    baseline, candidate = report(), report()
    candidate["results"][0]["quality"]["totals"]["reading_order"] = score
    sync_quality(candidate)
    assert compare(baseline, candidate)["comparable"] is False


def test_empty_quality_and_historical_semantics_are_not_comparable():
    baseline, candidate = report(), report()
    candidate["results"][0]["quality"]["totals"] = {}
    sync_quality(candidate)
    assert compare(baseline, candidate)["comparable"] is False
    candidate = report()
    candidate["results"][0]["quality"].pop("evaluation_semantics")
    sync_quality(candidate)
    assert compare(baseline, candidate)["comparable"] is False


def test_denominators_regressions_and_document_errors_are_distinct():
    baseline, candidate = report(), report()
    candidate["results"][0]["quality"]["totals"]["table_rows"]["total"] = 4
    sync_quality(candidate)
    assert compare(baseline, candidate)["comparable"] is False
    candidate = report()
    candidate["results"][0]["quality"]["totals"]["table_rows"]["passed"] = 1
    sync_quality(candidate)
    result = compare(baseline, candidate)
    assert result["comparable"] is True
    assert result["quality_regression"] is True
    assert result["results"][0]["passed_expectations_delta"]["table_rows"] == -1
    candidate = report()
    candidate["results"][0]["quality"]["document_errors"] = ["unit-doc"]
    sync_quality(candidate)
    assert compare(baseline, candidate)["quality_regression"] is True


def test_absent_memory_remains_null_and_unknown_hardware_is_not_ready():
    baseline, candidate = report(), report()
    for item in (baseline, candidate):
        row = item["results"][0]
        row["peak_rss_mb_median"] = None
        for sample in row["samples"]:
            sample["peak_rss_mb"] = None
    candidate["profile"]["cpu_model"] = None
    result = compare(baseline, candidate)
    assert result["comparable"] is True
    assert result["cross_hardware_comparison_ready"] is False
    assert result["results"][0]["candidate_over_reference_rss_ratio"] is None


def test_equal_aggregate_scores_do_not_hide_new_failed_cases():
    baseline, candidate = report(), report()
    baseline["results"][0]["quality"]["failed_checks"] = [{"kind": "table_rows", "index": 0}]
    candidate["results"][0]["quality"]["failed_checks"] = [{"kind": "table_rows", "index": 1}]
    sync_quality(baseline)
    sync_quality(candidate)
    assert compare(baseline, candidate)["quality_regression"] is True


@pytest.mark.parametrize("mode,exit_code", [("self", 0), ("regression", 1), ("changed", 1)])
def test_cli_emits_json_and_signals_blockers_or_quality_regressions(tmp_path, monkeypatch, capsys, mode, exit_code):
    baseline, candidate = report(), report()
    if mode == "regression":
        candidate["results"][0]["quality"]["totals"]["table_rows"]["passed"] = 1
        sync_quality(candidate)
    if mode == "changed":
        candidate["settings"]["ocr_max_pages"] = 1
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(baseline))
    right.write_text(json.dumps(candidate))
    monkeypatch.setattr("sys.argv", ["compare_extraction.py", str(left), str(right)])
    assert main() == exit_code
    output = json.loads(capsys.readouterr().out)
    assert output["comparable"] is (mode != "changed")


def test_duplicate_missing_and_empty_variants_are_not_accepted():
    baseline, candidate = report(), report()
    candidate["results"].append(deepcopy(candidate["results"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        compare(baseline, candidate)
    candidate = report()
    candidate["results"][0]["variant"] = "different"
    assert compare(baseline, candidate)["comparable"] is False
    candidate["results"] = []
    with pytest.raises(ValueError, match="non-empty"):
        compare(baseline, candidate)


def test_report_loading_rejects_old_version_and_oversized_files(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()))
    assert load_report(path)["version"] == 2
    path.write_text('{"version": 1}')
    with pytest.raises(ValueError, match="version-2"):
        load_report(path)
    with path.open("wb") as stream:
        stream.truncate(8_000_001)
    with pytest.raises(ValueError, match="8 MB"):
        load_report(path)


def test_ocr_weights_are_hashed_and_missing_languages_are_explicit(tmp_path):
    weights = tmp_path / "spa.traineddata"
    weights.write_bytes(b"unit-test-language-data")
    output = f'List of available languages in "{tmp_path}" (3):\nspa\neng\n../escape'
    assert _language_data({"status": "ok", "output": output}) == {
        "spa": hashlib.sha256(weights.read_bytes()).hexdigest(), "eng": None,
    }
    assert _language_data({"status": "error", "output": output}) == {}
    assert _language_data({"status": "ok", "output": 'List of available languages in "relative" (1):\nspa'}) == {}
