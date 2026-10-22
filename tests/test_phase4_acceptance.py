"""Plan/comparison/subprocess contracts only; never contact Ollama."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest

from eval import run_phase4_acceptance as runner
from eval.compare_answers import compare, load_report
from tests.test_answer_sufficiency import details


def paired():
    run = {"version": 3, "inputs": {"cases_sha256": "a", "pdf_sha256": "b", "config_file_sha256": "c", "code_sha256": {"file": "d"}},
           "inputs_unchanged": True, "copied_pdf_unchanged": True,
           "models": {"local": {"digest": "model-digest"}}, "ollama_version": "same",
           "ingest": {"pdf_engine": "pdfplumber"}, "retrieval_k": 10,
           "client_hardware": {"cpu_model": "same CPU"}, "endpoint": "http://127.0.0.1:11434",
           "generation_samples_per_case": 1, "judge_enabled": False,
           "cognition": {"answer_sufficiency_check": False, "request_timeout_s": 600},
           "sufficiency_control": {"enabled": False, "answer_protocol_version": None},
           "presentation_control": {"strategy": "ranked", "calls": [{"context_sha256_after": "same"}]},
           "memory": {"client_peak_rss_mb": 100, "inference_peak_rss_mb": None}}
    turn = {"case_id": "positive", "turn_index": 0, "negative": False,
            "question": "Question?", "retrieved": ["Document"], "expected_sources": ["Document"],
            "expected_evidence": [], "review_criteria": "Review", "source_details": details(),
            "answer": "Fact", "answer_status": "supported", "keyword_recall": 1.0,
            "context_evidence_recall": 1.0, "cited_evidence_recall": 0.0,
            "selected_evidence_recall": 0.0, "citation_selections": [], "latency_s": 2.0}
    before = {"controlled_run": run, "turns": [turn]}
    after = deepcopy(before)
    after["controlled_run"]["sufficiency_control"].update(enabled=True, answer_protocol_version=1)
    after["controlled_run"]["cognition"]["answer_sufficiency_check"] = True
    after["turns"][0]["sufficiency"] = {"version": 2, "decision": "answerable", "candidate_source_details": details(),
                                       "generation": {"decision": "accepted"}}
    after["turns"][0]["source_details"] = [details()[1]]
    return before, after


def test_comparison_checks_pre_selection_context_and_never_claims_semantic_acceptance():
    before, after = paired()
    result = compare(before, after)
    assert result["comparable"]
    assert result["candidate"]["positive_n"] == 1
    assert result["semantic_acceptance"] == "pending-human-review"
    assert result["candidate"]["memory"]["inference_peak_rss_mb"] is None


@pytest.mark.parametrize("mutation", ["model", "corpus", "code", "unstable", "hardware", "order", "ranking", "proof_map", "missing_generation", "settings", "same_mode"])
def test_mismatched_reports_have_no_paired_observations(mutation):
    before, after = paired()
    run, turn = after["controlled_run"], after["turns"][0]
    if mutation == "model":
        run["models"]["local"]["digest"] = "different"
    elif mutation == "corpus":
        run["inputs"]["pdf_sha256"] = "different"
    elif mutation == "code":
        run["inputs"]["code_sha256"] = {}
    elif mutation == "unstable":
        run["inputs_unchanged"] = False
    elif mutation == "hardware":
        run["client_hardware"] = {"cpu_model": "different"}
    elif mutation == "order":
        run["presentation_control"]["strategy"] = "rotate-first-last"
    elif mutation == "ranking":
        turn["retrieved"] = ["Other document"]
    elif mutation == "proof_map":
        turn["sufficiency"]["candidate_source_details"] = []
    elif mutation == "missing_generation":
        del turn["sufficiency"]["generation"]
    elif mutation == "settings":
        run["cognition"]["request_timeout_s"] = 60
    elif mutation == "same_mode":
        run["sufficiency_control"]["enabled"] = False
    result = compare(before, after)
    assert not result["comparable"] and result["reasons"]
    assert result["paired_observations"] is None


def test_unassessed_case_remains_in_counts_not_as_a_successful_refusal():
    before, after = paired()
    after["turns"][0]["sufficiency"]["generation"]["decision"] = "unassessed"
    after["turns"][0]["context_evidence_recall"] = None
    result = compare(before, after)
    assert result["comparable"]
    assert result["candidate"]["positive_n"] == 1
    assert result["candidate"]["positive_evidence_unassessed_n"] == 1
    assert result["candidate"]["evidence_control_failure_n"] == 1


def test_duplicates_and_historical_report_versions_are_not_reinterpreted(tmp_path):
    before, after = paired()
    before["turns"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        compare(before, after)
    after["controlled_run"]["version"] = 2
    path = tmp_path / "historical.json"
    original = json.dumps(after)
    path.write_text(original)
    with pytest.raises(ValueError, match="version-3"):
        load_report(path)
    assert path.read_text() == original


def schedule(repeat=1):
    # The tracked example, not the developer's untracked grimore.toml: the
    # plan must be reproducible on a fresh clone (CI has no local config).
    return runner.plan(runner.DEFAULT_CASES, runner.DEFAULT_NEGATIVE, runner.DEFAULT_PDF,
                       runner.REPO_ROOT / "grimore.toml.example", repeat=repeat)


def test_default_is_only_a_plan_with_complete_pairs_and_alternating_order(monkeypatch):
    child = MagicMock(side_effect=AssertionError("no measurements while planning"))
    monkeypatch.setattr(runner.subprocess, "run", child)
    planned = schedule(repeat=2)
    assert planned["status"] == "planned-not-executed"
    assert len(planned["runs"]) == 12 and planned["oracle_turns"] == 36
    assert planned["runs"][0]["enabled"] is False
    assert planned["runs"][6]["enabled"] is True
    child.assert_not_called()


@pytest.mark.parametrize("repeat", [0, -1, True, 11])
def test_invalid_repeat_cannot_trigger_measurement(repeat):
    with pytest.raises(ValueError, match="repeat"):
        schedule(repeat)


def test_plan_refuses_mixed_roles_without_reading_held_out_cases(monkeypatch):
    original = runner.load_golden
    def mixed(path):
        cases = original(path)
        if path == runner.DEFAULT_CASES:
            cases[0].negative = True
        return cases
    monkeypatch.setattr(runner, "load_golden", mixed)
    with pytest.raises(ValueError, match="separate"):
        schedule()


def test_existing_output_directory_is_preserved_before_starting_a_child(tmp_path, monkeypatch):
    planned = schedule()
    marker = tmp_path / "preserve.json"
    marker.write_text("keep")
    child = MagicMock()
    monkeypatch.setattr(runner.subprocess, "run", child)
    with pytest.raises(FileExistsError):
        runner.execute(planned, tmp_path)
    assert marker.read_text() == "keep"
    child.assert_not_called()


def test_changed_inputs_block_all_writes_and_child_processes(tmp_path, monkeypatch):
    planned = schedule()
    monkeypatch.setattr(runner, "fingerprints", lambda *a: {"different": True})
    child = MagicMock()
    monkeypatch.setattr(runner.subprocess, "run", child)
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="changed"):
        runner.execute(planned, output)
    assert not output.exists()
    child.assert_not_called()


def test_failed_processes_get_diagnostics_and_never_successful_pairs(tmp_path, monkeypatch):
    planned = schedule()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 2, stderr="controlled failure"))
    output = tmp_path / "new"
    result = runner.execute(planned, output)
    assert result["status"] == "incomplete-measurements" and result["pairs"] == []
    assert len(result["attempts"]) == 6
    assert all(o["status"] == "failed" for o in result["attempts"])
    assert len(list(output.glob("*.status.json"))) == 6
    assert json.loads((output / "summary.json").read_text()) == result


def test_client_timeout_stops_batch_without_piling_unknown_server_work(tmp_path, monkeypatch):
    planned = schedule()
    child = MagicMock(side_effect=subprocess.TimeoutExpired("worker", 1))
    monkeypatch.setattr(runner.subprocess, "run", child)
    result = runner.execute(planned, tmp_path / "new", run_timeout_s=1)
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["reason"] == "client_process_timeout"
    assert result["status"] == "incomplete-measurements"


def test_interruption_keeps_partial_artifacts_and_never_deletes_them(tmp_path, monkeypatch):
    planned = schedule()
    monkeypatch.setattr(runner.subprocess, "run", MagicMock(side_effect=KeyboardInterrupt))
    output = tmp_path / "new"
    result = runner.execute(planned, output)
    assert result["interrupted"] and len(result["attempts"]) == 1
    assert (output / "plan.json").is_file() and (output / "summary.json").is_file()


def test_successful_pairs_run_in_fresh_processes_and_preserve_leaf_reports(tmp_path, monkeypatch):
    planned = schedule()
    before, after = paired()
    captured = []
    def child(argv, **kwargs):
        assert kwargs["cwd"] == runner.REPO_ROOT and "shell" not in kwargs
        captured.append(argv)
        enabled = "--sufficiency-check" in argv
        role = "negative" if "unanswerable" in argv[argv.index("--cases") + 1] else "positive"
        report = deepcopy(after if enabled else before)
        run = report["controlled_run"]
        run["inputs"] = planned["inputs"][role]
        run["cognition"]["model_llm_local"] = planned["model"]
        path = Path(argv[argv.index("--output") + 1])
        path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(argv, 0, stderr="")
    monkeypatch.setattr(runner.subprocess, "run", child)
    output = tmp_path / "new"
    result = runner.execute(planned, output)
    assert result["status"] == "completed-measurements" and len(captured) == 6
    assert len(result["pairs"]) == 3 and all(p["comparable"] for p in result["pairs"])
    assert result["semantic_acceptance"] == "pending-human-review"
    assert all("--output" in argv and argv[0] == runner.sys.executable for argv in captured)
