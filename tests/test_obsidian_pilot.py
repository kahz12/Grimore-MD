"""Offline Obsidian-pilot measurements never fabricate consent, usage or acceptance."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from integrations.obsidian import measure_pilot as pilot


AS_OF = datetime(2026, 10, 15, tzinfo=timezone.utc)


@pytest.fixture
def completed():
    data = pilot.template()
    for participant in data["participants"]:
        participant.update(
            consent_at="2026-09-30T09:00:00Z", tasks_agreed_at="2026-09-30T10:00:00Z",
            start_date="2026-10-01", daily_use=[True] * 14, maintenance_review="keep",
        )
        for task in participant["tasks"]:
            task["usual"] = {"steps": 8, "duration_seconds": 100, "outcome": "completed",
                             "assistance": "none", "observed_at": "2026-09-30T12:00:00Z"}
            task["obsidian"] = {"steps": 4, "duration_seconds": 70, "outcome": "completed",
                                "assistance": "none", "observed_at": "2026-10-01T12:00:00Z"}
    return data


def test_empty_template_preserves_pending_denominators():
    result = pilot.report(pilot.template(), as_of=AS_OF)
    assert result["schema_valid"] is True
    assert result["evidence_complete"] is False
    assert result["automatic_phase_closure"] is False
    assert result["acceptance"] == "pending_human_review"
    assert result["totals"]["expected_pairs"] == result["totals"]["missing_pairs"] == 15
    assert result["totals"]["median_steps_saved"] is None
    assert all(row["unknown_days"] == 14 for row in result["participants"])


def test_complete_measurements_never_automatically_accept(completed):
    before = deepcopy(completed)
    result = pilot.report(completed, as_of=AS_OF)
    assert result["evidence_complete"] is True
    assert result["acceptance"] == "pending_human_review"
    assert result["automatic_phase_closure"] is False
    assert result["totals"]["comparable_pairs"] == result["totals"]["fewer_steps"] == 15
    assert result["totals"]["median_steps_saved"] == 4
    assert result["totals"]["median_seconds_saved"] == 30
    assert result["totals"]["missing_pairs"] == 0
    assert result["participants"][0]["week_2_used_days"] == 7
    assert completed == before


def test_blocks_assistance_equal_and_worse_are_not_discarded(completed):
    tasks = completed["participants"][0]["tasks"]
    tasks[0]["obsidian"]["outcome"] = "blocked"
    tasks[1]["usual"]["assistance"] = "intervention"
    tasks[2]["obsidian"]["steps"] = 8
    other = completed["participants"][1]["tasks"][0]
    other["obsidian"]["steps"] = 10
    result = pilot.report(completed, as_of=AS_OF)["totals"]
    assert result["observed_pairs"] == 15
    assert result["comparable_pairs"] == 13
    assert result["excluded_pairs"] == 2
    assert result["obsidian_blocked"] == result["usual_assisted"] == 1
    assert result["equal_steps"] == result["more_steps"] == 1
    assert result["fewer_steps"] == 11


def test_null_days_not_confused_with_no_usage(completed):
    row = completed["participants"][0]
    row["daily_use"][7] = None
    row["daily_use"][8] = False
    result = pilot.report(completed, as_of=AS_OF)
    assert result["evidence_complete"] is False
    assert result["participants"][0]["week_2_used_days"] == 5
    assert result["participants"][0]["unknown_days"] == 1


def test_missing_trial_kept_in_total_denominator(completed):
    completed["participants"][0]["tasks"][0]["usual"] = None
    result = pilot.report(completed, as_of=AS_OF)
    assert not result["evidence_complete"]
    assert result["totals"]["expected_pairs"] == 15
    assert result["totals"]["missing_pairs"] == 1
    assert result["totals"]["comparable_pairs"] == 14
    assert result["totals"]["usual_observed"] == 14
    assert result["totals"]["obsidian_observed"] == 15
    assert "search.usual" in result["participants"][0]["pending"]


@pytest.mark.parametrize("count", [4, 11, True, 5.0, "5"])
def test_invalid_cohort_sizes(count):
    with pytest.raises(ValueError):
        pilot.template(count)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 1.0), ("integration", "other"),
    ("observation_days", 13), ("observation_days", 14.0), ("target_participants", True),
    ("participants", []), ("participants", {}),
])
def test_invalid_root_fields(completed, field, value):
    completed[field] = value
    with pytest.raises(ValueError):
        pilot.report(completed, as_of=AS_OF)


@pytest.mark.parametrize("field,value", [
    ("participant_id", "name@example.org"), ("consent_at", None),
    ("tasks_agreed_at", None), ("tasks_agreed_at", "2026-09-29T09:00:00Z"),
    ("consent_at", "2026-10-16T09:00:00Z"), ("consent_at", "2026-09-30T09:00:00"),
    ("start_date", None), ("start_date", "2026-09-30"),
    ("start_date", "2026-02-30"), ("start_date", "20261001"),
    ("start_date", "9999-12-31"),
    ("consent_at", "0001-01-01T00:00:00+23:59"),
    ("daily_use", [True] * 13), ("daily_use", [1] * 14),
    ("maintenance_review", "accepted"), ("tasks", []),
])
def test_invalid_participant_fields(completed, field, value):
    completed["participants"][0][field] = value
    with pytest.raises(ValueError):
        pilot.report(completed, as_of=AS_OF)


@pytest.mark.parametrize("field,value", [
    ("steps", True), ("steps", 0), ("steps", 10001), ("steps", 2.5),
    ("duration_seconds", 0), ("duration_seconds", 86401), ("duration_seconds", True),
    ("outcome", "pending"), ("assistance", "unknown"),
    ("observed_at", "2026-10-16T12:00:00Z"),
    ("observed_at", "2026-09-29T12:00:00Z"),
    ("observed_at", "2026-10-15T00:00:00Z"),
])
def test_invalid_trial_fields(completed, field, value):
    completed["participants"][0]["tasks"][0]["obsidian"][field] = value
    with pytest.raises(ValueError):
        pilot.report(completed, as_of=AS_OF)


@pytest.mark.parametrize("where", ["root", "participant", "task", "trial"])
def test_private_extra_fields_rejected_without_echoing_content(completed, where):
    task = completed["participants"][0]["tasks"][0]
    target = {"root": completed, "participant": completed["participants"][0],
              "task": task, "trial": task["obsidian"]}[where]
    target["secret-private-title"] = "private-query-canary"
    with pytest.raises(ValueError) as error:
        pilot.report(completed, as_of=AS_OF)
    assert "private" not in str(error.value)


def test_wrong_or_duplicate_task_ids_rejected(completed):
    completed["participants"][0]["tasks"][1]["task_id"] = "search"
    with pytest.raises(ValueError):
        pilot.report(completed, as_of=AS_OF)


def test_trial_order_and_observation_window(completed):
    task = completed["participants"][0]["tasks"][0]
    task["usual"]["observed_at"] = "2026-10-02T12:00:00Z"
    with pytest.raises(ValueError, match="precede"):
        pilot.report(completed, as_of=AS_OF)
    task["usual"]["observed_at"] = "2026-09-30T12:00:00Z"
    task["obsidian"]["observed_at"] = "2026-09-30T14:00:00Z"
    with pytest.raises(ValueError, match="window"):
        pilot.report(completed, as_of=AS_OF)


def test_trial_cannot_conflict_with_a_day_marked_unused(completed):
    completed["participants"][0]["daily_use"][0] = False
    with pytest.raises(ValueError, match="unused"):
        pilot.report(completed, as_of=AS_OF)


def test_assisted_success_is_observed_but_not_a_step_improvement(completed):
    for participant in completed["participants"]:
        for task in participant["tasks"]:
            task["obsidian"]["assistance"] = "prompt"
    result = pilot.report(completed, as_of=AS_OF)
    assert result["evidence_complete"]
    assert result["totals"]["obsidian_assisted"] == 15
    assert result["totals"]["comparable_pairs"] == 0
    assert result["totals"]["fewer_steps"] == 0
    assert result["totals"]["median_steps_saved"] is None


def test_unfinished_window_future_days_and_premature_review(completed):
    earlier = datetime(2026, 10, 7, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="future days"):
        pilot.report(completed, as_of=earlier)
    for participant in completed["participants"]:
        participant["daily_use"] = [True] * 7 + [None] * 7
    with pytest.raises(ValueError, match="review"):
        pilot.report(completed, as_of=earlier)
    for participant in completed["participants"]:
        participant["maintenance_review"] = "pending"
    result = pilot.report(completed, as_of=earlier)
    assert not result["evidence_complete"]
    assert all(not participant["window_elapsed"] for participant in result["participants"])


def test_timezone_normalization_and_no_naive_cutoff(completed):
    completed["participants"][0]["consent_at"] = "2026-09-30T04:00:00-05:00"
    assert pilot.report(completed, as_of=AS_OF)["evidence_complete"]
    with pytest.raises(ValueError, match="timezone"):
        pilot.report(completed, as_of=datetime(2026, 10, 15))


def test_template_is_exclusive_private_and_readonly_report(tmp_path):
    path = tmp_path / "record.json"
    pilot.write_template(path)
    assert path.stat().st_mode & 0o777 == 0o600
    before = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        pilot.write_template(path)
    pilot.report(pilot.read_record(path), as_of=AS_OF)
    assert sha256(path.read_bytes()).hexdigest() == before
    assert set(tmp_path.iterdir()) == {path}


def test_symlink_input_output_and_parent_never_followed(tmp_path):
    path = tmp_path / "record.json"
    pilot.write_template(path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        pilot.read_record(alias)
    with pytest.raises(FileExistsError):
        pilot.write_template(alias)
    parent = tmp_path / "parent"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        pilot.write_template(parent / "other.json")
    assert not (tmp_path / "other.json").exists()


@pytest.mark.parametrize("payload", [
    b"", b"\xff", b"{", b'{"x":NaN}', b'{"x":1,"x":2}',
    b"x" * (pilot.MAX_BYTES + 1),
])
def test_malformed_bounded_records(tmp_path, payload):
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        pilot.read_record(path)


def test_cli_structured_failures_and_partial_evidence(tmp_path, capsys):
    path = tmp_path / "record.json"
    assert pilot.main(["init", "--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["evidence_complete"] is False
    assert pilot.main(["report", "--input", str(path), "--as-of", "2026-10-15T00:00:00Z"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["missing_pairs"] == 15
    assert pilot.main(["init", "--output", str(path)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["ok"] is False
    assert str(path) not in output.err


def test_actual_cli_runs_without_grimore_models_or_database(tmp_path):
    script = Path(__file__).resolve().parents[1] / "integrations/obsidian/measure_pilot.py"
    path = tmp_path / "record.json"
    first = subprocess.run([sys.executable, str(script), "init", "--output", str(path)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0 and not first.stderr
    result = subprocess.run([sys.executable, str(script), "report", "--input", str(path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout)["automatic_phase_closure"] is False
    assert list(tmp_path.iterdir()) == [path]
