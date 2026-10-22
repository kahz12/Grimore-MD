from __future__ import annotations

import stat

import pytest
import yaml
from typer.testing import CliRunner

from grimore.cli import app
from grimore.pilot import (
    check_pilot_study,
    load_pilot_study,
    new_pilot_study,
    pilot_backup_path,
    record_pilot_observation,
    restore_pilot_backup,
    write_pilot_study,
)


def _ready_study(count: int = 5) -> dict:
    study = new_pilot_study(count, study_id="pilot-test")
    for participant in study["participants"]:
        participant["consent"].update({
            "recorded": True,
            "recorded_at": "2026-09-29T14:00:00Z",
        })
        for index, task in enumerate(participant["tasks"], start=1):
            task.update({
                "goal": f"Complete ordinary knowledge task {index}",
                "success_criteria": "The participant confirms the result is useful",
                "usual_method": "Search notes manually and compare the sources",
            })
    return study


def test_template_has_private_schema_and_exactly_three_tasks_per_person():
    study = new_pilot_study(5, study_id="pilot-a")

    assert len(study["participants"]) == 5
    assert all(len(item["tasks"]) == 3 for item in study["participants"])
    assert study["study"]["observation_days"] == 14
    assert set(study["study"]["privacy"].values()) == {False}
    serialized = yaml.safe_dump(study)
    for forbidden in ("\n  name:", "\n  email:", "vault_path:", "\n  question:"):
        assert forbidden not in serialized


def test_completed_intake_is_ready():
    result = check_pilot_study(_ready_study())

    assert result.ready is True
    assert result.participant_count == 5
    assert result.ready_participants == 5
    assert result.task_count == 15
    assert result.observed_stages == 0
    assert result.observed_journeys == 0
    assert result.completed_journeys == 0


def test_missing_consent_and_usual_method_are_blocking():
    study = _ready_study()
    study["participants"][0]["consent"]["recorded"] = False
    study["participants"][1]["tasks"][2]["usual_method"] = ""

    result = check_pilot_study(study)

    assert result.ready is False
    assert any("consent.recorded" in error for error in result.errors)
    assert any("usual_method is required" in error for error in result.errors)


def test_rejects_changed_thresholds_privacy_and_unknown_identity_fields():
    study = _ready_study()
    study["study"]["decision_thresholds"]["week_2_return_percent"] = 10
    study["study"]["privacy"]["store_document_content"] = True
    study["participants"][0]["email"] = "not-allowed@example.test"

    result = check_pilot_study(study)

    assert any("must remain 60" in error for error in result.errors)
    assert any("store_document_content must be false" in error for error in result.errors)
    assert any("unsupported fields: email" in error for error in result.errors)


def test_rejects_wrong_cohort_size_duplicate_ids_and_task_count():
    study = _ready_study()
    study["participants"][1]["participant_id"] = "P01"
    study["participants"][2]["tasks"].pop()
    study["participants"].pop()

    result = check_pilot_study(study)

    assert any("target cohort of 5" in error for error in result.errors)
    assert any("duplicate participant_id" in error for error in result.errors)
    assert any("exactly three tasks" in error for error in result.errors)


def test_rejects_name_like_participant_identifier():
    study = _ready_study()
    study["participants"][0]["participant_id"] = "Alice"

    result = check_pilot_study(study)

    assert any("opaque code such as P01" in error for error in result.errors)


def test_records_ordered_full_journey_with_recoverable_backup(tmp_path):
    target = tmp_path / "pilot.yaml"
    target.write_text(yaml.safe_dump(_ready_study(), sort_keys=False), encoding="utf-8")
    original = target.read_bytes()

    backup = record_pilot_observation(
        target,
        participant_id="P01",
        stage="configuration",
        outcome="completed",
        assistance="none",
        observed_at="2026-09-29T14:01:00Z",
    )

    assert backup == pilot_backup_path(target)
    assert backup.read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="complete preceding stage first"):
        record_pilot_observation(
            target,
            participant_id="P01",
            stage="source_check",
            outcome="completed",
            assistance="prompt",
        )
    with pytest.raises(ValueError, match="already observed"):
        record_pilot_observation(
            target,
            participant_id="P01",
            stage="configuration",
            outcome="completed",
            assistance="none",
        )

    for minute, stage in enumerate(
        ("first_answer", "source_check", "document_review"), start=2,
    ):
        record_pilot_observation(
            target,
            participant_id="P01",
            stage=stage,
            outcome="completed",
            assistance="prompt" if stage == "source_check" else "none",
            observed_at=f"2026-09-29T14:0{minute}:00Z",
        )

    result = check_pilot_study(load_pilot_study(target))
    assert result.observed_stages == 4
    assert result.observed_journeys == 1
    assert result.completed_journeys == 1


def test_restoring_backup_swaps_current_and_previous_versions(tmp_path):
    target = tmp_path / "pilot.yaml"
    target.write_text(yaml.safe_dump(_ready_study(), sort_keys=False), encoding="utf-8")
    record_pilot_observation(
        target,
        participant_id="P01",
        stage="configuration",
        outcome="blocked",
        assistance="intervention",
        observed_at="2026-09-29T14:01:00Z",
    )

    restore_pilot_backup(target)
    restored = load_pilot_study(target)
    assert restored["participants"][0]["journey"]["stages"]["configuration"] == {
        "outcome": "pending", "observed_at": None, "assistance": None,
    }

    restore_pilot_backup(target)
    reapplied = load_pilot_study(target)
    assert reapplied["participants"][0]["journey"]["stages"]["configuration"][
        "outcome"
    ] == "blocked"


def test_blocked_stage_stops_later_journey_steps(tmp_path):
    target = tmp_path / "pilot.yaml"
    target.write_text(yaml.safe_dump(_ready_study(), sort_keys=False), encoding="utf-8")
    record_pilot_observation(
        target,
        participant_id="P01",
        stage="configuration",
        outcome="blocked",
        assistance="intervention",
        observed_at="2026-09-29T14:01:00Z",
    )

    with pytest.raises(ValueError, match="complete preceding stage first"):
        record_pilot_observation(
            target,
            participant_id="P01",
            stage="first_answer",
            outcome="completed",
            assistance="none",
            observed_at="2026-09-29T14:02:00Z",
        )


def test_observe_upgrades_intake_only_record_and_rejects_private_notes(tmp_path):
    target = tmp_path / "pilot.yaml"
    study = _ready_study()
    for participant in study["participants"]:
        participant.pop("journey")
    target.write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")

    before = check_pilot_study(load_pilot_study(target))
    assert before.ready is True
    assert before.warnings
    record_pilot_observation(
        target,
        participant_id="P01",
        stage="configuration",
        outcome="completed",
        assistance="none",
        observed_at="2026-09-29T14:01:00Z",
    )
    assert load_pilot_study(target)["participants"][0]["journey"]

    study = load_pilot_study(target)
    study["participants"][0]["journey"]["stages"]["configuration"][
        "private_note"
    ] = "copied content"
    result = check_pilot_study(study)
    assert any("unsupported fields: private_note" in error for error in result.errors)


def test_write_is_private_atomic_and_refuses_overwrite(tmp_path):
    target = tmp_path / "pilot.yaml"

    write_pilot_study(target, participant_count=5, study_id="pilot-a")

    assert load_pilot_study(target)["study"]["study_id"] == "pilot-a"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".pilot.yaml.*"))
    with pytest.raises(FileExistsError):
        write_pilot_study(target)


def test_cli_initializes_and_checks_ready_study(tmp_path):
    target = tmp_path / "pilot.yaml"
    runner = CliRunner()
    created = runner.invoke(app, [
        "pilot", "init", str(target), "--participants", "5",
        "--study-id", "pilot-cli",
    ])
    assert created.exit_code == 0, created.output

    incomplete = runner.invoke(app, ["pilot", "check", str(target)])
    assert incomplete.exit_code == 1
    assert "Pilot is not ready" in incomplete.output

    target.write_text(
        yaml.safe_dump(_ready_study(), sort_keys=False), encoding="utf-8",
    )
    checked = runner.invoke(app, ["pilot", "check", str(target)])
    assert checked.exit_code == 0, checked.output
    assert "Pilot intake ready" in checked.output


def test_cli_records_observation_and_restores_backup(tmp_path):
    target = tmp_path / "pilot.yaml"
    target.write_text(
        yaml.safe_dump(_ready_study(), sort_keys=False), encoding="utf-8",
    )
    runner = CliRunner()

    observed = runner.invoke(app, [
        "pilot", "observe", str(target), "P01", "configuration",
        "--outcome", "completed", "--assistance", "none",
        "--at", "2026-09-29T14:01:00Z",
    ])
    assert observed.exit_code == 0, observed.output
    assert "Pilot observation saved" in observed.output

    restored = runner.invoke(app, ["pilot", "restore-backup", str(target)])
    assert restored.exit_code == 0, restored.output
    stage = load_pilot_study(target)["participants"][0]["journey"]["stages"][
        "configuration"
    ]
    assert stage["outcome"] == "pending"
