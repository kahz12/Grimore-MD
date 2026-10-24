"""Privacy-preserving study records for Grimore's individual-user pilot.

The pilot intake file stores pseudonymous participation metadata and task
definitions, never vault paths, document excerpts, questions, answers, names,
or contact details. Validation is intentionally strict so a typo cannot
silently remove one of the three agreed tasks from the study denominator.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from grimore.utils.atomic import atomic_write

PILOT_VERSION = 1
MIN_PARTICIPANTS = 5
MAX_PARTICIPANTS = 10
TASKS_PER_PARTICIPANT = 3
MAX_STUDY_BYTES = 512 * 1024
JOURNEY_STAGES = (
    "configuration", "first_answer", "source_check", "document_review",
)
JOURNEY_OUTCOMES = {"pending", "completed", "blocked"}
ASSISTANCE_LEVELS = {"none", "prompt", "intervention"}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_PARTICIPANT_ID_RE = re.compile(r"^[A-Z]{1,3}[0-9]{2,4}$")
_ROOT_KEYS = {"version", "study", "participants"}
_STUDY_KEYS = {
    "study_id", "created_at", "observation_days", "target_participants",
    "privacy", "decision_thresholds",
}
_PRIVACY_KEYS = {
    "store_personal_identifiers", "store_document_content",
    "store_questions_or_answers",
}
_THRESHOLD_KEYS = {
    "journey_without_help_percent", "two_tasks_resolved_percent",
    "week_2_return_percent", "useful_alerts_percent",
}
_PARTICIPANT_KEYS = {"participant_id", "consent", "tasks", "journey"}
_CONSENT_KEYS = {"recorded", "recorded_at", "scope"}
_TASK_KEYS = {"task_id", "goal", "success_criteria", "usual_method"}
_JOURNEY_KEYS = {"stages"}
_STAGE_KEYS = {"outcome", "observed_at", "assistance"}


@dataclass(frozen=True)
class PilotCheck:
    """Validation result plus denominators useful during recruitment."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    participant_count: int
    ready_participants: int
    task_count: int
    observed_stages: int
    observed_journeys: int
    completed_journeys: int

    @property
    def ready(self) -> bool:
        return not self.errors


def new_pilot_study(
    participant_count: int = MIN_PARTICIPANTS,
    *,
    study_id: str | None = None,
) -> dict[str, Any]:
    """Return a blank two-week pilot record with three tasks per participant."""
    if not MIN_PARTICIPANTS <= int(participant_count) <= MAX_PARTICIPANTS:
        raise ValueError(
            f"participant count must be between {MIN_PARTICIPANTS} and "
            f"{MAX_PARTICIPANTS}"
        )
    identifier = study_id or time.strftime("grimore-pilot-%Y%m%d", time.gmtime())
    if not _ID_RE.fullmatch(identifier):
        raise ValueError("study id must contain only letters, numbers, '_' or '-'")
    participants = []
    for index in range(1, participant_count + 1):
        participants.append({
            "participant_id": f"P{index:02d}",
            "consent": {
                "recorded": False,
                "recorded_at": None,
                "scope": "task definitions and structured outcomes only",
            },
            "tasks": [
                {
                    "task_id": f"T{task_index}",
                    "goal": "",
                    "success_criteria": "",
                    "usual_method": "",
                }
                for task_index in range(1, TASKS_PER_PARTICIPANT + 1)
            ],
            "journey": {
                "stages": {
                    stage: {
                        "outcome": "pending",
                        "observed_at": None,
                        "assistance": None,
                    }
                    for stage in JOURNEY_STAGES
                },
            },
        })
    return {
        "version": PILOT_VERSION,
        "study": {
            "study_id": identifier,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "observation_days": 14,
            "target_participants": participant_count,
            "privacy": {
                "store_personal_identifiers": False,
                "store_document_content": False,
                "store_questions_or_answers": False,
            },
            "decision_thresholds": {
                "journey_without_help_percent": 80,
                "two_tasks_resolved_percent": 70,
                "week_2_return_percent": 60,
                "useful_alerts_percent": 70,
            },
        },
        "participants": participants,
    }


def write_pilot_study(
    path: str | Path,
    *,
    participant_count: int = MIN_PARTICIPANTS,
    study_id: str | None = None,
) -> Path:
    """Create a new pilot YAML atomically without overwriting existing data."""
    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing pilot file: {target}")
    payload = yaml.safe_dump(
        new_pilot_study(participant_count, study_id=study_id),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    ).encode("utf-8")
    atomic_write(target, lambda fh: fh.write(payload), mode="wb")
    try:
        target.chmod(0o600)
    except (OSError, NotImplementedError):
        pass
    return target


def load_pilot_study(path: str | Path) -> dict[str, Any]:
    """Read a bounded YAML study file using PyYAML's safe loader."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"pilot file not found: {source}")
    size = source.stat().st_size
    if size > MAX_STUDY_BYTES:
        raise ValueError(f"pilot file exceeds {MAX_STUDY_BYTES} bytes")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid pilot YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("pilot file must contain a YAML mapping")
    return data


def check_pilot_study(data: dict[str, Any]) -> PilotCheck:
    """Validate schema, privacy invariants, consent, and task readiness."""
    errors: list[str] = []
    warnings: list[str] = []
    _reject_unknown(data, _ROOT_KEYS, "root", errors)
    if data.get("version") != PILOT_VERSION:
        errors.append(f"version must be {PILOT_VERSION}")

    study = _as_mapping(data.get("study"), "study", errors)
    _reject_unknown(study, _STUDY_KEYS, "study", errors)
    study_id = study.get("study_id")
    if not isinstance(study_id, str) or not _ID_RE.fullmatch(study_id):
        errors.append("study.study_id must be a pseudonymous identifier")
    if not _valid_timestamp(study.get("created_at")):
        errors.append("study.created_at must be an ISO-8601 timestamp")
    if study.get("observation_days") != 14:
        errors.append("study.observation_days must be 14")
    target = study.get("target_participants")
    if not isinstance(target, int) or isinstance(target, bool) or not (
        MIN_PARTICIPANTS <= target <= MAX_PARTICIPANTS
    ):
        errors.append("study.target_participants must be between 5 and 10")

    privacy = _as_mapping(study.get("privacy"), "study.privacy", errors)
    _reject_unknown(privacy, _PRIVACY_KEYS, "study.privacy", errors)
    for key in sorted(_PRIVACY_KEYS):
        if privacy.get(key) is not False:
            errors.append(f"study.privacy.{key} must be false")

    thresholds = _as_mapping(
        study.get("decision_thresholds"), "study.decision_thresholds", errors,
    )
    _reject_unknown(
        thresholds, _THRESHOLD_KEYS, "study.decision_thresholds", errors,
    )
    expected_thresholds = {
        "journey_without_help_percent": 80,
        "two_tasks_resolved_percent": 70,
        "week_2_return_percent": 60,
        "useful_alerts_percent": 70,
    }
    for key, expected in expected_thresholds.items():
        if thresholds.get(key) != expected:
            errors.append(
                f"study.decision_thresholds.{key} must remain {expected} "
                "during the pilot"
            )

    raw_participants = data.get("participants")
    if not isinstance(raw_participants, list):
        errors.append("participants must be a list")
        raw_participants = []
    if isinstance(target, int) and len(raw_participants) != target:
        errors.append(
            f"participants must contain the target cohort of {target} records"
        )
    if not MIN_PARTICIPANTS <= len(raw_participants) <= MAX_PARTICIPANTS:
        errors.append("participants must contain between 5 and 10 records")

    seen_participants: set[str] = set()
    ready_participants = 0
    task_count = 0
    observed_stages = 0
    observed_journeys = 0
    completed_journeys = 0
    missing_journeys = 0
    for index, raw in enumerate(raw_participants, start=1):
        label = f"participants[{index}]"
        participant = _as_mapping(raw, label, errors)
        _reject_unknown(participant, _PARTICIPANT_KEYS, label, errors)
        participant_id = participant.get("participant_id")
        if (
            not isinstance(participant_id, str)
            or not _PARTICIPANT_ID_RE.fullmatch(participant_id)
        ):
            errors.append(
                f"{label}.participant_id must be an opaque code such as P01"
            )
            participant_id = label
        elif participant_id in seen_participants:
            errors.append(f"duplicate participant_id: {participant_id}")
        seen_participants.add(str(participant_id))

        consent = _as_mapping(participant.get("consent"), f"{label}.consent", errors)
        _reject_unknown(consent, _CONSENT_KEYS, f"{label}.consent", errors)
        consent_ready = consent.get("recorded") is True
        if not consent_ready:
            errors.append(f"{label}.consent.recorded must be true before observation")
        if consent_ready and not _valid_timestamp(consent.get("recorded_at")):
            errors.append(f"{label}.consent.recorded_at must be an ISO-8601 timestamp")
        scope = consent.get("scope")
        if not isinstance(scope, str) or not scope.strip():
            errors.append(f"{label}.consent.scope is required")

        tasks = participant.get("tasks")
        if not isinstance(tasks, list):
            errors.append(f"{label}.tasks must be a list")
            tasks = []
        task_count += len(tasks)
        if len(tasks) != TASKS_PER_PARTICIPANT:
            errors.append(f"{label}.tasks must contain exactly three tasks")
        seen_tasks: set[str] = set()
        participant_tasks_ready = len(tasks) == TASKS_PER_PARTICIPANT
        for task_index, raw_task in enumerate(tasks, start=1):
            task_label = f"{label}.tasks[{task_index}]"
            task = _as_mapping(raw_task, task_label, errors)
            _reject_unknown(task, _TASK_KEYS, task_label, errors)
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or not _ID_RE.fullmatch(task_id):
                errors.append(f"{task_label}.task_id must be an identifier")
                participant_tasks_ready = False
            elif task_id in seen_tasks:
                errors.append(f"{label} has duplicate task_id: {task_id}")
                participant_tasks_ready = False
            seen_tasks.add(str(task_id))
            for field in ("goal", "success_criteria", "usual_method"):
                value = task.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{task_label}.{field} is required")
                    participant_tasks_ready = False
                elif len(value) > 500:
                    errors.append(f"{task_label}.{field} exceeds 500 characters")
                    participant_tasks_ready = False
        if consent_ready and participant_tasks_ready:
            ready_participants += 1

        journey = participant.get("journey")
        if journey is None:
            # Intake-only participants have no ``journey`` block yet; it is
            # optional here and counted as missing. The first `pilot observe`
            # command materialises the structure.
            missing_journeys += 1
            continue
        journey_map = _as_mapping(journey, f"{label}.journey", errors)
        _reject_unknown(journey_map, _JOURNEY_KEYS, f"{label}.journey", errors)
        stages = _as_mapping(
            journey_map.get("stages"), f"{label}.journey.stages", errors,
        )
        unknown_stages = sorted(set(stages) - set(JOURNEY_STAGES))
        missing_stages = sorted(set(JOURNEY_STAGES) - set(stages))
        if unknown_stages:
            errors.append(
                f"{label}.journey.stages contains unsupported stages: "
                + ", ".join(unknown_stages)
            )
        if missing_stages:
            errors.append(
                f"{label}.journey.stages is missing: " + ", ".join(missing_stages)
            )
        outcomes: list[str] = []
        previous_timestamp: datetime | None = None
        journey_can_continue = True
        for stage in JOURNEY_STAGES:
            if stage not in stages:
                continue
            stage_label = f"{label}.journey.stages.{stage}"
            record = _as_mapping(stages.get(stage), stage_label, errors)
            _reject_unknown(record, _STAGE_KEYS, stage_label, errors)
            outcome = record.get("outcome")
            if outcome not in JOURNEY_OUTCOMES:
                errors.append(
                    f"{stage_label}.outcome must be pending, completed, or blocked"
                )
                outcomes.append("invalid")
                continue
            outcomes.append(str(outcome))
            if outcome == "pending":
                journey_can_continue = False
                if record.get("observed_at") is not None:
                    errors.append(f"{stage_label}.observed_at must be null while pending")
                if record.get("assistance") is not None:
                    errors.append(f"{stage_label}.assistance must be null while pending")
                continue
            observed_stages += 1
            if not journey_can_continue:
                errors.append(
                    f"{stage_label} cannot be observed unless preceding stages completed"
                )
            observed_at = _parse_timestamp(record.get("observed_at"))
            if observed_at is None:
                errors.append(f"{stage_label}.observed_at must be an ISO-8601 timestamp")
            elif previous_timestamp is not None and observed_at < previous_timestamp:
                errors.append(f"{stage_label}.observed_at is earlier than the prior stage")
            else:
                previous_timestamp = observed_at
            if record.get("assistance") not in ASSISTANCE_LEVELS:
                errors.append(
                    f"{stage_label}.assistance must be none, prompt, or intervention"
                )
            if outcome == "blocked":
                journey_can_continue = False
        if len(outcomes) == len(JOURNEY_STAGES) and all(
            outcome != "pending" for outcome in outcomes
        ):
            observed_journeys += 1
        if outcomes == ["completed"] * len(JOURNEY_STAGES):
            completed_journeys += 1

    if not raw_participants:
        warnings.append("no participant records were found")
    if missing_journeys:
        warnings.append(
            f"{missing_journeys} participant record(s) use the intake-only schema; "
            "pilot observe will add journey stages"
        )
    return PilotCheck(
        errors=tuple(errors),
        warnings=tuple(warnings),
        participant_count=len(raw_participants),
        ready_participants=ready_participants,
        task_count=task_count,
        observed_stages=observed_stages,
        observed_journeys=observed_journeys,
        completed_journeys=completed_journeys,
    )


def record_pilot_observation(
    path: str | Path,
    *,
    participant_id: str,
    stage: str,
    outcome: str,
    assistance: str,
    observed_at: str | None = None,
    replace: bool = False,
) -> Path:
    """Record one ordered journey stage and save a private pre-change backup."""
    if stage not in JOURNEY_STAGES:
        raise ValueError("stage must be one of: " + ", ".join(JOURNEY_STAGES))
    if outcome not in JOURNEY_OUTCOMES - {"pending"}:
        raise ValueError("outcome must be completed or blocked")
    if assistance not in ASSISTANCE_LEVELS:
        raise ValueError("assistance must be none, prompt, or intervention")
    timestamp = observed_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if _parse_timestamp(timestamp) is None:
        raise ValueError("observed_at must be an ISO-8601 timestamp")

    target = Path(path).expanduser()
    if target.is_symlink():
        raise ValueError("pilot file may not be a symbolic link")
    data = load_pilot_study(target)
    intake = check_pilot_study(data)
    if intake.errors:
        raise ValueError("pilot intake is not ready: " + intake.errors[0])
    participant = next(
        (
            item for item in data.get("participants", [])
            if item.get("participant_id") == participant_id
        ),
        None,
    )
    if participant is None:
        raise LookupError(f"participant not found: {participant_id}")
    if "journey" not in participant:
        participant["journey"] = _new_journey()
    stages = participant["journey"]["stages"]
    # Intake-only files may omit the journey entirely. A malformed partial
    # journey is not silently repaired because that could hide a typo.
    if set(stages) != set(JOURNEY_STAGES):
        raise ValueError("participant journey does not contain the four expected stages")
    record = stages[stage]
    if record.get("outcome") != "pending" and not replace:
        raise ValueError("stage already observed; pass --replace to correct it")
    stage_index = JOURNEY_STAGES.index(stage)
    for prior in JOURNEY_STAGES[:stage_index]:
        if stages[prior].get("outcome") != "completed":
            raise ValueError(f"complete preceding stage first: {prior}")
    if stage_index:
        prior_time = _parse_timestamp(stages[JOURNEY_STAGES[stage_index - 1]].get("observed_at"))
        current_time = _parse_timestamp(timestamp)
        if prior_time is not None and current_time is not None and current_time < prior_time:
            raise ValueError("observed_at cannot be earlier than the preceding stage")
    record.update({
        "outcome": outcome,
        "observed_at": timestamp,
        "assistance": assistance,
    })
    updated = check_pilot_study(data)
    if updated.errors:
        raise ValueError("updated pilot record is invalid: " + updated.errors[0])
    return _write_pilot_update(target, data)


def restore_pilot_backup(path: str | Path) -> Path:
    """Swap the current pilot file with its last private pre-change backup."""
    target = Path(path).expanduser()
    backup = pilot_backup_path(target)
    if target.is_symlink() or backup.is_symlink():
        raise ValueError("pilot files may not be symbolic links")
    current = target.read_bytes()
    previous = backup.read_bytes()
    if len(previous) > MAX_STUDY_BYTES:
        raise ValueError("pilot backup is too large")
    try:
        restored = yaml.safe_load(previous.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("pilot backup is unreadable") from exc
    if not isinstance(restored, dict) or restored.get("version") != PILOT_VERSION:
        raise ValueError("pilot backup has an unsupported schema")
    restored_check = check_pilot_study(restored)
    if restored_check.errors:
        raise ValueError("pilot backup is invalid: " + restored_check.errors[0])
    atomic_write(target, lambda fh: fh.write(previous), mode="wb")
    atomic_write(backup, lambda fh: fh.write(current), mode="wb")
    _make_private(target)
    _make_private(backup)
    return target


def pilot_backup_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    return target.with_name(target.name + ".bak")


def _new_journey() -> dict[str, Any]:
    return {
        "stages": {
            stage: {
                "outcome": "pending",
                "observed_at": None,
                "assistance": None,
            }
            for stage in JOURNEY_STAGES
        },
    }


def _write_pilot_update(target: Path, data: dict[str, Any]) -> Path:
    before = target.read_bytes()
    backup = pilot_backup_path(target)
    payload = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, width=100,
    ).encode("utf-8")
    atomic_write(backup, lambda fh: fh.write(before), mode="wb")
    _make_private(backup)
    atomic_write(target, lambda fh: fh.write(payload), mode="wb")
    _make_private(target)
    return backup


def _make_private(path: Path) -> None:
    try:
        path.chmod(0o600)
    except (OSError, NotImplementedError):
        pass


def _as_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    errors.append(f"{label} must be a mapping")
    return {}


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], label: str, errors: list[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _valid_timestamp(value: Any) -> bool:
    return _parse_timestamp(value) is not None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def is_private_file(path: str | Path) -> bool:
    """Return whether POSIX group/other permission bits are absent."""
    try:
        return not (Path(path).stat().st_mode & 0o077)
    except (OSError, NotImplementedError):
        return os.name == "nt"
