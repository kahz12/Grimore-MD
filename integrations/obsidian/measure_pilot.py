"""Offline, structured comparisons of recorded pilot trials. No vault access or telemetry.

This repository helper is separate from the plugin and from the pilot intake
and observation tool (``grimore pilot``). Humans agree tasks and record
observations; this tool never certifies consent, real usage, usefulness or
acceptance from self-reported counts.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import sys


MAX_BYTES = 256 * 1024
TASK_IDS = ("search", "source", "review")
MODES = ("usual", "obsidian")
UTC = timezone.utc


def template(participant_count: int = 5) -> dict:
    if type(participant_count) is not int or not 5 <= participant_count <= 10:
        raise ValueError("Participant count must be an integer between 5 and 10")
    return {
        "schema_version": 1,
        "integration": "obsidian-readonly",
        "observation_days": 14,
        "target_participants": participant_count,
        "participants": [
            {
                "participant_id": f"P{index:02d}",
                "consent_at": None,
                "tasks_agreed_at": None,
                "start_date": None,
                "daily_use": [None] * 14,
                "maintenance_review": "pending",
                "tasks": [
                    {"task_id": task, "usual": None, "obsidian": None}
                    for task in TASK_IDS
                ],
            }
            for index in range(1, participant_count + 1)
        ],
    }


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        # The message stays generic on purpose: unexpected field names or
        # values could contain private content, so they are not echoed.
        raise ValueError(f"{label}: missing or unsupported fields")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}: expected an integer between {low} and {high}")


def _timestamp(value, label):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError(f"{label}: expected an ISO-8601 timestamp with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}: invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label}: timezone is required")
    try:
        return result.astimezone(UTC)
    except OverflowError as exc:
        raise ValueError(f"{label}: timestamp is outside the supported range") from exc


def _start(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("start_date: expected YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("start_date: invalid calendar date") from exc
    if parsed.isoformat() != value or parsed > date(9999, 12, 17):
        raise ValueError("start_date: expected YYYY-MM-DD")
    return datetime.combine(parsed, datetime.min.time(), UTC)


def _trial(value, label):
    _fields(value, ("steps", "duration_seconds", "outcome", "assistance", "observed_at"), label)
    _integer(value["steps"], 1, 10000, label + ".steps")
    _integer(value["duration_seconds"], 1, 86400, label + ".duration_seconds")
    if value["outcome"] not in ("completed", "blocked"):
        raise ValueError(f"{label}: unsupported outcome")
    if value["assistance"] not in ("none", "prompt", "intervention"):
        raise ValueError(f"{label}: unsupported assistance")
    return _timestamp(value["observed_at"], label + ".observed_at")


def report(data: dict, *, as_of: datetime | None = None) -> dict:
    """Strictly validate before summarizing, preserving pending denominators.

    Both trials must complete without help to enter the step/time comparison.
    Failed or assisted trials remain visible; they cannot improve the result.
    """
    now = as_of if as_of is not None else datetime.now(UTC)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("as_of: timezone is required")
    now = now.astimezone(UTC)
    _fields(data, ("schema_version", "integration", "observation_days",
                   "target_participants", "participants"), "study")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["integration"] != "obsidian-readonly"
            or type(data["observation_days"]) is not int or data["observation_days"] != 14):
        raise ValueError("Unsupported schema, integration or observation window")
    count = data["target_participants"]
    _integer(count, 5, 10, "target_participants")
    participants = data["participants"]
    if not isinstance(participants, list) or len(participants) != count:
        raise ValueError("participants: the declared cohort must be preserved")
    results = []
    expected_pairs = count * len(TASK_IDS)
    totals = {
        "expected_pairs": expected_pairs, "observed_pairs": 0,
        "usual_observed": 0, "obsidian_observed": 0,
        "comparable_pairs": 0, "fewer_steps": 0, "equal_steps": 0,
        "more_steps": 0, "usual_blocked": 0, "obsidian_blocked": 0,
        "usual_assisted": 0, "obsidian_assisted": 0,
    }
    step_deltas = []
    time_deltas = []
    for index, participant in enumerate(participants, 1):
        label = f"participant[{index}]"
        _fields(participant, (
            "participant_id", "consent_at", "tasks_agreed_at", "start_date",
            "daily_use", "maintenance_review", "tasks",
        ), label)
        if participant["participant_id"] != f"P{index:02d}":
            raise ValueError("participants: expected ordered opaque IDs P01, P02, ...")
        pending = []
        timestamps = {}
        for field in ("consent_at", "tasks_agreed_at"):
            raw = participant[field]
            if raw is None:
                pending.append(field)
            else:
                timestamps[field] = _timestamp(raw, field)
                if timestamps[field] > now:
                    raise ValueError(f"{field}: future records are not allowed")
        if (len(timestamps) == 2
                and timestamps["tasks_agreed_at"] < timestamps["consent_at"]):
            raise ValueError("Tasks must be agreed after consent")
        start = _start(participant["start_date"]) if participant["start_date"] is not None else None
        if start is None:
            pending.append("start_date")
        elif (len(timestamps) != 2 or start < timestamps["tasks_agreed_at"]):
            raise ValueError("The observation window must start after consent and task agreement")
        end = start + timedelta(days=14) if start else None
        elapsed = end is not None and now >= end
        if not elapsed:
            pending.append("14_day_window")
        daily = participant["daily_use"]
        if (not isinstance(daily, list) or len(daily) != 14
                or any(value is not None and type(value) is not bool for value in daily)):
            raise ValueError("daily_use: expected exactly 14 boolean or null values")
        for day, used in enumerate(daily):
            if used is not None and (start is None or (start + timedelta(days=day)).date() > now.date()):
                raise ValueError("daily_use: no observations without a start or for future days")
        if any(value is None for value in daily):
            pending.append("daily_use")
        maintenance = participant["maintenance_review"]
        if maintenance not in ("pending", "keep", "revise", "stop"):
            raise ValueError("maintenance_review: unsupported decision")
        if maintenance == "pending":
            pending.append("maintenance_review")
        elif not elapsed:
            raise ValueError("Maintenance review must follow the two-week window")
        tasks = participant["tasks"]
        if not isinstance(tasks, list) or len(tasks) != 3:
            raise ValueError("tasks: exactly three pre-agreed categories are required")
        pair_count = comparable = improved = 0
        for task, expected_task in zip(tasks, TASK_IDS, strict=True):
            _fields(task, ("task_id", *MODES), "task")
            if task["task_id"] != expected_task:
                raise ValueError("tasks: expected search, source, review in order")
            observed = {}
            for mode in MODES:
                trial = task[mode]
                if trial is None:
                    pending.append(f"{expected_task}.{mode}")
                    continue
                when = _trial(trial, mode)
                if len(timestamps) != 2 or start is None:
                    raise ValueError("Trials require consent, pre-agreed tasks and a start date")
                if when < timestamps["tasks_agreed_at"] or when > now:
                    raise ValueError("Trials must follow task agreement and cannot be in the future")
                if mode == "obsidian" and not start <= when < end:
                    raise ValueError("Obsidian trials must fall inside the two-week window")
                if mode == "obsidian" and daily[(when.date() - start.date()).days] is False:
                    raise ValueError("An Obsidian trial conflicts with a day recorded as unused")
                observed[mode] = when
                totals[f"{mode}_observed"] += 1
                totals[f"{mode}_blocked"] += trial["outcome"] == "blocked"
                totals[f"{mode}_assisted"] += trial["assistance"] != "none"
            if len(observed) != 2:
                continue
            if observed["usual"] > observed["obsidian"]:
                raise ValueError("The usual-flow trial must precede the Obsidian trial")
            pair_count += 1
            if any(task[mode]["outcome"] != "completed" or task[mode]["assistance"] != "none"
                   for mode in MODES):
                continue
            comparable += 1
            delta = task["usual"]["steps"] - task["obsidian"]["steps"]
            improved += delta > 0
            totals["fewer_steps" if delta > 0 else "more_steps" if delta < 0 else "equal_steps"] += 1
            step_deltas.append(delta)
            time_deltas.append(task["usual"]["duration_seconds"] - task["obsidian"]["duration_seconds"])
        totals["observed_pairs"] += pair_count
        totals["comparable_pairs"] += comparable
        results.append({
            "participant_id": participant["participant_id"],
            "window_elapsed": elapsed, "pending": pending,
            "evidence_complete": not pending,
            "observed_pairs": pair_count, "comparable_pairs": comparable,
            "fewer_steps": improved,
            "week_1_used_days": sum(value is True for value in daily[:7]),
            "week_2_used_days": sum(value is True for value in daily[7:]),
            "unknown_days": sum(value is None for value in daily),
            "maintenance_review": maintenance,
        })
    return {
        "schema_valid": True, "integration": "obsidian-readonly",
        "as_of": now.isoformat(), "evidence_complete": all(row["evidence_complete"] for row in results),
        "acceptance": "pending_human_review", "automatic_phase_closure": False,
        "totals": {**totals, "missing_pairs": expected_pairs - totals["observed_pairs"],
                   "excluded_pairs": totals["observed_pairs"] - totals["comparable_pairs"],
                   "median_steps_saved": statistics.median(step_deltas) if step_deltas else None,
                   "median_seconds_saved": statistics.median(time_deltas) if time_deltas else None},
        "participants": results,
    }


def _json(data: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON fields are not allowed")
            result[key] = value
        return result

    def finite(_value):
        raise ValueError("Non-finite JSON values are not allowed")

    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=finite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON record") from exc


def read_record(path: Path) -> dict:
    path = path.absolute()
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError("The record must be a regular file without symlink aliases")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise ValueError("The record is empty or exceeds the size limit")
    return _json(data)


def write_template(path: Path, participant_count: int = 5) -> None:
    data = template(participant_count)
    path = path.absolute()
    if path.parent != path.parent.resolve():
        raise ValueError("The output parent must not have symlink aliases")
    payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    # O_EXCL fails if the path exists in any form (a symlink, an empty file,
    # ...), so an existing output is never overwritten.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an exclusive blank record, never an observation")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--participants", type=int, default=5)
    summarize = sub.add_parser("report", help="Validate and summarize without writes or models")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--as-of", help="Explicit timezone-aware cutoff, recorded in the report")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            write_template(args.output, args.participants)
            result = {"created": True, "evidence_complete": False, "acceptance": "pending_human_review"}
        else:
            result = report(read_record(args.input), as_of=_timestamp(args.as_of, "as_of") if args.as_of else None)
    except OSError:
        print(json.dumps({"ok": False, "error": "Cannot read or create the selected file"}), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
