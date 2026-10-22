"""Durable, bounded progress snapshots for CLI and daemon indexing."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from grimore.utils.atomic import atomic_write
from grimore.utils.paths import index_progress_path

_MAX_ITEMS = 200
_OUTCOMES = {"processed", "skipped", "failed", "previewed", "removed"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


class IndexProgressReporter:
    """Atomically publish one producer's current indexing state."""

    def __init__(self, vault_root: Path | str, source: str) -> None:
        self._disabled = False
        try:
            self.path: Path | None = index_progress_path(vault_root, source)
        except OSError:
            # Progress reporting is observability, not part of the indexing
            # transaction. An unavailable cache must never stop a scan.
            self.path = None
            self._disabled = True
        self.source = source
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None

    def start(
        self,
        *,
        total: int | None,
        dry_run: bool,
        change_set_id: str | None = None,
    ) -> None:
        now = _now()
        self._state = {
            "version": 1,
            "run_id": uuid.uuid4().hex,
            "source": self.source,
            "status": "running",
            "dry_run": bool(dry_run),
            "change_set_id": change_set_id,
            "pid": os.getpid(),
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "total": max(0, int(total)) if total is not None else None,
            "current_path": None,
            "counts": {
                "completed": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "previewed": 0,
                "removed": 0,
                "chunks": 0,
            },
            "items": [],
        }
        self._publish()

    def begin_item(self, path: str) -> None:
        with self._lock:
            if self._state is None:
                return
            self._state["current_path"] = str(path)
            self._state["updated_at"] = _now()
            self._publish_locked()

    def record(
        self,
        outcome: str,
        path: str,
        *,
        reason: str | None = None,
        error: str | None = None,
        chunks: int = 0,
        changes: dict | None = None,
    ) -> None:
        if outcome not in _OUTCOMES:
            raise ValueError(f"unsupported indexing outcome: {outcome}")
        with self._lock:
            if self._state is None:
                return
            counts = self._state["counts"]
            counts["completed"] += 1
            counts[outcome] += 1
            counts["chunks"] += max(0, int(chunks or 0))
            item: dict[str, Any] = {
                "ts": _now(),
                "path": str(path),
                "outcome": outcome,
            }
            if reason:
                item["reason"] = str(reason)[:200]
            if error:
                item["error"] = str(error)[:500]
            if chunks:
                item["chunks"] = max(0, int(chunks))
            if changes:
                item["changes"] = self._bounded_changes(changes)
            self._state["items"].append(item)
            self._state["items"] = self._state["items"][-_MAX_ITEMS:]
            self._state["current_path"] = None
            self._state["updated_at"] = item["ts"]
            self._publish_locked()

    @staticmethod
    def _bounded_changes(changes: dict) -> dict:
        """Keep previews informative without turning progress into a data dump."""
        fields = []
        for raw in list(changes.get("fields") or [])[:12]:
            if not isinstance(raw, dict):
                continue

            def compact(value):
                if isinstance(value, str):
                    return value[:300] + ("…" if len(value) > 300 else "")
                if isinstance(value, list):
                    return [compact(item) for item in value[:20]]
                if value is None or isinstance(value, (bool, int, float)):
                    return value
                return str(value)[:300]

            fields.append({
                "name": str(raw.get("name") or "")[:100],
                "before": compact(raw.get("before")),
                "after": compact(raw.get("after")),
            })
        return {
            "target": str(changes.get("target") or "")[:500],
            "action": (
                "create" if changes.get("action") == "create" else "update"
            ),
            "fields": fields,
        }

    def finish(self, *, status: str | None = None) -> None:
        with self._lock:
            if self._state is None:
                return
            if status is None:
                status = (
                    "completed_with_errors"
                    if self._state["counts"]["failed"] else "completed"
                )
            now = _now()
            self._state["status"] = status
            self._state["current_path"] = None
            self._state["updated_at"] = now
            self._state["finished_at"] = now
            self._publish_locked()

    def _publish(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        if self._state is None or self._disabled or self.path is None:
            return
        payload = json.dumps(
            self._state, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        try:
            atomic_write(self.path, lambda fh: fh.write(payload), mode="wb")
        except OSError:
            self._disabled = True


def read_index_progress(vault_root: Path | str) -> list[dict[str, Any]]:
    """Read both producer snapshots, marking abandoned runs interrupted."""
    runs: list[dict[str, Any]] = []
    for source in ("scan", "daemon"):
        try:
            path = index_progress_path(vault_root, source)
        except OSError:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("version") != 1:
            continue
        if data.get("status") == "running" and not _pid_alive(data.get("pid")):
            data["status"] = "interrupted"
            data["current_path"] = None
        runs.append(data)
    return sorted(runs, key=lambda run: str(run.get("updated_at") or ""), reverse=True)


def recovery_actions(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build concise, deduplicated recovery actions from current snapshots."""
    actions: list[dict[str, str]] = []
    if not runs:
        return [{
            "id": "start-preview",
            "title": "No indexing run recorded",
            "detail": "Run a safe preview first to validate the pipeline.",
            "command": "grimore scan --dry-run",
        }]
    if any(run.get("status") == "interrupted" for run in runs):
        actions.append({
            "id": "resume-interrupted",
            "title": "An indexing process stopped unexpectedly",
            "detail": "Fix the reported cause, then rerun; completed files fast-skip.",
            "command": "grimore scan --no-dry-run",
        })
    if any(int((run.get("counts") or {}).get("failed", 0)) > 0 for run in runs):
        actions.append({
            "id": "retry-failed",
            "title": "Some files failed",
            "detail": "Review the failed rows below. A rerun retries incomplete files.",
            "command": "grimore scan --no-dry-run",
        })
    privacy_skips = any(
        item.get("reason") == "privacy"
        for run in runs for item in (run.get("items") or [])
    )
    if privacy_skips:
        actions.append({
            "id": "review-privacy",
            "title": "Some files were excluded by privacy policy",
            "detail": "Keep them excluded, or remove privacy: never_process intentionally.",
            "command": "",
        })
    return actions
