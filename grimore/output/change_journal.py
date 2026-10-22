"""Recoverable snapshots for document-changing operations.

Each applied operation owns one change set under Grimore's private cache.  A
file is copied before its first write and fingerprinted again afterwards.
Restoration is refused if the current file no longer matches that recorded
post-write fingerprint, so later human edits are never silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from grimore.utils.atomic import atomic_write
from grimore.utils.hashing import sha256_file
from grimore.utils.paths import cache_dir
from grimore.utils.security import SecurityGuard

_CHANGE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MANIFEST = "manifest.json"


class ChangeConflictError(RuntimeError):
    """A target changed after Grimore wrote it and cannot be restored safely."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__(
            "files changed after the recorded operation: " + ", ".join(paths)
        )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _vault_key(vault_root: Path) -> str:
    return hashlib.sha256(str(vault_root).encode("utf-8")).hexdigest()[:20]


class ChangeJournal:
    """Create, list, and restore bounded per-vault file snapshots."""

    def __init__(
        self,
        vault_root: str | Path,
        *,
        storage_root: str | Path | None = None,
        max_sets: int = 20,
    ) -> None:
        self.vault_root = Path(vault_root).expanduser().resolve()
        base = Path(storage_root) if storage_root is not None else cache_dir() / "changes"
        self.root = base / _vault_key(self.vault_root)
        self.max_sets = max(1, int(max_sets))
        self._active: dict[str, Any] | None = None

    def start(self, operation: str, *, details: dict | None = None) -> str:
        if self._active is not None:
            raise RuntimeError("a change set is already active")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except (OSError, NotImplementedError):
            pass
        change_id = uuid.uuid4().hex
        change_dir = self.root / change_id
        (change_dir / "before").mkdir(parents=True)
        manifest = {
            "version": 1,
            "id": change_id,
            "operation": str(operation)[:100],
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "restored_at": None,
            "details": dict(details or {}),
            "files": [],
        }
        self._active = manifest
        self._write_manifest(manifest)
        return change_id

    def capture_before(self, target: str | Path) -> None:
        manifest = self._require_active()
        resolved, relative = self._resolve_target(target)
        if any(item["path"] == relative for item in manifest["files"]):
            return
        existed = resolved.is_file()
        if resolved.exists() and not existed:
            raise ValueError(f"change target is not a regular file: {relative}")
        index = len(manifest["files"])
        snapshot_name = f"before/{index:04d}.bin" if existed else None
        mode = None
        before_sha = None
        if existed:
            data = resolved.read_bytes()
            snapshot = self.root / manifest["id"] / str(snapshot_name)
            atomic_write(snapshot, lambda fh: fh.write(data), mode="wb")
            try:
                snapshot.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
            mode = stat.S_IMODE(resolved.stat().st_mode)
            before_sha = hashlib.sha256(data).hexdigest()
        manifest["files"].append({
            "path": relative,
            "existed_before": existed,
            "before_sha256": before_sha,
            "before_mode": mode,
            "snapshot": snapshot_name,
            "after_exists": None,
            "after_sha256": None,
        })
        self._write_manifest(manifest)

    def capture_after(self, target: str | Path) -> None:
        manifest = self._require_active()
        resolved, relative = self._resolve_target(target)
        entry = next(
            (item for item in manifest["files"] if item["path"] == relative),
            None,
        )
        if entry is None:
            raise RuntimeError(f"capture_before was not called for {relative}")
        entry["after_exists"] = resolved.is_file()
        entry["after_sha256"] = sha256_file(resolved) if resolved.is_file() else None
        self._write_manifest(manifest)

    def finish(self, status: str = "completed") -> dict:
        manifest = self._require_active()
        if status not in {"completed", "completed_with_errors", "interrupted"}:
            raise ValueError("unsupported change-set status")
        manifest["status"] = status
        manifest["finished_at"] = _now()
        self._write_manifest(manifest)
        result = self._summary(manifest)
        self._active = None
        self._rotate()
        return result

    def list_change_sets(self) -> list[dict]:
        manifests = []
        if not self.root.is_dir():
            return manifests
        for manifest_path in self.root.glob(f"*/{_MANIFEST}"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(data, dict) and data.get("version") == 1:
                manifests.append(self._summary(data))
        return sorted(
            manifests,
            key=lambda item: str(item.get("finished_at") or item.get("started_at") or ""),
            reverse=True,
        )

    def restore(self, change_id: str) -> dict:
        manifest = self._load_manifest(change_id)
        if manifest.get("restored_at"):
            raise ValueError("change set was already restored")
        if manifest.get("status") not in {"completed", "completed_with_errors"}:
            raise ValueError("change set is not complete and cannot be restored")
        entries = [
            entry for entry in manifest.get("files", [])
            if entry.get("after_exists") is not None
        ]
        if not entries:
            raise ValueError("change set contains no restorable files")

        conflicts = []
        resolved_entries = []
        for entry in entries:
            target, relative = self._resolve_target(entry.get("path") or "")
            current_exists = target.is_file()
            current_sha = sha256_file(target) if current_exists else None
            if (
                current_exists != bool(entry.get("after_exists"))
                or current_sha != entry.get("after_sha256")
            ):
                conflicts.append(relative)
            resolved_entries.append((entry, target))
        if conflicts:
            raise ChangeConflictError(conflicts)

        # Validate every required snapshot before restoring the first file.
        # Without this preflight, a missing/corrupt later snapshot could leave
        # a multi-file change set only partially restored.
        for entry, _target in resolved_entries:
            if not entry.get("existed_before"):
                continue
            snapshot = self.root / change_id / str(entry.get("snapshot") or "")
            if not snapshot.is_file():
                raise ValueError(f"snapshot is missing for {entry['path']}")
            if sha256_file(snapshot) != entry.get("before_sha256"):
                raise ValueError(f"snapshot checksum failed for {entry['path']}")

        for entry, target in resolved_entries:
            if entry.get("existed_before"):
                snapshot = self.root / change_id / str(entry.get("snapshot") or "")
                data = snapshot.read_bytes()
                atomic_write(target, lambda fh, payload=data: fh.write(payload), mode="wb")
                mode = entry.get("before_mode")
                if isinstance(mode, int):
                    try:
                        target.chmod(mode)
                    except (OSError, NotImplementedError):
                        pass
            else:
                target.unlink()

        manifest["restored_at"] = _now()
        manifest["status"] = "restored"
        self._write_manifest(manifest)
        return self._summary(manifest)

    def _resolve_target(self, target: str | Path) -> tuple[Path, str]:
        raw = Path(target)
        if not raw.is_absolute():
            raw = self.vault_root / raw
        resolved = SecurityGuard.resolve_within_vault(raw, self.vault_root)
        if raw.is_symlink():
            raise ValueError("change targets may not be symbolic links")
        relative = str(resolved.relative_to(self.vault_root))
        if not relative or relative == ".":
            raise ValueError("change target must be a file inside the vault")
        return resolved, relative

    def _require_active(self) -> dict:
        if self._active is None:
            raise RuntimeError("no active change set")
        return self._active

    def _load_manifest(self, change_id: str) -> dict:
        if not isinstance(change_id, str) or not _CHANGE_ID_RE.fullmatch(change_id):
            raise ValueError("invalid change-set id")
        path = self.root / change_id / _MANIFEST
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LookupError("change set not found") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("change-set manifest is unreadable") from exc
        if not isinstance(data, dict) or data.get("id") != change_id:
            raise ValueError("change-set manifest is invalid")
        return data

    def _write_manifest(self, manifest: dict) -> None:
        path = self.root / manifest["id"] / _MANIFEST
        payload = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        atomic_write(path, lambda fh: fh.write(payload), mode="wb")
        try:
            path.chmod(0o600)
        except (OSError, NotImplementedError):
            pass

    @staticmethod
    def _summary(manifest: dict) -> dict:
        files = []
        for entry in manifest.get("files", []):
            if entry.get("after_exists") is None:
                continue
            action = "created" if not entry.get("existed_before") else "updated"
            files.append({"path": entry.get("path"), "action": action})
        status = str(manifest.get("status") or "unknown")
        return {
            "id": manifest.get("id"),
            "operation": manifest.get("operation"),
            "status": status,
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "restored_at": manifest.get("restored_at"),
            "details": manifest.get("details") or {},
            "file_count": len(files),
            "files": files,
            "restorable": (
                status in {"completed", "completed_with_errors"}
                and bool(files)
                and not manifest.get("restored_at")
            ),
        }

    def _rotate(self) -> None:
        sets = []
        for path in self.root.iterdir():
            if not path.is_dir() or not _CHANGE_ID_RE.fullmatch(path.name):
                continue
            try:
                mtime = (path / _MANIFEST).stat().st_mtime
            except OSError:
                continue
            sets.append((mtime, path))
        for _mtime, path in sorted(sets, reverse=True)[self.max_sets:]:
            shutil.rmtree(path, ignore_errors=True)
