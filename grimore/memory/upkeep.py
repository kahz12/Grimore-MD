"""
Low-level DB housekeeping primitives: file size, VACUUM, and WAL
checkpointing. The scheduling side (what runs when) lives in
:mod:`grimore.memory.maintenance`; these are the operations it calls.
"""
import sqlite3
from pathlib import Path

from grimore.memory._base import DbBase


class UpkeepMixin(DbBase):
    """VACUUM / WAL-checkpoint primitives for :class:`Database`."""

    def file_size_bytes(self) -> int:
        """On-disk byte size of the main DB file (0 if missing)."""
        try:
            return Path(self.db_path).stat().st_size
        except OSError:
            return 0

    def vacuum(self) -> dict:
        """
        Rewrite the DB file to reclaim free pages. Returns bytes freed and the
        pre/post sizes. VACUUM can't run inside a transaction and briefly holds
        an exclusive lock — the daemon schedules it during the low-traffic tick.
        """
        before = self.file_size_bytes()
        # autocommit (isolation_level=None) — VACUUM refuses to run otherwise.
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        after = self.file_size_bytes()
        return {"before_bytes": before, "after_bytes": after, "reclaimed_bytes": max(before - after, 0)}

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> dict:
        """
        Fold the WAL back into the main DB and optionally truncate the -wal
        sidecar. ``mode`` can be PASSIVE, FULL, RESTART, or TRUNCATE (default —
        the only mode that actually shrinks the sidecar on disk).

        Returns ``{busy, log_frames, checkpointed_frames}`` from SQLite.
        """
        mode_up = mode.upper()
        if mode_up not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError(f"invalid wal_checkpoint mode: {mode!r}")
        with self._get_connection() as conn:
            row = conn.execute(f"PRAGMA wal_checkpoint({mode_up})").fetchone()
        busy, log_frames, ckpt = (row or (0, 0, 0))
        return {"busy": int(busy), "log_frames": int(log_frames), "checkpointed_frames": int(ckpt)}
