"""
Periodic maintenance for the persistence layer.

Orchestrates three housekeeping tasks the daemon runs on a schedule:

1. **Purge unused tags** — rows in ``tags`` no longer referenced by any note.
2. **WAL checkpoint** — fold the -wal sidecar back into the main DB and
   truncate it; otherwise it grows without bound under a long-running daemon.
3. **VACUUM** — rewrite the DB file to reclaim free pages left behind by
   deletes (prunes, re-embeds).

Order matters: purge first (frees rows), checkpoint next (so VACUUM sees the
latest state), VACUUM last (rewrite reclaims the now-free pages).
"""
from dataclasses import dataclass, field
from time import perf_counter

from grimoire.memory.db import Database
from grimoire.utils.config import MaintenanceConfig
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MaintenanceReport:
    """Structured result of a maintenance run — useful for tests + logging."""
    tags_purged: int = 0
    checkpoint: dict = field(default_factory=dict)
    vacuum: dict = field(default_factory=dict)
    duration_s: float = 0.0
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tags_purged": self.tags_purged,
            "checkpoint": self.checkpoint,
            "vacuum": self.vacuum,
            "duration_s": round(self.duration_s, 3),
            "skipped": list(self.skipped),
        }


class MaintenanceRunner:
    """Executes the three housekeeping primitives in the right order."""

    def __init__(self, db: Database, config: MaintenanceConfig):
        self.db = db
        self.config = config

    def run(self, *, reason: str = "scheduled") -> MaintenanceReport:
        """
        Run the configured maintenance tasks and return a report. Errors in
        one task don't abort the rest — the daemon shouldn't die because
        VACUUM hit a transient lock.
        """
        report = MaintenanceReport()
        t0 = perf_counter()
        logger.info("maintenance_start", reason=reason)

        if self.config.purge_tags:
            try:
                report.tags_purged = self.db.purge_unused_tags()
            except Exception as e:
                logger.warning("maintenance_purge_failed", error=str(e))
                report.skipped.append("purge_tags")
        else:
            report.skipped.append("purge_tags")

        if self.config.wal_checkpoint:
            try:
                report.checkpoint = self.db.wal_checkpoint("TRUNCATE")
            except Exception as e:
                logger.warning("maintenance_checkpoint_failed", error=str(e))
                report.skipped.append("wal_checkpoint")
        else:
            report.skipped.append("wal_checkpoint")

        if self.config.vacuum:
            try:
                report.vacuum = self.db.vacuum()
            except Exception as e:
                logger.warning("maintenance_vacuum_failed", error=str(e))
                report.skipped.append("vacuum")
        else:
            report.skipped.append("vacuum")

        report.duration_s = perf_counter() - t0
        logger.info("maintenance_complete", **report.as_dict())
        return report
