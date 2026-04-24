"""
Database Backup Management.
This module provides functionality for creating rotating backups of the
Grimoire SQLite database to prevent data loss.
"""
import os
import shutil
import time
from pathlib import Path
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class BackupManager:
    """
    Handles creation and rotation of database backups.
    """
    def __init__(self, db_path: str, backup_dir: str = "backups", max_backups: int = 5):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(exist_ok=True)
        try:
            # Set restrictive permissions for the backup directory
            os.chmod(self.backup_dir, 0o700)
        except OSError as e:
            logger.warning("backup_dir_chmod_failed", error=str(e))

    def create_backup(self):
        """
        Creates a copy of the current database with a timestamp suffix.
        After creating a backup, it triggers the rotation logic.
        """
        if not self.db_path.exists():
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"grimoire_{timestamp}.db"

        try:
            # Copy the database file preserving metadata
            shutil.copy2(self.db_path, backup_path)
            try:
                # Restrict permissions for the backup file
                os.chmod(backup_path, 0o600)
            except OSError as e:
                logger.warning("backup_chmod_failed", path=str(backup_path), error=str(e))
            logger.info("backup_created", path=str(backup_path))
            self._rotate_backups()
        except Exception as e:
            logger.error("backup_failed", error=str(e))

    def latest_backup_mtime(self) -> float | None:
        """
        Return the mtime of the most recent backup file, or None when no
        backups exist. Used by the daemon to anchor its daily window against
        real history instead of process-start time — otherwise a daemon that
        restarts every few hours would never reach the 24h threshold.
        """
        backups = list(self.backup_dir.glob("grimoire_*.db"))
        if not backups:
            return None
        try:
            return max(b.stat().st_mtime for b in backups)
        except OSError:
            return None

    def _rotate_backups(self):
        """
        Deletes the oldest backups if the count exceeds max_backups.
        Ensures only the most recent snapshots are kept.
        """
        backups = sorted(self.backup_dir.glob("grimoire_*.db"), key=lambda x: x.stat().st_mtime)
        while len(backups) > self.max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logger.info("backup_rotated", removed=oldest.name)
