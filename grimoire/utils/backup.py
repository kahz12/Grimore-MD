import shutil
import time
from pathlib import Path
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class BackupManager:
    def __init__(self, db_path: str, backup_dir: str = "backups", max_backups: int = 5):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self):
        """
        Creates a copy of the current DB with a timestamp.
        """
        if not self.db_path.exists():
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"grimoire_{timestamp}.db"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info("backup_created", path=str(backup_path))
            self._rotate_backups()
        except Exception as e:
            logger.error("backup_failed", error=str(e))

    def _rotate_backups(self):
        """
        Keeps only the most recent N backups.
        """
        backups = sorted(self.backup_dir.glob("grimoire_*.db"), key=lambda x: x.stat().st_mtime)
        while len(backups) > self.max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logger.info("backup_rotated", removed=oldest.name)
