"""
System Notifications.
This module provides a unified interface for sending notifications.
It automatically detects and uses 'termux-api' if running on Android (Termux),
otherwise it falls back to standard logging.
"""
import subprocess
import shutil
from grimore.utils.logger import get_logger

logger = get_logger(__name__)

class Notifier:
    """
    Handles sending visual/auditory alerts to the user.
    """
    def __init__(self):
        # Check if the termux-notification command is available in the PATH
        self.has_termux_api = shutil.which("termux-notification") is not None

    def notify(self, title: str, message: str, priority: str = "default"):
        """
        Sends a notification.
        If Termux API is available, it shows a system notification.
        Otherwise, it logs the message at the INFO level.
        """
        if self.has_termux_api:
            try:
                subprocess.run([
                    "termux-notification",
                    "--title", title,
                    "--content", message,
                    "--priority", priority,
                    "--id", "grimore_daemon"
                ], check=True)
            except Exception as e:
                logger.error("notification_failed", error=str(e))
        else:
            logger.info("system_notification", title=title, message=message)

    def notify_batch_processed(self, count: int):
        """Sends a notification after a batch of notes has been processed."""
        if count >= 3:
            self.notify(
                "Grimore: Notes Processed",
                f"{count} new notes have been analyzed and tagged.",
                "low"
            )

    def notify_connections_found(self, count: int):
        """Sends a notification when new semantic connections are discovered."""
        if count > 0:
            self.notify(
                "Grimore: New Connections",
                f"The Oracle has discovered {count} common threads in your vault.",
                "default"
            )
