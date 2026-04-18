import subprocess
import shutil
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class Notifier:
    def __init__(self):
        self.has_termux_api = shutil.which("termux-notification") is not None

    def notify(self, title: str, message: str, priority: str = "default"):
        """
        Sends a notification. Uses termux-api if available.
        """
        if self.has_termux_api:
            try:
                subprocess.run([
                    "termux-notification",
                    "--title", title,
                    "--content", message,
                    "--priority", priority,
                    "--id", "grimoire_daemon"
                ], check=True)
            except Exception as e:
                logger.error("notification_failed", error=str(e))
        else:
            logger.info("system_notification", title=title, message=message)

    def notify_batch_processed(self, count: int):
        if count >= 3:
            self.notify(
                "Grimorio: Notas Procesadas",
                f"Se han analizado y etiquetado {count} notas nuevas.",
                "low"
            )

    def notify_connections_found(self, count: int):
        if count > 0:
            self.notify(
                "Grimorio: Nuevas Conexiones",
                f"El Oráculo ha descubierto {count} hilos conductores en tu bóveda.",
                "default"
            )
