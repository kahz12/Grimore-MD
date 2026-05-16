"""
Background service for Project Grimore.
This module implements the GrimoreDaemon, which monitors the vault for real-time
changes and automatically triggers cognitive processing (tagging, embedding).
"""
import os as _os  # used only for getpid; aliased to avoid mass-renaming locals
import signal
import threading
import time
from pathlib import Path

_os_getpid = _os.getpid
from grimore.utils.config import Config
from grimore.utils.event_log import DaemonEventLog
from grimore.utils.logger import get_logger
from grimore.utils.paths import daemon_lock_path, daemon_log_path
from grimore.ingest.observer import VaultObserver
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.memory.maintenance import MaintenanceRunner
from grimore.output.git_guard import GitGuard
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.tagger import Tagger
from grimore.memory.taxonomy import load_taxonomy_from_vault
from grimore.cognition.embedder import Embedder
from grimore.cognition.connector import Connector
from grimore.output.link_injector import LinkInjector
from grimore.utils.notifications import Notifier
from grimore.utils.security import SecurityGuard
from grimore.utils.backup import BackupManager
from grimore.utils.system import (
    acquire_pid_lock,
    release_pid_lock,
)

logger = get_logger(__name__)

class GrimoreDaemon:
    """
    Orchestrates background tasks: file system watching, periodic backups,
    and automated processing of new/modified notes.
    """
    def __init__(self, config: Config, pid_file: str | None = None):
        self.config = config
        # Default the lock path through platformdirs so two starts from
        # different cwds still see the same lock file. Callers (tests) can
        # still inject an explicit path.
        self.pid_file = pid_file if pid_file is not None else str(daemon_lock_path())
        self._pid_lock_fd: int | None = None
        # The event log is opt-out via DaemonConfig.log_events; default is on.
        self.event_log = DaemonEventLog(
            daemon_log_path(),
            enabled=getattr(config.daemon, "log_events", True),
        )
        self._stop_requested = threading.Event()
        self.db = Database(config.memory.db_path)
        self.parser = MarkdownParser()
        self.git_guard = GitGuard(config.vault.path)
        self.writer = FrontmatterWriter()
        self.injector = LinkInjector()
        self.notifier = Notifier()
        self.security = SecurityGuard(config.vault.path)
        self.backup = BackupManager(config.memory.db_path)
        self.maintenance = MaintenanceRunner(self.db, config.maintenance)
        
        # Initialize cognitive components
        self.router = LLMRouter(config)
        self.vault_tax = load_taxonomy_from_vault(Path(config.vault.path))
        self.tagger = Tagger(config, self.router, self.vault_tax)
        self.embedder = Embedder(config, cache=self.db)
        self.connector = Connector(self.db, self.embedder)
        
        self.vault_root = Path(config.vault.path).resolve()
        self.observer = None
        self._counter_lock = threading.Lock()
        self.processed_count = 0
        # Internal cadence counters use the monotonic clock so they're
        # immune to NTP steps, manual TZ changes and Termux suspend/resume.
        self.last_batch_time = time.monotonic()
        self.last_maintenance_time = time.monotonic()
        # The daily-backup window stays on the wall clock because it's
        # anchored to the mtime of files under backups/ — those mtimes are
        # wall-clock by definition. Fallback to time.time() for first run.
        self.last_backup_time = self.backup.latest_backup_mtime() or time.time()

    def _log_path(self, file_path: Path) -> str:
        """Helper to get a relative path for cleaner logging."""
        try:
            return str(Path(file_path).resolve().relative_to(self.vault_root))
        except (ValueError, OSError):
            return Path(file_path).name

    def _install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT (and SIGBREAK on Windows) for clean shutdown.

        Signal handlers can only be installed from the main thread, so this
        is called once from start(). Each handler simply sets the stop event;
        the main loop notices on its next tick and unwinds through stop().
        """
        def _handler(signum, _frame):
            logger.info("daemon_signal_received", signum=int(signum))
            self.event_log.write("signal", signum=int(signum))
            self._stop_requested.set()

        # SIGINT is universal (Ctrl-C); SIGTERM is POSIX; SIGBREAK is Windows
        # (matches CTRL_BREAK_EVENT sent by stop_daemon).
        for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # ValueError: not in main thread. OSError: signal not supported.
                # Either way, fall back to KeyboardInterrupt-only shutdown.
                logger.warning("signal_install_failed", signal=sig_name)

    def start(self):
        """
        Starts the file system observer and enters a management loop
        for notifications and backups.
        """
        # Take the advisory lock before doing anything expensive: if another
        # daemon already owns the vault, we want to fail fast rather than
        # half-initialise (and risk e.g. clobbering DB connections).
        self._pid_lock_fd = acquire_pid_lock(self.pid_file)
        if self._pid_lock_fd is None:
            logger.error("daemon_already_running", pid_file=self.pid_file)
            print(
                f"Another Grimore daemon already holds the lock on "
                f"{self.pid_file}. Refusing to start a second instance."
            )
            raise SystemExit(1)

        self._install_signal_handlers()

        logger.info("daemon_starting", vault=self.config.vault.path)
        self.event_log.write("daemon_started", vault=self.config.vault.path, pid=str(_os_getpid()))
        self.observer = VaultObserver(
            vault_path=self.config.vault.path,
            callback=self.process_file,
            ignored_dirs=self.config.vault.ignored_dirs
        )
        self.observer.start()

        try:
            while not self._stop_requested.is_set():
                # Two clocks: monotonic for internal cadences (batch, maintenance)
                # which must not jump if the wall clock is adjusted; wall clock
                # for the backup window since it's compared against file mtimes.
                now_mono = time.monotonic()
                now_wall = time.time()

                # Batch notification: send summary every 5 minutes if activity occurred
                with self._counter_lock:
                    pending = self.processed_count
                if pending > 0 and (now_mono - self.last_batch_time > 300):
                    self.notifier.notify_batch_processed(pending)
                    with self._counter_lock:
                        # Subtract the count we just flushed; a worker may have
                        # incremented further during notify, so don't zero blind.
                        self.processed_count -= pending
                    self.last_batch_time = now_mono

                # Daily backup check (every 24h, anchored to wall clock + mtimes).
                if now_wall - self.last_backup_time > 86400:
                    self.backup.create_backup()
                    self.last_backup_time = now_wall

                # Periodic DB maintenance (VACUUM + WAL checkpoint + tag purge).
                mcfg = self.config.maintenance
                if mcfg.enabled:
                    interval = max(1, mcfg.interval_hours) * 3600
                    if now_mono - self.last_maintenance_time > interval:
                        self.maintenance.run(reason="scheduled")
                        self.last_maintenance_time = now_mono

                # Wake on either the timer or a stop signal — wait() returns
                # True the moment a handler sets the event, so shutdown is
                # responsive instead of waiting up to 10s for the next tick.
                self._stop_requested.wait(timeout=10)
        except KeyboardInterrupt:
            pass
        finally:
            # Release the lock and clean up the PID file even if the loop
            # exits by exception. The kernel would release the flock on
            # process death anyway; this just keeps the on-disk file tidy.
            self.stop()

    def stop(self):
        """Gracefully stops the file system observer, flushes the DB and releases the lock."""
        logger.info("daemon_stopping")
        self.event_log.write("daemon_stopping")
        if self.observer:
            try:
                self.observer.stop()
            except Exception as e:
                logger.warning("observer_stop_failed", error=str(e))
            self.observer = None
        # Best-effort flush so a hard shutdown does not leave WAL behind.
        try:
            self.db.wal_checkpoint()
        except Exception as e:  # pragma: no cover - defensive only
            logger.warning("db_checkpoint_failed", error=str(e))
        if self._pid_lock_fd is not None:
            release_pid_lock(self._pid_lock_fd, self.pid_file)
            self._pid_lock_fd = None

    def process_file(self, file_path: Path):
        """
        The core pipeline triggered when a file is created or modified.
        Follows a similar flow to the 'scan' command but in real-time.
        """
        rel = self._log_path(file_path)
        logger.info("processing_file", path=rel)

        try:
            # 0. Vault-scope guard: reject symlinks or events that somehow
            # point outside the vault we were told to watch.
            try:
                SecurityGuard.resolve_within_vault(file_path, self.vault_root)
            except ValueError:
                logger.warning("path_escape_skipped", path=rel)
                self.event_log.write("skip", path=rel, reason="path_escape")
                return

            # 1. Parse file (defence-in-depth: parser re-validates vault scope)
            note = self.parser.parse_file(file_path, vault_root=self.vault_root)

            # 2. Security & Policy Check
            privacy = note.metadata.get("privacy", "local")
            if privacy == "never_process":
                logger.info("policy_skip", path=rel, reason="privacy: never_process")
                self.event_log.write("skip", path=rel, reason="privacy")
                return

            # Scan for sensitive data (PII, API keys)
            sensitive_findings = self.security.scan_for_sensitive_data(note.content)
            force_local = bool(sensitive_findings)
            if force_local:
                logger.warning("sensitive_data_detected", path=rel, types=sensitive_findings)
                if self.config.cognition.allow_remote:
                    logger.warning("remote_disabled_for_note", path=rel,
                                   reason="sensitive_data_detected")

            # 3. Check hash idempotency to avoid redundant LLM calls
            if self.db.get_content_hash_by_path(str(file_path)) == note.content_hash:
                logger.info("file_unchanged", path=rel)
                self.event_log.write("skip", path=rel, reason="unchanged")
                return

            # 4. Git Guard: automatic snapshot
            if self.config.output.auto_commit and not self.config.output.dry_run:
                self.git_guard.commit_pre_change(str(file_path))

            # 5. Cognition: Tagging & Summary via LLM
            clean_content = self.security.sanitize_prompt(note.content)
            cognition_data = self.tagger.tag_note(clean_content)
            
            # 6. Output: Update note frontmatter
            metadata_updates = {
                "tags": cognition_data["tags"],
                "summary": cognition_data["summary"],
                "category": cognition_data.get("category", ""),
                "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self.writer.write_metadata(file_path, metadata_updates, dry_run=self.config.output.dry_run)

            # 7. Embeddings: Vectorize for semantic index
            embedded = self.embedder.embed_chunks(clean_content)

            # 8. Update Database records
            note_id = self.db.upsert_note(str(file_path), note.title, note.content_hash)
            self.db.update_last_tagged(str(file_path))
            if note_id is not None:
                self.db.upsert_tags(note_id, cognition_data["tags"])
                self.db.set_note_category(note_id, cognition_data.get("category") or None)

            if embedded and note_id is not None:
                # Store new chunks and their vectors
                self.db.delete_note_embeddings(note_id)
                for idx, (chunk_text, vector) in enumerate(embedded):
                    self.db.store_embedding(
                        note_id,
                        idx,
                        chunk_text[:500],
                        self.embedder.serialize_vector(vector),
                    )
                logger.info("file_embedded", path=rel, chunks=len(embedded))

            with self._counter_lock:
                self.processed_count += 1
            self.last_batch_time = time.monotonic()
            logger.info("file_processed_complete", path=rel)
            self.event_log.write(
                "processed",
                path=rel,
                tags=len(cognition_data.get("tags") or []),
                chunks=len(embedded) if embedded else 0,
            )

        except Exception as e:
            logger.error("processing_failed", path=rel, error=str(e))
            self.event_log.write("error", path=rel, error=str(e))
            self.notifier.notify("Grimore: Critical Error", f"Error processing {file_path.name}")
