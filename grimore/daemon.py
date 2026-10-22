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

from grimore.utils.config import Config
from grimore.utils.event_log import DaemonEventLog
from grimore.utils.hashing import sha256_file
from grimore.utils.index_progress import IndexProgressReporter
from grimore.utils.logger import get_logger
from grimore.utils.paths import daemon_lock_path, daemon_log_path
from grimore.ingest.observer import VaultObserver
from grimore.ingest.extraction_profile import (
    pdf_extraction_profile, pdf_index_profile, pdf_retry_pending,
)
from grimore.ingest.pdf_reconcile import reconcile_unchanged_pdf
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.memory.classification_overrides import apply_classification_override
from grimore.memory.db import Database
from grimore.temporal import derive_document_dates
from grimore.memory.maintenance import MaintenanceRunner
from grimore.output.git_guard import GitGuard
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.output.change_journal import ChangeJournal
from grimore.cognition.chunker import build_candidate_chunks
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.reembed import reembed_note
from grimore.cognition.privacy import ingest_services
from grimore.cognition.tagger import Tagger, tag_document
from grimore.memory.taxonomy import load_taxonomy_from_vault
from grimore.cognition.embedder import Embedder
from grimore.utils.notifications import Notifier
from grimore.utils.security import SecurityGuard
from grimore.utils.backup import BackupManager
from grimore.utils.system import (
    acquire_pid_lock,
    release_pid_lock,
)

# Bound once, below the imports, so the module body stays import-only up top.
_os_getpid = _os.getpid

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
        self.index_progress = IndexProgressReporter(config.vault.path, "daemon")
        self._stop_requested = threading.Event()
        self.db = Database(config.memory.db_path)
        self.parser = MarkdownParser()
        self.git_guard = GitGuard(config.vault.path)
        self.writer = FrontmatterWriter()
        self.notifier = Notifier()
        self.security = SecurityGuard(config.vault.path)
        self.backup = BackupManager(config.memory.db_path)
        self.maintenance = MaintenanceRunner(self.db, config.maintenance)

        # Initialize cognitive components
        self.router = LLMRouter(config)
        self.vault_tax = load_taxonomy_from_vault(Path(config.vault.path))
        self.tagger = Tagger(config, self.router, self.vault_tax)
        self.embedder = Embedder(config, cache=self.db)

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
        self.index_progress.start(total=None, dry_run=self.config.output.dry_run)
        # Pull the debounce + polling knobs from DaemonConfig so users can
        # tune them via grimore.toml without editing source. The defaults
        # match the v2.1 behaviour (45s debounce, native inotify/FSEvents).
        self.observer = VaultObserver(
            vault_path=self.config.vault.path,
            callback=self.process_file,
            delete_callback=self.remove_file,
            ignored_dirs=self.config.vault.ignored_dirs,
            supported_extensions=self.config.vault.formats,
            sniff_magic=self.config.ingest.sniff_magic,
            debounce_seconds=getattr(self.config.daemon, "debounce_seconds", 45),
            poll_fallback=getattr(self.config.daemon, "poll_fallback", False),
            poll_interval_s=getattr(self.config.daemon, "poll_interval_s", 30.0),
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
        progress = getattr(self, "index_progress", None)
        if progress is not None:
            progress.finish(status="stopped")
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
        # Release the per-thread connections. The daemon indexes on the
        # watchdog observer thread, so there is one here and one on the main
        # thread; close() reaches both. Ordered after the checkpoint, which
        # needs a live connection.
        try:
            self.db.close()
        except Exception as e:  # pragma: no cover - defensive only
            logger.warning("db_close_failed", error=str(e))
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
        progress = getattr(self, "index_progress", None)
        change_journal = None
        if progress is not None:
            progress.begin_item(rel)

        try:
            # 0. Vault-scope guard: reject symlinks or events that somehow
            # point outside the vault we were told to watch.
            try:
                SecurityGuard.resolve_within_vault(file_path, self.vault_root)
            except ValueError:
                logger.warning("path_escape_skipped", path=rel)
                self.event_log.write("skip", path=rel, reason="path_escape")
                if progress is not None:
                    progress.record("skipped", rel, reason="path_escape")
                return

            # 0a. Never re-ingest files inside our own sidecar tree —
            # they are Grimore output, not user content.
            sidecar_root = (self.vault_root / self.config.vault.sidecar_dir).resolve()
            try:
                Path(file_path).resolve().relative_to(sidecar_root)
            except ValueError:
                pass
            else:
                self.event_log.write("skip", path=rel, reason="sidecar")
                if progress is not None:
                    progress.record("skipped", rel, reason="sidecar")
                return

            # Size+mtime is the O(1) fast path. Hash bytes only when the
            # metadata fingerprint changed or has not been recorded yet.
            try:
                disk_stat = file_path.stat()
                stored_state = self.db.get_ingest_state(str(file_path))
                current_pdf_profile = pdf_extraction_profile(self.config)
                profile_changed = bool(
                    stored_state
                    and (stored_state[3] == "pdf" or file_path.suffix.lower() == ".pdf")
                    and stored_state[4] != current_pdf_profile
                )
                retry_pending = bool(
                    stored_state and pdf_retry_pending(stored_state[4], self.config)
                )
                temporal_backfill = self.db.note_timeline_needs_backfill(str(file_path))
                if not temporal_backfill and not profile_changed and stored_state is not None and stored_state[1:3] == (
                    disk_stat.st_size, disk_stat.st_mtime_ns,
                ):
                    self.event_log.write("skip", path=rel, reason="unchanged_stat")
                    if progress is not None:
                        progress.record("skipped", rel, reason="unchanged_stat")
                    return
                file_hash = sha256_file(file_path)
            except OSError as e:
                logger.error("file_hash_failed", path=rel, error=str(e))
                self.event_log.write("error", path=rel, error=str(e))
                if progress is not None:
                    progress.record("failed", rel, reason="file_hash", error=str(e))
                return
            stored_hash = (
                stored_state[0] if stored_state
                else self.db.get_file_hash(str(file_path))
            )
            if stored_hash == file_hash and not temporal_backfill and not profile_changed:
                if not self.config.output.dry_run:
                    self.db.update_file_state(
                        str(file_path), file_hash,
                        disk_stat.st_size, disk_stat.st_mtime_ns,
                    )
                self.event_log.write("skip", path=rel, reason="unchanged_bytes")
                if progress is not None:
                    progress.record("skipped", rel, reason="unchanged_bytes")
                return

            # 1. Parse file (defence-in-depth: parser re-validates vault scope)
            note = self.parser.parse_file(
                file_path, vault_root=self.vault_root, config=self.config,
            )
            document_dates = derive_document_dates(
                note.metadata, note.format, mtime_ns=disk_stat.st_mtime_ns,
            )

            # 2. Security & Policy Check
            privacy = note.metadata.get("privacy", "local")
            if privacy == "never_process":
                logger.info("policy_skip", path=rel, reason="privacy: never_process")
                self.event_log.write("skip", path=rel, reason="privacy")
                if progress is not None:
                    progress.record("skipped", rel, reason="privacy")
                return

            # Scan for sensitive data (PII, API keys)
            sensitive_findings = self.security.scan_for_sensitive_data(note.content)
            force_local = bool(sensitive_findings)
            if force_local:
                logger.warning("sensitive_data_detected", path=rel, types=sensitive_findings)

            # 3. Text is unchanged, but PDF page anchors or diagnostics may
            # still differ. Reconcile those before publishing the new file
            # fingerprint so write/embedding failures remain retryable.
            if ((not profile_changed or retry_pending) and
                    self.db.get_content_hash_by_path(str(file_path)) == note.content_hash):
                extraction = extraction_diagnostics(note)
                refreshed = (
                    self.writer.refresh_extraction_metadata(
                        note, extraction,
                        vault_root=self.vault_root,
                        sidecar_dir=self.config.vault.sidecar_dir,
                        write_sidecars=self.config.vault.write_sidecars,
                        dry_run=True,
                    ) if extraction is not None else None
                )
                if refreshed and not self.config.output.dry_run:
                    try:
                        change_journal = ChangeJournal(self.vault_root)
                        change_journal.start(
                            "daemon-scan",
                            details={"source": "daemon", "path": rel,
                                     "reason": "extraction_metadata"},
                        )
                    except OSError as exc:
                        logger.warning(
                            "change_journal_unavailable", path=rel, error=str(exc),
                        )
                        change_journal = None
                    self.writer.refresh_extraction_metadata(
                        note, extraction,
                        vault_root=self.vault_root,
                        sidecar_dir=self.config.vault.sidecar_dir,
                        write_sidecars=self.config.vault.write_sidecars,
                        dry_run=False,
                        change_journal=change_journal,
                    )
                    if change_journal is not None:
                        change_journal.finish()
                        change_journal = None
                embedded_chunks = 0
                if note.format == "pdf" and not self.config.output.dry_run:
                    with ingest_services(
                        self.config, self.tagger, self.embedder,
                        sensitive=force_local,
                    ) as (_note_tagger, note_embedder):
                        result = reconcile_unchanged_pdf(
                            self.db, note_embedder, note, self.config,
                            clean_content=self.security.sanitize_prompt(note.content),
                            file_hash=file_hash,
                            size_bytes=disk_stat.st_size,
                            mtime_ns=disk_stat.st_mtime_ns,
                            document_dates=document_dates,
                        )
                    embedded_chunks = result.embedded
                elif not self.config.output.dry_run:
                    self.db.update_file_state(
                        str(file_path), file_hash,
                        disk_stat.st_size, disk_stat.st_mtime_ns,
                    )
                    self.db.record_document_dates(str(file_path), **document_dates)
                if note.format == "pdf":
                    logger.info(
                        "pdf_same_text_refreshed", path=rel,
                        embedded=embedded_chunks,
                    )
                    self.event_log.write(
                        "preview" if self.config.output.dry_run else "processed",
                        path=rel, reason="pdf_same_text_refresh",
                        chunks=embedded_chunks,
                        extraction_status=extraction["status"],
                    )
                    if progress is not None:
                        if self.config.output.dry_run and refreshed:
                            try:
                                refreshed["target"] = str(
                                    Path(refreshed["target"]).resolve().relative_to(
                                        self.vault_root,
                                    )
                                )
                            except (OSError, ValueError):
                                refreshed["target"] = Path(refreshed["target"]).name
                        progress.record(
                            "previewed" if self.config.output.dry_run else "processed",
                            rel, reason="pdf_same_text_refresh",
                            chunks=embedded_chunks,
                            changes=refreshed if self.config.output.dry_run else None,
                        )
                    if not self.config.output.dry_run:
                        with self._counter_lock:
                            self.processed_count += 1
                        self.last_batch_time = time.monotonic()
                else:
                    logger.info("file_unchanged", path=rel)
                    self.event_log.write("skip", path=rel, reason="unchanged")
                    if progress is not None:
                        progress.record("skipped", rel, reason="unchanged_content")
                return

            with ingest_services(
                self.config, self.tagger, self.embedder, sensitive=force_local,
            ) as (note_tagger, note_embedder):
                # 4. Git Guard: automatic snapshot
                if self.config.output.auto_commit and not self.config.output.dry_run:
                    self.git_guard.commit_pre_change(str(file_path))

                # 5. Cognition: Tagging & Summary via LLM
                clean_content = self.security.sanitize_prompt(note.content)
                cognition_data = tag_document(note_tagger, note, clean_content)
                cognition_data = apply_classification_override(
                    self.db, str(file_path), cognition_data,
                )

                # 6. Output: Update note frontmatter (inline for .md, sidecar
                # for every other format).
                metadata_updates = {
                    "tags": cognition_data["tags"],
                    "summary": cognition_data["summary"],
                    "category": cognition_data.get("category", ""),
                    "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                extraction = extraction_diagnostics(note)
                if extraction is not None:
                    metadata_updates["extraction"] = extraction
                if "section_analysis" in cognition_data:
                    metadata_updates["section_analysis"] = cognition_data["section_analysis"]
                elif "section_analysis" in note.metadata:
                    metadata_updates["section_analysis"] = None
                if self.config.output.dry_run:
                    changes = self.writer.preview_metadata(
                        note,
                        metadata_updates,
                        vault_root=self.vault_root,
                        sidecar_dir=self.config.vault.sidecar_dir,
                        write_sidecars=self.config.vault.write_sidecars,
                    )
                    if changes:
                        try:
                            changes["target"] = str(
                                Path(changes["target"]).resolve().relative_to(
                                    self.vault_root,
                                )
                            )
                        except (OSError, ValueError):
                            changes["target"] = Path(changes["target"]).name
                else:
                    changes = None
                    try:
                        change_journal = ChangeJournal(self.vault_root)
                        change_journal.start(
                            "daemon-scan",
                            details={"source": "daemon", "path": rel},
                        )
                    except OSError as exc:
                        logger.warning(
                            "change_journal_unavailable", path=rel, error=str(exc),
                        )
                        change_journal = None
                target = self.writer.write_metadata(
                    note,
                    metadata_updates,
                    vault_root=self.vault_root,
                    sidecar_dir=self.config.vault.sidecar_dir,
                    write_sidecars=self.config.vault.write_sidecars,
                    dry_run=self.config.output.dry_run,
                    change_journal=change_journal,
                )

                if change_journal is not None:
                    change_journal.finish()
                    change_journal = None

                if self.config.output.dry_run:
                    self.event_log.write("preview", path=rel,
                                         tags=len(cognition_data.get("tags") or []))
                    if progress is not None:
                        progress.record("previewed", rel, changes=changes)
                    return

                if not self.config.output.dry_run:
                    latest_stat = file_path.stat()
                    if (latest_stat.st_size, latest_stat.st_mtime_ns) != (
                        disk_stat.st_size, disk_stat.st_mtime_ns,
                    ):
                        file_hash = sha256_file(file_path)
                    disk_stat = latest_stat

                # 7. Update Database records
                sidecar_target_str = (
                    str(target) if (note.format != "md" and target is not None) else None
                )
                note_id = self.db.upsert_note(
                    str(file_path), note.title, "",
                    format=note.format,
                    file_hash="",
                    sidecar_path=sidecar_target_str,
                    size_bytes=-1,
                    mtime_ns=-1,
                )
                self.db.update_last_tagged(str(file_path))
                if note_id is not None:
                    self.db.upsert_tags(note_id, cognition_data["tags"])
                    self.db.set_note_category(note_id, cognition_data.get("category") or None)

                # 8. Embeddings: Vectorize for semantic index. Section-aware
                # when the adapter handed us page / heading anchors (PDF /
                # EPUB / DOCX / HTML); body-only chunking otherwise. The
                # incremental re-embed path means an edit to one paragraph
                # of a long doc only re-embeds that paragraph's chunk.
                chunk_count = 0
                if note_id is not None:
                    candidate_chunks = build_candidate_chunks(
                        note, clean_content, note_embedder, self.config,
                    )
                    result = reembed_note(
                        self.db, note_embedder, note_id, candidate_chunks,
                        text_truncation=getattr(
                            self.config.cognition, "chunk_store_chars", 500
                        ),
                    )
                    if result.stored != len(candidate_chunks):
                        raise RuntimeError("embedding incomplete; file remains pending for retry")
                    chunk_count = result.stored
                    self.db.complete_note_indexing(
                        note_id, note.content_hash, file_hash,
                        disk_stat.st_size, disk_stat.st_mtime_ns,
                        extraction_profile=(
                            pdf_index_profile(self.config, extraction)
                            if note.format == "pdf" else None
                        ),
                        extraction=extraction,
                        **document_dates,
                    )
                    logger.info(
                        "file_embedded", path=rel,
                        kept=result.kept, embedded=result.embedded,
                        removed=result.removed,
                    )

                with self._counter_lock:
                    self.processed_count += 1
                self.last_batch_time = time.monotonic()
                logger.info("file_processed_complete", path=rel)
                self.event_log.write(
                    "processed",
                    path=rel,
                    tags=len(cognition_data.get("tags") or []),
                    chunks=chunk_count,
                    extraction_status=(extraction["status"] if extraction else None),
                )
                if progress is not None:
                    progress.record(
                        "processed", rel, chunks=chunk_count,
                        reason=("partial_extraction" if extraction and
                                extraction["status"] == "partial" else None),
                    )

        except Exception as e:
            if change_journal is not None:
                try:
                    change_journal.finish("interrupted")
                except Exception:
                    pass
            logger.error("processing_failed", path=rel, error=str(e))
            self.event_log.write("error", path=rel, error=str(e))
            if progress is not None:
                progress.record("failed", rel, reason="processing", error=str(e))
            self.notifier.notify("Grimore: Critical Error", f"Error processing {file_path.name}")

    def remove_file(self, file_path: Path) -> None:
        """Remove every index row belonging to a deleted or renamed source."""
        rel = self._log_path(file_path)
        try:
            resolved = SecurityGuard.resolve_within_vault(file_path, self.vault_root)
        except ValueError:
            logger.warning("delete_path_escape_skipped", path=rel)
            self.event_log.write("skip", path=rel, reason="delete_path_escape")
            return
        if self.config.output.dry_run:
            self.event_log.write("preview_delete", path=rel)
            progress = getattr(self, "index_progress", None)
            if progress is not None:
                progress.record("previewed", rel, reason="delete")
            return
        removed = self.db.delete_note_by_path(str(resolved))
        if removed:
            logger.info("file_removed_from_index", path=rel)
            self.event_log.write("deleted", path=rel)
            progress = getattr(self, "index_progress", None)
            if progress is not None:
                progress.record("removed", rel)
