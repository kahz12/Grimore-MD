"""
File system monitoring for Project Grimore.
Uses the watchdog library to track changes in the document vault and
implements a debounce mechanism to avoid processing files while they are
still being edited. Extension filtering is driven by ``config.vault.formats``
so the watcher picks up every format the user has enabled (Markdown by
default; PDFs, ePubs, DOCX, … as later phases ship their adapters).
"""
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from queue import Queue, Empty
from threading import Thread
from typing import Iterable, Optional
from grimore.utils.config import is_ignored_path
from grimore.utils.logger import get_logger

logger = get_logger(__name__)


def _normalise_extensions(exts: Optional[Iterable[str]]) -> frozenset[str]:
    """Lowercase, strip leading dots, drop empties. ``None`` → MD-only."""
    if not exts:
        return frozenset({"md"})
    out: set[str] = set()
    for ext in exts:
        if not ext:
            continue
        out.add(ext.lower().lstrip("."))
    return frozenset(out) or frozenset({"md"})


class VaultEventHandler(FileSystemEventHandler):
    """
    Handles low-level file system events (modified, created, moved).
    Filters events by file extension (against ``supported_extensions``)
    and ignored directories.

    When ``sniff_magic`` is True, files whose extension misses the
    configured set are content-sniffed before being dropped. If the
    sniffer recognises the bytes as a format we ship an adapter for,
    the event is still queued — keeping the daemon in step with the
    ``grimore scan`` sweep, which also widens its pickup under the
    same flag.
    """
    def __init__(
        self,
        queue: Queue,
        ignored_dirs: list[str],
        supported_extensions: Optional[Iterable[str]] = None,
        sniff_magic: bool = False,
    ):
        self.queue = queue
        self.ignored_dirs = ignored_dirs
        self.supported_extensions = _normalise_extensions(supported_extensions)
        self.sniff_magic = sniff_magic

    def _enqueue(self, raw_path: str, event_type: str):
        """Filters and pushes valid file events into the processing queue."""
        path = Path(raw_path)
        # suffix includes the leading dot — strip it for the membership test.
        ext = path.suffix.lower().lstrip(".")
        if ext not in self.supported_extensions:
            # Extension miss. Either give up, or — when the user has
            # opted into content sniffing — ask libmagic whether the
            # bytes look like a format we can actually ingest.
            if not (self.sniff_magic and self._sniff_matches(path)):
                return
        if is_ignored_path(path, self.ignored_dirs):
            return
        self.queue.put((path, time.time()))
        logger.debug("file_event", type=event_type, path=str(path))

    def _sniff_matches(self, path: Path) -> bool:
        """True if libmagic claims ``path`` is one of the configured
        formats. Imported lazily so the daemon never pays the libmagic
        cost when sniffing is off."""
        try:
            from grimore.ingest.sniffer import sniff_extension
        except Exception:  # pragma: no cover - import guard
            return False
        sniffed = sniff_extension(path)
        return bool(sniffed and sniffed in self.supported_extensions)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._enqueue(event.src_path, "modified")

    def on_created(self, event):
        if event.is_directory:
            return
        self._enqueue(event.src_path, "created")

    def on_moved(self, event):
        if event.is_directory:
            return
        # Process destination; a rename is effectively a new file for the vault.
        self._enqueue(event.dest_path, "moved")

class VaultObserver:
    """
    High-level observer that manages the watchdog process and implements
    a debounce timer (default 45s) to ensure file writes are complete.

    ``poll_fallback`` swaps the native inotify/FSEvents/ReadDirectoryChangesW
    observer for ``watchdog.observers.polling.PollingObserver`` — necessary
    on filesystems where the native APIs don't deliver events reliably
    (Termux's ``/storage`` shim, some NFS/FUSE mounts). The trade-off is
    poll latency in exchange for never missing a change.
    """
    def __init__(
        self,
        vault_path: str,
        callback,
        ignored_dirs: list[str],
        debounce_seconds: int = 45,
        supported_extensions: Optional[Iterable[str]] = None,
        sniff_magic: bool = False,
        poll_fallback: bool = False,
        poll_interval_s: float = 30.0,
    ):
        self.vault_path = vault_path
        self.callback = callback  # Function to call when a file is ready to process
        self.ignored_dirs = ignored_dirs
        self.supported_extensions = _normalise_extensions(supported_extensions)
        self.sniff_magic = sniff_magic
        self.debounce_seconds = debounce_seconds
        self.poll_fallback = poll_fallback
        self.poll_interval_s = poll_interval_s
        self.queue = Queue()
        self.pending_changes = {}  # Maps Path -> last_event_timestamp
        self._stop = False

    def start(self):
        """Starts the watchdog observer and the background processing thread."""
        if self.poll_fallback:
            self.observer = PollingObserver(timeout=self.poll_interval_s)
        else:
            self.observer = Observer()
        self.handler = VaultEventHandler(
            self.queue, self.ignored_dirs, self.supported_extensions,
            sniff_magic=self.sniff_magic,
        )
        self.observer.schedule(self.handler, self.vault_path, recursive=True)
        self.observer.start()
        
        # Processor thread handles the debounce logic
        self.processor_thread = Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
        
        logger.info(
            "observer_started",
            path=self.vault_path,
            debounce=self.debounce_seconds,
            mode="polling" if self.poll_fallback else "native",
            poll_interval=self.poll_interval_s if self.poll_fallback else None,
        )

    def stop(self):
        """Signals the observer to stop and waits for the thread to join."""
        self._stop = True
        self.observer.stop()
        self.observer.join()
        # Also wait for the debounce/processor thread so in-flight callbacks
        # finish before the process tears down (daemon=True alone lets them
        # die mid-write on interpreter exit).
        processor = getattr(self, "processor_thread", None)
        if processor is not None:
            # Slightly over the 1s poll inside _process_queue so a tick in
            # progress can complete cleanly.
            processor.join(timeout=2.0)

    def _process_queue(self):
        """
        Main loop for the background thread.
        Moves items from the event queue to pending_changes, then executes the
        callback for files that haven't been touched for at least debounce_seconds.
        """
        while not self._stop:
            try:
                # Get all available events from queue
                while True:
                    try:
                        path, timestamp = self.queue.get_nowait()
                        self.pending_changes[path] = timestamp
                    except Empty:
                        break
                
                # Check for debounced events
                now = time.time()
                to_process = []
                for path, last_time in list(self.pending_changes.items()):
                    if now - last_time >= self.debounce_seconds:
                        to_process.append(path)
                        del self.pending_changes[path]
                
                for path in to_process:
                    logger.info("debounce_expired", path=str(path))
                    self.callback(path)
                
                time.sleep(1)
            except Exception as e:
                logger.error("observer_processor_error", error=str(e))
                time.sleep(1)
