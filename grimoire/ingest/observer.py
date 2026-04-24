"""
File system monitoring for Project Grimoire.
Uses the watchdog library to track changes in the Markdown vault and implements
a debounce mechanism to avoid processing files while they are still being edited.
"""
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue, Empty
from threading import Thread
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class VaultEventHandler(FileSystemEventHandler):
    """
    Handles low-level file system events (modified, created, moved).
    Filters events by file extension (.md) and ignored directories.
    """
    def __init__(self, queue: Queue, ignored_dirs: list[str]):
        self.queue = queue
        self.ignored_dirs = ignored_dirs

    def _enqueue(self, raw_path: str, event_type: str):
        """Filters and pushes valid file events into the processing queue."""
        if not raw_path.endswith(".md"):
            return
        path = Path(raw_path)
        if any(ignored in str(path) for ignored in self.ignored_dirs):
            return
        self.queue.put((path, time.time()))
        logger.debug("file_event", type=event_type, path=str(path))

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
    """
    def __init__(self, vault_path: str, callback, ignored_dirs: list[str], debounce_seconds: int = 45):
        self.vault_path = vault_path
        self.callback = callback  # Function to call when a file is ready to process
        self.ignored_dirs = ignored_dirs
        self.debounce_seconds = debounce_seconds
        self.queue = Queue()
        self.pending_changes = {}  # Maps Path -> last_event_timestamp
        self._stop = False

    def start(self):
        """Starts the watchdog observer and the background processing thread."""
        self.observer = Observer()
        self.handler = VaultEventHandler(self.queue, self.ignored_dirs)
        self.observer.schedule(self.handler, self.vault_path, recursive=True)
        self.observer.start()
        
        # Processor thread handles the debounce logic
        self.processor_thread = Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
        
        logger.info("observer_started", path=self.vault_path, debounce=self.debounce_seconds)

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
