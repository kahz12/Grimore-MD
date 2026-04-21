import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue, Empty
from threading import Thread
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class VaultEventHandler(FileSystemEventHandler):
    def __init__(self, queue: Queue, ignored_dirs: list[str]):
        self.queue = queue
        self.ignored_dirs = ignored_dirs

    def _enqueue(self, raw_path: str, event_type: str):
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
    def __init__(self, vault_path: str, callback, ignored_dirs: list[str], debounce_seconds: int = 45):
        self.vault_path = vault_path
        self.callback = callback
        self.ignored_dirs = ignored_dirs
        self.debounce_seconds = debounce_seconds
        self.queue = Queue()
        self.pending_changes = {}
        self._stop = False

    def start(self):
        self.observer = Observer()
        self.handler = VaultEventHandler(self.queue, self.ignored_dirs)
        self.observer.schedule(self.handler, self.vault_path, recursive=True)
        self.observer.start()
        
        self.processor_thread = Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
        
        logger.info("observer_started", path=self.vault_path, debounce=self.debounce_seconds)

    def stop(self):
        self._stop = True
        self.observer.stop()
        self.observer.join()

    def _process_queue(self):
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
