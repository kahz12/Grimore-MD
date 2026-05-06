"""
Append-only one-line event log for the daemon.

Format: ``<ISO8601 UTC>\t<event>\t<key=value>...\n``. Tab-separated so
it stays grep-friendly while still trivial to feed into ``cut``/``awk``
or a structured parser. Keys with whitespace are JSON-encoded.

Why a separate file when structlog already exists: ``daemon.log`` is the
human-readable trail (`tail -f` while editing), distinct from the
structured noise that may include tracebacks and library chatter. Both
co-exist; this one is intentionally narrow.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class DaemonEventLog:
    """Thread-safe append-only writer for the daemon event log."""

    def __init__(self, path: Path | str, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = enabled
        # Multiple watchdog threads can fire process_file concurrently; the
        # lock keeps each line atomic so two events don't interleave.
        self._lock = threading.Lock()
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Touch with 0o600 if missing; matches the hardening on shell history.
            if not self._path.exists():
                fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                os.close(fd)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _format_value(value: Any) -> str:
        """Render a single field value safely for a TSV line."""
        s = str(value)
        if any(c in s for c in (" ", "\t", "\n", "=")):
            return json.dumps(s, ensure_ascii=False)
        return s

    def write(self, event: str, **fields: Any) -> None:
        """Append a single event line. Silent no-op if disabled."""
        if not self._enabled:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        parts = [ts, event]
        for key, value in fields.items():
            parts.append(f"{key}={self._format_value(value)}")
        line = "\t".join(parts) + "\n"
        with self._lock:
            # 'a' mode is atomic for short writes on POSIX (single write() call);
            # combined with the threading lock that's enough for our line sizes.
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)
