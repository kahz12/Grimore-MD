"""
Atomic File Writing.
This module provides a safe way to write to files by first writing to a
temporary file and then renaming it to the final destination.
This prevents data corruption in case of a crash during the write operation.
"""
import os
import tempfile
from pathlib import Path
from typing import Callable


def atomic_write(path: Path, writer: Callable[[object], None], *, mode: str = "wb") -> None:
    """
    Writes content to a file atomically.
    
    1. Creates a temporary file in the same directory as the target.
    2. Calls the 'writer' function with the temp file object.
    3. Flushes and syncs the temp file to disk.
    4. Preserves file permissions of the original file if it exists.
    5. Replaces the original file with the temporary one.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file in the same directory to ensure os.replace works across the same mount point
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(parent))
    try:
        with os.fdopen(fd, mode) as fh:
            writer(fh)
            fh.flush()
            # Ensure data is physically written to disk
            os.fsync(fh.fileno())

        # Copy permissions from the original file if it exists
        if path.exists():
            try:
                os.chmod(tmp_path, path.stat().st_mode)
            except OSError:
                pass

        # Atomic replacement
        os.replace(tmp_path, path)
    except Exception:
        # Cleanup temporary file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
