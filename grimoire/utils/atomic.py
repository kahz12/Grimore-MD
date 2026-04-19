import os
import tempfile
from pathlib import Path
from typing import Callable


def atomic_write(path: Path, writer: Callable[[object], None], *, mode: str = "wb") -> None:
    """
    Write to ``path`` atomically: ``writer(fileobj)`` writes into a sibling
    temp file which is then ``os.replace``d over the destination.
    Permissions of an existing destination are preserved.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(parent))
    try:
        with os.fdopen(fd, mode) as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())

        if path.exists():
            try:
                os.chmod(tmp_path, path.stat().st_mode)
            except OSError:
                pass

        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
