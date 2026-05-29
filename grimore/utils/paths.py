"""
Cross-platform filesystem paths for Grimore's runtime artefacts
(daemon lock/log, shell history, …).

Backed by ``platformdirs`` so each OS gets the conventional location
without if/else branches:

  Linux         ~/.cache/grimore/
  Windows       C:\\Users\\<u>\\AppData\\Local\\grimore\\Cache\\
  Termux        ~/.cache/grimore/                (Linux-style)
  macOS         ~/Library/Caches/grimore/        (incidental)

Centralising them here means CLI, shell, daemon and tests all agree on
where state lives — and lets tests redirect them by monkey-patching
``user_cache_dir`` exactly once.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from platformdirs import user_cache_dir

_APP_NAME = "grimore"

DAEMON_LOCK_FILENAME = "daemon.lock"
DAEMON_LOG_FILENAME = "daemon.log"
SHELL_HISTORY_PREFIX = "shell_history"


def cache_dir() -> Path:
    """Return Grimore's user cache directory, creating it if needed.

    Best-effort tightens perms to 0700 on POSIX; Windows ignores the chmod.
    """
    path = Path(user_cache_dir(_APP_NAME))
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except (OSError, NotImplementedError):
        # Windows or a filesystem that doesn't support POSIX chmod.
        pass
    return path


def daemon_lock_path() -> Path:
    """PID + advisory lock file for the singleton daemon."""
    return cache_dir() / DAEMON_LOCK_FILENAME


def daemon_log_path() -> Path:
    """Append-only event log written by the daemon on each save."""
    return cache_dir() / DAEMON_LOG_FILENAME


def sidecar_path_for(
    source: Path | str,
    vault_root: Path | str,
    sidecar_dir: str = ".grimore/sidecars",
) -> Path:
    """Mirror ``source`` (a document inside the vault) to its sidecar ``.md``.

    A PDF at ``<vault>/Books/Foo.pdf`` maps to
    ``<vault>/<sidecar_dir>/Books/Foo.pdf.md``. The original extension is
    preserved in the sidecar filename so two documents that differ only in
    extension (``Foo.pdf`` vs ``Foo.epub``) don't collide on a single
    ``Foo.md`` sidecar.

    Raises ``ValueError`` when ``source`` is not inside ``vault_root`` —
    callers should already have run that check via ``SecurityGuard``, but
    re-validating here keeps the helper safe to drop into new call sites.
    """
    src = Path(source).resolve()
    root = Path(vault_root).resolve()
    try:
        rel = src.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"source {source!r} is not inside vault_root {vault_root!r}"
        ) from exc
    sidecar_root = root / sidecar_dir
    return sidecar_root / rel.parent / f"{rel.name}.md"


def resolve_threads_dir(threads_dir: str | Path) -> Path:
    """Resolve the user-configured threads directory to an absolute path.

    Relative paths anchor under ``Path.home()`` (so the default
    ``.grimore/threads`` lives at ``~/.grimore/threads``); absolute paths
    are honoured as given. The directory is created on demand with
    permissive defaults — callers that need 0700 should chmod after.
    """
    raw = Path(threads_dir).expanduser()
    path = raw if raw.is_absolute() else Path.home() / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def shell_history_path(vault_root: Path | str) -> Path:
    """Per-vault prompt-toolkit history file.

    The vault path is hashed (sha256, first 16 hex) so the cache directory
    listing does not leak the absolute vault location and so two vaults do
    not share question history.
    """
    vault_abs = str(Path(vault_root).resolve())
    digest = hashlib.sha256(vault_abs.encode("utf-8")).hexdigest()[:16]
    path = cache_dir() / f"{SHELL_HISTORY_PREFIX}.{digest}"
    if not path.exists():
        # Create with 0o600 explicitly so umask can't widen it.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    return path
