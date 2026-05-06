"""
System Utilities.
This module provides low-level system interactions such as process management,
daemon backgrounding, and PID file handling.

PID lock acquisition is platform-aware: ``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows. Both grant the same guarantee — the kernel
releases the lock on process exit, so a crash never strands a stale lock.
"""
import os
import subprocess
import sys
import time

# fcntl exists on Linux/macOS/Termux; msvcrt is the Windows fallback.
# Importing both at module load keeps the platform check confined to the
# acquire/release helpers below.
try:
    import fcntl  # type: ignore[import]
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

try:
    import msvcrt  # type: ignore[import]
    _HAS_MSVCRT = True
except ImportError:
    msvcrt = None  # type: ignore[assignment]
    _HAS_MSVCRT = False

_HAS_PROCFS = os.path.isdir("/proc")


def _read_cmdline_argv(pid: int) -> list[str]:
    """Return /proc/<pid>/cmdline parsed into argv tokens.

    Returns ``[]`` on platforms without procfs (Windows/macOS) — callers
    treat the empty list as "argv unavailable" and fall back to PID-only
    verification gated on the lock + os.kill liveness.
    """
    if not _HAS_PROCFS:
        return []
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return []
    if not raw:
        return []
    # cmdline is NUL-separated and usually NUL-terminated.
    parts = raw.split(b"\x00")
    if parts and parts[-1] == b"":
        parts.pop()
    return [p.decode("utf-8", errors="replace") for p in parts]


def _is_grimore_process(pid: int) -> bool:
    """
    Strict argv-shape check that the PID corresponds to a Grimore daemon.

    Accepts the two ways the daemon can legitimately enter the process table:
      1. Background form spawned by ``start_daemon_background``:
             [<python>, "-m", "grimore", "daemon", ...]
      2. Console-script form (foreground ``grimore daemon run``):
             [<.../grimore>, "daemon", ...]

    A bare substring match for "grimore" used to be enough but had
    false positives (e.g. an editor opened on this repo, an unrelated
    script with "grimore" in its argv). Matching argv structure
    eliminates that ambiguity. (B-06)

    On platforms without procfs (Windows, macOS) we cannot read another
    process's argv portably, so we trust the lock + ``os.kill(pid, 0)``
    liveness check that already guarded the call. The flock guarantees no
    second daemon can hold the lock; only a stranded stale PID file (the
    daemon dying and a non-Grimore process inheriting that PID before the
    next start) could fool us, and that race exists on POSFS too.
    """
    argv = _read_cmdline_argv(pid)
    if not argv:
        # Platforms without procfs accept on liveness alone (see docstring).
        return not _HAS_PROCFS

    if len(argv) < 2:
        return False

    head = os.path.basename(argv[0])

    # Form 1: python -m grimore daemon [...]
    if argv[1:4] == ["-m", "grimore", "daemon"]:
        return True

    # Form 2: <prefix>/bin/grimore daemon [...]
    if head == "grimore" and argv[1] == "daemon":
        return True

    return False


def _read_pid(pid_file: str) -> int | None:
    """Reads the PID from a file."""
    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def is_running(pid_file: str) -> bool:
    """
    Checks if a Grimore daemon is currently running by inspecting the PID file
    and verifying the process exists and matches our name.
    """
    if not os.path.exists(pid_file):
        return False
    pid = _read_pid(pid_file)
    if pid is None:
        return False
    try:
        # Signal 0 checks if the process is alive without sending a real signal
        os.kill(pid, 0)
    except OSError:
        return False
    return _is_grimore_process(pid)


def _flock_exclusive_nb(fd: int) -> bool:
    """Acquire a non-blocking exclusive lock on ``fd``. Returns False if
    another process already holds it. Raises on unexpected OS errors."""
    if _HAS_FCNTL:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if _HAS_MSVCRT:
        # msvcrt.locking takes a byte count; one byte is enough to make the
        # lock advisory-but-cooperative across processes on Windows.
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    raise RuntimeError(
        "no fcntl or msvcrt available — cannot lock PID file on this platform"
    )


def _flock_release(fd: int) -> None:
    """Best-effort release of the lock acquired by ``_flock_exclusive_nb``."""
    if _HAS_FCNTL:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return
    if _HAS_MSVCRT:
        try:
            # Rewind so the unlock targets the same byte as the lock.
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


def acquire_pid_lock(pid_file: str) -> int | None:
    """
    Take an exclusive advisory lock on the PID file and stamp the current PID
    inside it. The kernel releases the lock when the holding process exits,
    so a crash never leaves a stale lock around — only a stale (but unlocked)
    PID file, which the next acquirer simply overwrites.

    Returns the open file descriptor on success (caller must keep it alive
    for the daemon's lifetime), or ``None`` if another process already holds
    the lock.
    """
    # Make sure the parent directory exists — the lock often lives under the
    # platformdirs cache, which the user may not have on a fresh install.
    parent = os.path.dirname(pid_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _flock_exclusive_nb(fd):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        raise
    try:
        # fchmod is POSIX-only; on Windows the 0o600 mode at open() time is
        # ignored anyway, so just skip silently.
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o600)
            except (OSError, NotImplementedError):
                pass
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.fsync(fd)
    except OSError:
        _flock_release(fd)
        os.close(fd)
        raise
    return fd


def release_pid_lock(fd: int | None, pid_file: str) -> None:
    """Release the advisory lock and remove the PID file. Best-effort."""
    if fd is not None:
        _flock_release(fd)
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        os.unlink(pid_file)
    except OSError:
        pass


def start_daemon_background(pid_file: str, log_file: str):
    """
    Starts the Grimore daemon in the background as a separate session.
    Redirects stdout and stderr to the specified log file.

    The spawned daemon process is responsible for acquiring the advisory
    lock on ``pid_file`` and writing its own PID; this function only does
    a best-effort liveness pre-check for fast UX feedback. If two starts
    race, the kernel-level flock guarantees only one daemon survives.
    """
    if is_running(pid_file):
        print("Daemon is already running.")
        return

    cmd = [sys.executable, "-m", "grimore", "daemon"]

    # Make sure the parent directory of the log exists (it lives under
    # platformdirs.user_cache_dir, which may not yet exist on a fresh box).
    log_parent = os.path.dirname(log_file)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)

    # Open with explicit 0o600 so a brand-new log file is never world-readable,
    # regardless of the caller's umask. The mode arg is only honoured when the
    # file is created; if it already exists with looser perms, leave it alone
    # so we don't surprise an operator who set permissions deliberately.
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    # Detach correctly per platform:
    #   POSIX: start_new_session=True so the daemon survives shell exit and
    #          ignores SIGHUP/SIGTERM aimed at the parent process group.
    #   Windows: CREATE_NEW_PROCESS_GROUP so we can later send CTRL_BREAK_EVENT
    #          to stop the daemon, plus DETACHED_PROCESS so it doesn't grab
    #          the parent's console.
    popen_kwargs: dict = {"stdout": fd, "stderr": fd, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        popen_kwargs["close_fds"] = True
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    finally:
        os.close(fd)

    # Give the child a moment to either acquire the lock or fail; if it
    # exited immediately (lock contention, import error, …) tell the user
    # to look at the log instead of falsely reporting success.
    time.sleep(0.3)
    if process.poll() is not None:
        print(
            f"Daemon failed to start (exited with code {process.returncode}). "
            f"Check {log_file} for details."
        )
        return

    print(f"Daemon started in background (PID: {process.pid})")


def _wait_for_exit(pid: int, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Poll with signal-0 until ``pid`` is gone or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(interval)
    return False


def _send_graceful(pid: int) -> None:
    """Send the platform's "please shut down cleanly" signal.

    POSIX: SIGTERM (15). Windows: CTRL_BREAK_EVENT, paired with the
    CREATE_NEW_PROCESS_GROUP flag set at spawn time so the signal reaches
    the daemon's group and not our own console.
    """
    if sys.platform == "win32":
        import signal as _signal  # local import: signal.CTRL_BREAK_EVENT only on Windows
        os.kill(pid, _signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    else:
        os.kill(pid, 15)


def _send_kill(pid: int) -> None:
    """Force-terminate ``pid``. POSIX uses SIGKILL; Windows uses TerminateProcess
    via signal.SIGTERM (which on Windows maps to TerminateProcess)."""
    if sys.platform == "win32":
        import signal as _signal
        os.kill(pid, _signal.SIGTERM)  # type: ignore[attr-defined]
    else:
        os.kill(pid, 9)


def stop_daemon(pid_file: str):
    """
    Stops a running daemon by sending the platform's graceful-shutdown signal
    and waiting for it to exit before clearing the PID file — otherwise a
    quick stop→start can leave two daemons competing for the same database.
    """
    if not os.path.exists(pid_file):
        print("No PID file found. Is it running?")
        return

    pid = _read_pid(pid_file)
    if pid is None:
        print("PID file is corrupt; refusing to send signals.")
        os.remove(pid_file)
        return

    if not _is_grimore_process(pid):
        print(
            f"PID {pid} does not look like a grimore process; refusing to kill. "
            "Removing stale PID file."
        )
        os.remove(pid_file)
        return

    try:
        _send_graceful(pid)
    except Exception as e:
        print(f"Error stopping daemon: {e}")
        return

    if _wait_for_exit(pid, timeout=5.0):
        os.remove(pid_file)
        print(f"Stopped daemon (PID: {pid})")
        return

    # Graceful window exhausted — escalate to a hard kill and give it another beat.
    print(f"Daemon (PID {pid}) ignored graceful signal after 5s; escalating to kill.")
    try:
        _send_kill(pid)
    except OSError:
        pass
    if _wait_for_exit(pid, timeout=2.0):
        os.remove(pid_file)
        print(f"Force-killed daemon (PID: {pid})")
    else:
        print(
            f"WARNING: PID {pid} still alive after kill. "
            "Leaving PID file in place; investigate manually."
        )
