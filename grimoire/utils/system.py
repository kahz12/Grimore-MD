"""
System Utilities.
This module provides low-level system interactions such as process management,
daemon backgrounding, and PID file handling.
"""
import fcntl
import os
import subprocess
import sys
import time

DEFAULT_PID_FILE = "grimoire.pid"


def _read_cmdline(pid: int) -> str:
    """Returns /proc/<pid>/cmdline as a string, used to identify processes."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _is_grimoire_process(pid: int) -> bool:
    """Best-effort check that the PID corresponds to a Grimoire daemon process."""
    cmdline = _read_cmdline(pid)
    return "grimoire" in cmdline


def _read_pid(pid_file: str) -> int | None:
    """Reads the PID from a file."""
    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def is_running(pid_file: str) -> bool:
    """
    Checks if a Grimoire daemon is currently running by inspecting the PID file
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
    return _is_grimoire_process(pid)


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
    fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError:
        os.close(fd)
        raise
    try:
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.fsync(fd)
    except OSError:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        raise
    return fd


def release_pid_lock(fd: int | None, pid_file: str) -> None:
    """Release the advisory lock and remove the PID file. Best-effort."""
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
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
    Starts the Grimoire daemon in the background as a separate session.
    Redirects stdout and stderr to the specified log file.

    The spawned daemon process is responsible for acquiring the advisory
    lock on ``pid_file`` and writing its own PID; this function only does
    a best-effort liveness pre-check for fast UX feedback. If two starts
    race, the kernel-level flock guarantees only one daemon survives.
    """
    if is_running(pid_file):
        print("Daemon is already running.")
        return

    cmd = [sys.executable, "-m", "grimoire", "daemon"]

    # Open with explicit 0o600 so a brand-new log file is never world-readable,
    # regardless of the caller's umask. The mode arg is only honoured when the
    # file is created; if it already exists with looser perms, leave it alone
    # so we don't surprise an operator who set permissions deliberately.
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as log:
        # Popen with start_new_session=True creates a background process (daemon-like)
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

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


def stop_daemon(pid_file: str):
    """
    Stops a running daemon by sending SIGTERM and waiting for it to exit
    before clearing the PID file — otherwise a quick stop→start can leave
    two daemons competing for the same database.
    """
    if not os.path.exists(pid_file):
        print("No PID file found. Is it running?")
        return

    pid = _read_pid(pid_file)
    if pid is None:
        print("PID file is corrupt; refusing to send signals.")
        os.remove(pid_file)
        return

    if not _is_grimoire_process(pid):
        print(
            f"PID {pid} does not look like a grimoire process; refusing to kill. "
            "Removing stale PID file."
        )
        os.remove(pid_file)
        return

    try:
        os.kill(pid, 15)  # SIGTERM (Request graceful shutdown)
    except Exception as e:
        print(f"Error stopping daemon: {e}")
        return

    if _wait_for_exit(pid, timeout=5.0):
        os.remove(pid_file)
        print(f"Stopped daemon (PID: {pid})")
        return

    # Graceful window exhausted — escalate to SIGKILL and give it another beat.
    print(f"Daemon (PID {pid}) ignored SIGTERM after 5s; escalating to SIGKILL.")
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    if _wait_for_exit(pid, timeout=2.0):
        os.remove(pid_file)
        print(f"Force-killed daemon (PID: {pid})")
    else:
        print(
            f"WARNING: PID {pid} still alive after SIGKILL. "
            "Leaving PID file in place; investigate manually."
        )
