"""
System Utilities.
This module provides low-level system interactions such as process management,
daemon backgrounding, and PID file handling.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


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


def _atomic_write_pid(pid_file: str, pid: int) -> None:
    """Writes the PID atomically to a file with restrictive permissions."""
    pid_path = Path(pid_file).resolve()
    fd, tmp_path = tempfile.mkstemp(
        prefix=".pid.", dir=str(pid_path.parent)
    )
    try:
        os.write(fd, f"{pid}\n".encode("utf-8"))
        os.fchmod(fd, 0o600)
        os.close(fd)
        os.replace(tmp_path, pid_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def start_daemon_background(pid_file: str, log_file: str):
    """
    Starts the Grimoire daemon in the background as a separate session.
    Redirects stdout and stderr to the specified log file.
    """
    if is_running(pid_file):
        print("Daemon is already running.")
        return

    # Remove stale PID file if it exists
    if os.path.exists(pid_file):
        os.remove(pid_file)

    cmd = [sys.executable, "-m", "grimoire", "daemon"]

    with open(log_file, "a") as log:
        # Popen with start_new_session=True creates a background process (daemon-like)
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

    _atomic_write_pid(pid_file, process.pid)
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
