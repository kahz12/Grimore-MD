import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_cmdline(pid: int) -> str:
    """Return /proc/<pid>/cmdline as a printable string, or '' if unavailable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _is_grimoire_process(pid: int) -> bool:
    """Best-effort check that the PID corresponds to a grimoire daemon."""
    cmdline = _read_cmdline(pid)
    return "grimoire" in cmdline


def _read_pid(pid_file: str) -> int | None:
    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def is_running(pid_file: str) -> bool:
    if not os.path.exists(pid_file):
        return False
    pid = _read_pid(pid_file)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return _is_grimoire_process(pid)


def _atomic_write_pid(pid_file: str, pid: int) -> None:
    """Write the PID atomically with restrictive permissions."""
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
    Starts the daemon in the background.
    For Termux/Linux simplicity, we use a nohup-like approach.
    """
    if is_running(pid_file):
        print("Daemon is already running.")
        return

    # If a stale PID file exists, remove it before starting.
    if os.path.exists(pid_file):
        os.remove(pid_file)

    cmd = [sys.executable, "-m", "grimoire", "daemon"]

    with open(log_file, "a") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

    _atomic_write_pid(pid_file, process.pid)
    print(f"Daemon started in background (PID: {process.pid})")


def stop_daemon(pid_file: str):
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
        os.kill(pid, 15)  # SIGTERM
        os.remove(pid_file)
        print(f"Stopped daemon (PID: {pid})")
    except Exception as e:
        print(f"Error stopping daemon: {e}")
