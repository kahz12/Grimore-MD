import os
import subprocess
import sys
from pathlib import Path

def is_running(pid_file: str) -> bool:
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False

def start_daemon_background(pid_file: str, log_file: str):
    """
    Starts the daemon in the background.
    For Termux/Linux simplicity, we use a nohup-like approach.
    """
    if is_running(pid_file):
        print("Daemon is already running.")
        return

    # Using sys.executable to run 'python -m grimoire daemon'
    cmd = [sys.executable, "-m", "grimoire", "daemon"]
    
    with open(log_file, 'a') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True
        )
    
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))
    
    print(f"Daemon started in background (PID: {process.pid})")

def stop_daemon(pid_file: str):
    if not os.path.exists(pid_file):
        print("No PID file found. Is it running?")
        return

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 15) # SIGTERM
        os.remove(pid_file)
        print(f"Stopped daemon (PID: {pid})")
    except Exception as e:
        print(f"Error stopping daemon: {e}")
