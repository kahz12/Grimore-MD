"""
Tests for the v2.1 cross-platform daemon plumbing:

  * grimore.utils.paths   — platformdirs-backed cache paths
  * grimore.utils.system  — cross-platform PID lock + procfs fallback
  * grimore.utils.event_log — atomic one-line event writer
  * grimore.daemon        — signal handlers + clean-shutdown event
  * grimore.utils.config  — DaemonConfig surface

These exercise the I/O surfaces (lock files, event logs, config parsing)
without spinning up the full GrimoreDaemon — that one needs Ollama, a
real vault and watchdog plumbing, all of which are integration territory.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

import pytest

from grimore.utils import paths as paths_mod
from grimore.utils import system as sysmod
from grimore.utils.config import Config, DaemonConfig, load_config
from grimore.utils.event_log import DaemonEventLog


# ── paths ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cache_redirect(tmp_path, monkeypatch):
    """Pin ``platformdirs.user_cache_dir`` to a tmp dir so tests don't
    touch the real ~/.cache/grimore (or AppData on Windows)."""
    fake_cache = tmp_path / "cache" / "grimore"
    monkeypatch.setattr(
        paths_mod, "user_cache_dir", lambda app=None: str(fake_cache)
    )
    return fake_cache


class TestPaths:
    def test_cache_dir_creates_with_0o700(self, cache_redirect):
        path = paths_mod.cache_dir()
        assert path == cache_redirect
        assert path.is_dir()
        if hasattr(os, "fchmod"):
            assert (path.stat().st_mode & 0o777) == 0o700

    def test_daemon_lock_path_under_cache(self, cache_redirect):
        lock = paths_mod.daemon_lock_path()
        assert lock.parent == cache_redirect
        assert lock.name == "daemon.lock"

    def test_daemon_log_path_under_cache(self, cache_redirect):
        log = paths_mod.daemon_log_path()
        assert log.parent == cache_redirect
        assert log.name == "daemon.log"

    def test_shell_history_per_vault(self, tmp_path, cache_redirect):
        a = tmp_path / "vault_a"
        b = tmp_path / "vault_b"
        a.mkdir()
        b.mkdir()
        path_a = paths_mod.shell_history_path(a)
        path_b = paths_mod.shell_history_path(b)
        assert path_a != path_b
        assert path_a.parent == cache_redirect
        assert path_a.name.startswith("shell_history.")
        assert path_a.exists()
        if hasattr(os, "fchmod"):
            assert (path_a.stat().st_mode & 0o777) == 0o600


# ── pid lock ──────────────────────────────────────────────────────────────


class TestPidLock:
    def test_acquire_writes_pid_and_locks(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        fd = sysmod.acquire_pid_lock(str(lock_path))
        assert fd is not None
        try:
            assert lock_path.read_text().strip() == str(os.getpid())
            # Second acquirer in the same process must be refused.
            second = sysmod.acquire_pid_lock(str(lock_path))
            assert second is None
        finally:
            sysmod.release_pid_lock(fd, str(lock_path))
        # release_pid_lock removes the file.
        assert not lock_path.exists()

    def test_acquire_creates_parent_dir(self, tmp_path):
        lock_path = tmp_path / "nested" / "daemon.lock"
        fd = sysmod.acquire_pid_lock(str(lock_path))
        try:
            assert lock_path.exists()
        finally:
            sysmod.release_pid_lock(fd, str(lock_path))

    def test_release_pid_lock_handles_none(self, tmp_path):
        lock_path = tmp_path / "ghost.lock"
        # Should not raise even with no fd and no file on disk.
        sysmod.release_pid_lock(None, str(lock_path))


# ── procfs fallback ───────────────────────────────────────────────────────


class TestIsGrimoreProcessFallback:
    def test_no_procfs_accepts_on_liveness(self, monkeypatch):
        """On Windows/macOS argv is unreadable; verifier defers to liveness."""
        monkeypatch.setattr(sysmod, "_HAS_PROCFS", False)
        monkeypatch.setattr(sysmod, "_read_cmdline_argv", lambda pid: [])
        assert sysmod._is_grimore_process(pid=12345) is True

    def test_procfs_present_but_argv_empty_rejects(self, monkeypatch):
        monkeypatch.setattr(sysmod, "_HAS_PROCFS", True)
        monkeypatch.setattr(sysmod, "_read_cmdline_argv", lambda pid: [])
        assert sysmod._is_grimore_process(pid=12345) is False


# ── event log ─────────────────────────────────────────────────────────────


class TestEventLog:
    def test_writes_tsv_lines(self, tmp_path):
        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path)
        log.write("processed", path="notes/a.md", chunks=3)
        contents = log_path.read_text()
        line = contents.strip()
        parts = line.split("\t")
        # Format: <ISO ts>\tprocessed\tpath=notes/a.md\tchunks=3
        assert parts[1] == "processed"
        assert "path=notes/a.md" in parts
        assert "chunks=3" in parts
        # ISO-8601 UTC: starts with YYYY-MM-DD
        assert parts[0][:4].isdigit()

    def test_appends_not_overwrites(self, tmp_path):
        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path)
        log.write("a")
        log.write("b")
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_disabled_writes_nothing(self, tmp_path):
        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path, enabled=False)
        log.write("event")
        assert not log_path.exists()

    def test_quotes_values_with_whitespace(self, tmp_path):
        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path)
        log.write("err", message="boom: file not found")
        line = log_path.read_text().strip()
        # The whitespace value gets JSON-quoted so it can be parsed back later.
        assert 'message="boom: file not found"' in line

    def test_concurrent_writes_atomic(self, tmp_path):
        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path)

        def worker(n):
            for i in range(50):
                log.write("tick", who=n, i=i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = log_path.read_text().splitlines()
        assert len(lines) == 4 * 50
        # Every line is well-formed: a tab-separated record.
        for line in lines:
            assert "\t" in line
            assert line.split("\t")[1] == "tick"


# ── config ────────────────────────────────────────────────────────────────


class TestDaemonConfig:
    def test_defaults_disabled_with_event_log_on(self):
        cfg = DaemonConfig()
        assert cfg.enabled is False
        assert cfg.log_events is True

    def test_load_config_parses_daemon_section(self, tmp_path, monkeypatch):
        toml = tmp_path / "grimore.toml"
        toml.write_text(
            "[daemon]\n"
            "enabled = true\n"
            "log_events = false\n"
        )
        cfg = load_config(str(toml))
        assert cfg.daemon.enabled is True
        assert cfg.daemon.log_events is False

    def test_load_config_unknown_key_warns_not_crashes(self, tmp_path):
        toml = tmp_path / "grimore.toml"
        toml.write_text("[daemon]\nbogus = 1\n")
        # Should not raise even with an unknown key.
        cfg = load_config(str(toml))
        assert isinstance(cfg.daemon, DaemonConfig)

    def test_phase0_observer_fields_have_sensible_defaults(self):
        # Phase 0.2: new knobs are present, off/sane by default so the v2.1
        # behaviour is unchanged for users who don't touch [daemon].
        cfg = DaemonConfig()
        assert cfg.debounce_seconds == 45
        assert cfg.poll_fallback is False
        assert cfg.poll_interval_s == 30.0

    def test_phase0_observer_fields_load_from_toml(self, tmp_path):
        toml = tmp_path / "grimore.toml"
        toml.write_text(
            "[daemon]\n"
            "debounce_seconds = 5\n"
            "poll_fallback = true\n"
            "poll_interval_s = 10.0\n"
        )
        cfg = load_config(str(toml))
        assert cfg.daemon.debounce_seconds == 5
        assert cfg.daemon.poll_fallback is True
        assert cfg.daemon.poll_interval_s == 10.0


# ── event log: parse + tail ───────────────────────────────────────────────


class TestEventLogParse:
    def test_parse_event_line_simple(self):
        from grimore.utils.event_log import parse_event_line

        out = parse_event_line("2026-05-28T00:00:00Z\tprocessed\tpath=notes/a.md\tchunks=3")
        assert out == {
            "ts": "2026-05-28T00:00:00Z", "event": "processed",
            "path": "notes/a.md", "chunks": "3",
        }

    def test_parse_decodes_json_quoted_values(self):
        from grimore.utils.event_log import parse_event_line

        # _format_value JSON-encodes values containing whitespace; parse
        # round-trips them so a path with a space is intact downstream.
        out = parse_event_line(
            '2026-05-28T00:00:00Z\tprocessed\tpath="notes/with space.md"'
        )
        assert out["path"] == "notes/with space.md"

    def test_parse_malformed_returns_empty(self):
        from grimore.utils.event_log import parse_event_line
        assert parse_event_line("") == {}
        assert parse_event_line("garbage") == {}

    def test_tail_events_returns_last_n(self, tmp_path):
        from grimore.utils.event_log import DaemonEventLog, tail_events

        log_path = tmp_path / "daemon.log"
        log = DaemonEventLog(log_path)
        for i in range(7):
            log.write("processed", path=f"f{i}.md", chunks=i)
        out = tail_events(log_path, n=3)
        assert [e["path"] for e in out] == ["f4.md", "f5.md", "f6.md"]

    def test_tail_missing_file_returns_empty(self, tmp_path):
        from grimore.utils.event_log import tail_events
        assert tail_events(tmp_path / "does-not-exist.log", n=5) == []

    def test_tail_zero_is_empty(self, tmp_path):
        from grimore.utils.event_log import tail_events
        (tmp_path / "x.log").write_text("a\nb\n")
        assert tail_events(tmp_path / "x.log", n=0) == []


# ── daemon status ─────────────────────────────────────────────────────────


class TestDaemonStatus:
    def test_not_running_shape(self, tmp_path, cache_redirect):
        # No PID file exists → `running` False, no PID/uptime, empty events.
        from grimore.operations import _do_daemon_status
        from grimore.utils.config import Config

        out = _do_daemon_status(
            Config(),
            pid_file=str(tmp_path / "nope.pid"),
            log_file=str(tmp_path / "nope.log"),
        )
        assert out["running"] is False
        assert out["pid"] is None
        assert out["events"] == []
        assert out["debounce_seconds"] == 45  # default
        assert "uptime_s" not in out

    def test_running_includes_pid_uptime_events(self, tmp_path, monkeypatch, cache_redirect):
        # Stand up a fake "running" daemon: write a PID file pointing at our
        # own pid (is_running checks os.kill(pid, 0)), seed the event log,
        # then bypass _is_grimore_process which would reject the python test
        # runner as not-grimore.
        from grimore.operations import _do_daemon_status
        from grimore.utils import system as sysmod
        from grimore.utils.event_log import DaemonEventLog
        from grimore.utils.config import Config

        monkeypatch.setattr(sysmod, "_is_grimore_process", lambda pid: True)

        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text(str(os.getpid()))
        log_file = tmp_path / "daemon.log"
        log = DaemonEventLog(log_file)
        log.write("daemon_started", vault="/v", pid=os.getpid())
        log.write("processed", path="a.md", chunks=2)

        out = _do_daemon_status(
            Config(), pid_file=str(pid_file), log_file=str(log_file), tail_n=5,
        )
        assert out["running"] is True
        assert out["pid"] == os.getpid()
        assert out["uptime_s"] >= 0.0
        assert len(out["events"]) == 2
        assert out["last_event"] == "processed"
        assert out["last_path"] == "a.md"


class TestUptimeFormatting:
    def test_fmt_uptime_branches(self):
        from grimore.operations import _fmt_uptime
        assert _fmt_uptime(None) == "?"
        assert _fmt_uptime(0) == "0s"
        assert _fmt_uptime(45) == "45s"
        assert _fmt_uptime(125) == "2m 5s"
        assert _fmt_uptime(3 * 3600 + 42 * 60) == "3h 42m"


# ── daemon signal plumbing ────────────────────────────────────────────────


class _StubObserver:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _StubDB:
    def __init__(self) -> None:
        self.checkpointed = False

    def wal_checkpoint(self):
        self.checkpointed = True
        return {"busy": 0, "log": 0, "checkpointed": 0}


def _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect):
    """Construct a GrimoreDaemon without invoking heavy services.

    We patch out everything the constructor reaches for; the resulting
    instance is enough to exercise start/stop signal handling.
    """
    from grimore import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "Database", lambda *a, **kw: _StubDB())
    monkeypatch.setattr(daemon_mod, "MarkdownParser", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "GitGuard", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "FrontmatterWriter", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "LinkInjector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "Notifier", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "SecurityGuard", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "BackupManager",
                        lambda *a, **kw: type("_B", (), {"latest_backup_mtime": lambda self: 0})())
    monkeypatch.setattr(daemon_mod, "MaintenanceRunner", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "LLMRouter", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "load_taxonomy_from_vault", lambda *a, **kw: {})
    monkeypatch.setattr(daemon_mod, "Tagger", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "Embedder", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod, "Connector", lambda *a, **kw: object())

    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = Config()
    cfg.vault.path = str(vault)
    cfg.memory.db_path = str(tmp_path / "g.db")
    cfg.maintenance.enabled = False
    cfg.daemon = DaemonConfig(enabled=True, log_events=True)

    instance = daemon_mod.GrimoreDaemon(cfg, pid_file=str(tmp_path / "daemon.lock"))
    # Swap the observer for a no-op so start() doesn't try to watch anything.
    monkeypatch.setattr(daemon_mod, "VaultObserver",
                        lambda *a, **kw: _StubObserver())
    return instance, cfg


class TestDaemonSignalHandling:
    def test_default_pid_file_uses_platformdirs(self, tmp_path, monkeypatch, cache_redirect):
        from grimore import daemon as daemon_mod
        # Re-patch Database etc. so __init__ doesn't blow up.
        instance, _ = _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect)
        # Sanity: the instance we built used an explicit pid_file. Now make
        # sure the no-arg form picks up the platformdirs default.
        instance2 = daemon_mod.GrimoreDaemon(instance.config)
        assert instance2.pid_file == str(paths_mod.daemon_lock_path())

    def test_event_log_routes_to_platformdirs(self, tmp_path, monkeypatch, cache_redirect):
        instance, _ = _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect)
        assert instance.event_log.path == paths_mod.daemon_log_path()
        assert instance.event_log.enabled is True

    def test_event_log_disabled_via_config(self, tmp_path, monkeypatch, cache_redirect):
        from grimore import daemon as daemon_mod
        instance, cfg = _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect)
        cfg.daemon = DaemonConfig(enabled=True, log_events=False)
        instance2 = daemon_mod.GrimoreDaemon(cfg, pid_file=str(tmp_path / "x.lock"))
        assert instance2.event_log.enabled is False

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX-only signal delivery (Windows uses CTRL_BREAK_EVENT)",
    )
    def test_sigterm_breaks_main_loop(self, tmp_path, monkeypatch, cache_redirect):
        """A SIGTERM during the run loop must set _stop_requested and unwind."""
        instance, _ = _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect)

        def fire_signal_soon():
            time.sleep(0.1)
            os.kill(os.getpid(), signal.SIGTERM)

        t = threading.Thread(target=fire_signal_soon)
        # Re-install the default SIGTERM handler on the way out so the
        # rest of the pytest session is unaffected.
        prev = signal.getsignal(signal.SIGTERM)
        try:
            t.start()
            instance.start()
        finally:
            t.join()
            signal.signal(signal.SIGTERM, prev)
        assert instance._stop_requested.is_set()
        # stop() ran its cleanup path.
        assert instance._pid_lock_fd is None

    def test_stop_releases_lock_and_flushes_db(self, tmp_path, monkeypatch, cache_redirect):
        instance, _ = _build_minimal_daemon(tmp_path, monkeypatch, cache_redirect)
        # Wire the stub observer manually since stop() expects either None or
        # something with .stop(). Simulate post-start state.
        instance.observer = _StubObserver()
        # Manually acquire the lock so stop() has something to release.
        fd = sysmod.acquire_pid_lock(instance.pid_file)
        instance._pid_lock_fd = fd
        instance.stop()
        assert instance.observer is None
        assert instance._pid_lock_fd is None
        assert instance.db.checkpointed is True
