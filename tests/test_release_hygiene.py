"""Behaviours that only matter at the edges of a run:

* `daemon stop` must succeed quietly even when the daemon has already removed
  its own PID file;
* the version a client reads from the API or MCP must be the package's own;
* `grimore preflight` must report a missing vault instead of raising.
"""
import os

import pytest

from grimore.utils import system


class TestStopClearsThePidFile:
    """The daemon unlinks its own PID file on the way out, so by the time
    stop_daemon sees the process gone the file is usually already missing.
    That is not an error: a successful stop must not look like a failure.
    """

    def _pid_file(self, tmp_path, pid=4242):
        p = tmp_path / "daemon.lock"
        p.write_text(str(pid))
        return str(p)

    def test_a_graceful_stop_survives_the_daemon_clearing_its_own_file(
        self, tmp_path, monkeypatch, capsys,
    ):
        pid_file = self._pid_file(tmp_path)
        monkeypatch.setattr(system, "_is_grimore_process", lambda pid: True)
        monkeypatch.setattr(system, "_send_graceful", lambda pid: None)

        def exit_and_clean(pid, timeout):
            # What the real daemon does: it removes the PID file itself.
            os.remove(pid_file)
            return True

        monkeypatch.setattr(system, "_wait_for_exit", exit_and_clean)
        system.stop_daemon(pid_file)          # must not raise
        assert not os.path.exists(pid_file)
        # And says nothing about it. FileNotFoundError is a subclass of
        # OSError, so the broad handler alone would already stop the crash --
        # what the specific branch buys is silence on the path that is not a
        # problem, and only asserting "no exception" would let it be removed.
        assert "Could not remove PID file" not in capsys.readouterr().out

    def test_a_normal_stop_still_removes_the_file(self, tmp_path, monkeypatch):
        pid_file = self._pid_file(tmp_path)
        monkeypatch.setattr(system, "_is_grimore_process", lambda pid: True)
        monkeypatch.setattr(system, "_send_graceful", lambda pid: None)
        monkeypatch.setattr(system, "_wait_for_exit", lambda pid, timeout: True)
        system.stop_daemon(pid_file)
        assert not os.path.exists(pid_file)

    def test_the_escalation_path_is_tolerant_too(self, tmp_path, monkeypatch):
        pid_file = self._pid_file(tmp_path)
        monkeypatch.setattr(system, "_is_grimore_process", lambda pid: True)
        monkeypatch.setattr(system, "_send_graceful", lambda pid: None)
        monkeypatch.setattr(system, "_send_kill", lambda pid: os.remove(pid_file))
        calls = iter([False, True])           # graceful times out, kill works
        monkeypatch.setattr(system, "_wait_for_exit",
                            lambda pid, timeout: next(calls))
        system.stop_daemon(pid_file)          # must not raise
        assert not os.path.exists(pid_file)

    def test_a_real_permission_error_is_still_reported(self, tmp_path,
                                                       monkeypatch, capsys):
        """Tolerating a missing file must not swallow a genuine failure."""
        pid_file = self._pid_file(tmp_path)

        def refuse(path):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(system.os, "remove", refuse)
        system._clear_pid_file(pid_file)
        assert "Could not remove PID file" in capsys.readouterr().out


class TestVersionIsSingleSourced:
    """The API and MCP both report a version to clients. It must be the
    package's own, never a restated literal that can drift from the release.
    """

    def test_api_matches_the_package(self):
        from grimore import __version__
        from grimore.api.app import _API_VERSION
        assert _API_VERSION == __version__

    def test_mcp_matches_the_package(self):
        from grimore import __version__
        from grimore.mcp_server import _SERVER_INFO
        assert _SERVER_INFO["version"] == __version__

    def test_no_hard_coded_version_literals_remain(self):
        """Versioned modules must derive their value from the package constant."""
        import pathlib
        import re
        for mod in ("grimore/api/app.py", "grimore/mcp_server.py"):
            src = pathlib.Path(mod).read_text()
            for line in src.splitlines():
                if re.search(r'(_API_VERSION|"version")\s*[:=]\s*"\d+\.\d+', line):
                    pytest.fail(f"{mod} restates a version literal: {line.strip()}")


class TestFirstRunOnAMissingVault:
    """`grimore preflight` is the first command the README tells a new user to
    run, and at that point the vault often does not exist yet. GitGuard must
    tolerate that, so the preflight check built to explain the problem is
    what the user sees rather than a GitPython traceback.
    """

    def test_git_guard_survives_a_vault_that_does_not_exist(self, tmp_path):
        from grimore.output.git_guard import GitGuard
        guard = GitGuard(str(tmp_path / "definitely-not-here"))
        assert guard.repo is None
        assert guard.is_repo_ready() is False

    def test_git_guard_still_handles_a_directory_that_is_not_a_repo(self, tmp_path):
        from grimore.output.git_guard import GitGuard
        plain = tmp_path / "plain"
        plain.mkdir()
        guard = GitGuard(str(plain))
        assert guard.repo is None
        assert guard.is_repo_ready() is False

    def test_preflight_reports_the_missing_vault_instead_of_raising(self, tmp_path):
        from grimore.utils.config import (
            CognitionConfig, Config, MemoryConfig, VaultConfig,
        )
        from grimore.utils.preflight import PreflightChecker
        cfg = Config(
            vault=VaultConfig(path=str(tmp_path / "no-vault")),
            cognition=CognitionConfig(),
            memory=MemoryConfig(db_path=str(tmp_path / "g.db")),
        )
        report = PreflightChecker(cfg).run(check_git=True)   # must not raise
        vault = [c for c in report.checks if c.name == "vault_accessible"]
        assert vault and not vault[0].ok
        assert "does not exist" in vault[0].message
