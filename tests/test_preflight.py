"""Tests for the startup preflight validator."""
from __future__ import annotations

from unittest.mock import MagicMock

from grimore.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)
from grimore.utils.preflight import (
    CheckResult,
    PreflightChecker,
    PreflightReport,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_config(tmp_path, *, auto_commit: bool = False) -> Config:
    """Minimal Config pointing at a real on-disk vault dir."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(exist_ok=True)
    return Config(
        vault=VaultConfig(path=str(vault_dir), ignored_dirs=[]),
        cognition=CognitionConfig(
            model_llm_local="qwen2.5:3b",
            model_embeddings_local="nomic-embed-text",
        ),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=auto_commit, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


class _FakeResponse:
    """Stand-in for a requests.Response — just enough for the probe."""
    def __init__(self, payload, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _session_with_models(models: list[str]):
    """Build a MagicMock session whose GET returns those model names."""
    session = MagicMock()
    session.get.return_value = _FakeResponse({"models": [{"name": m} for m in models]})
    return session


def _failing_session(exc: Exception):
    session = MagicMock()
    session.get.side_effect = exc
    return session


# ── Report / result primitives ──────────────────────────────────────────────


class TestReport:
    def test_ok_true_when_all_pass(self):
        r = PreflightReport()
        r.add(CheckResult(name="a", ok=True))
        r.add(CheckResult(name="b", ok=True))
        assert r.ok is True
        assert r.errors == []
        assert r.warnings == []

    def test_ok_false_when_any_error(self):
        r = PreflightReport()
        r.add(CheckResult(name="a", ok=False, severity="error"))
        r.add(CheckResult(name="b", ok=True))
        assert r.ok is False
        assert len(r.errors) == 1

    def test_warning_does_not_block_ok(self):
        r = PreflightReport()
        r.add(CheckResult(name="git", ok=False, severity="warning"))
        assert r.ok is True
        assert r.has_warnings is True


# ── Ollama reachability ─────────────────────────────────────────────────────


class TestOllamaReachable:
    def test_happy_path(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        ollama = next(c for c in report.checks if c.name == "ollama_reachable")
        assert ollama.ok is True
        assert "2 models" in ollama.message

    def test_connection_refused_is_actionable(self, tmp_path):
        config = _make_config(tmp_path)
        session = _failing_session(ConnectionRefusedError("nope"))
        report = PreflightChecker(config, session=session).run()

        ollama = next(c for c in report.checks if c.name == "ollama_reachable")
        assert ollama.ok is False
        assert ollama.severity == "error"
        assert "ollama serve" in ollama.fix.lower()

    def test_http_error_propagates(self, tmp_path):
        config = _make_config(tmp_path)
        session = MagicMock()
        session.get.return_value = _FakeResponse({}, status_code=500)
        report = PreflightChecker(config, session=session).run()

        ollama = next(c for c in report.checks if c.name == "ollama_reachable")
        assert ollama.ok is False


# ── Models pulled ───────────────────────────────────────────────────────────


class TestModelsPulled:
    def test_both_present(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text:latest"])
        report = PreflightChecker(config, session=session).run()

        models = next(c for c in report.checks if c.name == "models_pulled")
        assert models.ok is True

    def test_missing_chat_model(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        models = next(c for c in report.checks if c.name == "models_pulled")
        assert models.ok is False
        assert "qwen2.5:3b" in models.message
        assert "ollama pull qwen2.5:3b" in models.fix

    def test_family_prefix_matches_pulled_tag(self, tmp_path):
        """If config says 'qwen2.5' and Ollama has 'qwen2.5:3b', it should pass."""
        config = _make_config(tmp_path)
        config.cognition.model_llm_local = "qwen2.5"
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        models = next(c for c in report.checks if c.name == "models_pulled")
        assert models.ok is True

    def test_missing_both_when_ollama_down(self, tmp_path):
        config = _make_config(tmp_path)
        session = _failing_session(ConnectionRefusedError())
        report = PreflightChecker(config, session=session).run()

        models = next(c for c in report.checks if c.name == "models_pulled")
        assert models.ok is False
        assert "ollama_reachable" in models.fix.lower()


# ── Vault accessibility ─────────────────────────────────────────────────────


class TestVaultAccess:
    def test_happy_path(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        vault = next(c for c in report.checks if c.name == "vault_accessible")
        assert vault.ok is True

    def test_missing_directory(self, tmp_path):
        config = _make_config(tmp_path)
        config.vault.path = str(tmp_path / "nonexistent")
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        vault = next(c for c in report.checks if c.name == "vault_accessible")
        assert vault.ok is False
        assert "does not exist" in vault.message

    def test_path_is_file_not_dir(self, tmp_path):
        config = _make_config(tmp_path)
        not_a_dir = tmp_path / "it-is-a-file.txt"
        not_a_dir.write_text("hi")
        config.vault.path = str(not_a_dir)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        vault = next(c for c in report.checks if c.name == "vault_accessible")
        assert vault.ok is False
        assert "not a directory" in vault.message


# ── Git readiness ───────────────────────────────────────────────────────────


class TestGitReady:
    def test_skipped_when_auto_commit_off(self, tmp_path):
        config = _make_config(tmp_path, auto_commit=False)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()
        assert not any(c.name == "git_ready" for c in report.checks)

    def test_warning_when_no_repo_and_auto_commit(self, tmp_path):
        config = _make_config(tmp_path, auto_commit=True)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()

        git = next(c for c in report.checks if c.name == "git_ready")
        assert git.ok is False
        # Warnings don't block startup, just surface the fix.
        assert git.severity == "warning"
        assert "git" in git.fix
        # Missing git alone should NOT mark the whole report as failed.
        assert report.ok is True
        assert report.has_warnings is True

    def test_override_forces_check(self, tmp_path):
        config = _make_config(tmp_path, auto_commit=False)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run(check_git=True)
        assert any(c.name == "git_ready" for c in report.checks)

    def test_override_suppresses_check(self, tmp_path):
        config = _make_config(tmp_path, auto_commit=True)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run(check_git=False)
        assert not any(c.name == "git_ready" for c in report.checks)


# ── End-to-end composition ──────────────────────────────────────────────────


class TestAggregate:
    def test_all_green_report_is_ok(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()
        assert report.ok is True
        assert report.errors == []

    def test_as_dict_shape(self, tmp_path):
        config = _make_config(tmp_path)
        session = _session_with_models(["qwen2.5:3b", "nomic-embed-text"])
        report = PreflightChecker(config, session=session).run()
        d = report.as_dict()
        assert d["ok"] is True
        assert {c["name"] for c in d["checks"]} == {
            "ollama_reachable", "models_pulled", "vault_accessible"
        }
