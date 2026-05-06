"""
Startup validation for Project Grimore.

Fails loudly and actionably *before* the first LLM call instead of mid-scan.
Every check returns a :py:class:`CheckResult` so the CLI can render them all
(even after one fails) and suggest a concrete fix.

Checks performed:

* **ollama_reachable** — ``GET {host}/api/tags`` succeeds within a short timeout.
* **models_pulled**    — both the chat and embedding models appear in that list.
* **vault_accessible** — configured vault path exists and is readable.
* **git_ready**        — only when ``output.auto_commit`` is True; warns, doesn't
  fail, because scans still run without auto-commit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from grimore.output.git_guard import GitGuard
from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """
    Outcome of a single preflight check.

    ``ok`` is the pass/fail bit. ``severity`` is "error" for things that must
    be fixed before Grimore can do useful work and "warning" for things that
    degrade functionality but don't block it (e.g. missing git).
    """
    name: str
    ok: bool
    severity: str = "error"
    message: str = ""
    fix: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
        }


@dataclass
class PreflightReport:
    """Collection of check results — queryable as a whole."""
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def ok(self) -> bool:
        """True only if no check reported severity=error."""
        return all(c.ok or c.severity != "error" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(not c.ok and c.severity == "warning" for c in self.checks)

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and c.severity == "warning"]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
        }


class PreflightChecker:
    """
    Runs the configured preflight checks against a loaded ``Config``.

    The ``session`` parameter is injected for tests; production code uses the
    default shared-retry session from :py:mod:`grimore.utils.http`.
    """

    def __init__(self, config, *, session=None, ollama_host: Optional[str] = None):
        self.config = config
        self.session = session or build_session(total_retries=1, backoff=0.1)
        self.ollama_host = ollama_host or self._resolve_ollama_host()

    def _resolve_ollama_host(self) -> str:
        """Resolve the Ollama URL the same way LLMRouter/Embedder do."""
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            return SecurityGuard.validate_llm_host(
                raw_host, allow_remote=self.config.cognition.allow_remote
            )
        except ValueError as exc:
            # Bubble up via the reachability check rather than crashing here.
            logger.warning("preflight_host_invalid", error=str(exc))
            return raw_host

    def run(self, *, check_git: Optional[bool] = None) -> PreflightReport:
        """
        Execute all checks and return a report. ``check_git`` overrides the
        default (which follows ``output.auto_commit``); mostly useful for the
        ``ask``/``status`` paths that don't write.
        """
        report = PreflightReport()
        tags = self._check_ollama_reachable(report)
        self._check_models_pulled(report, tags)
        self._check_vault_accessible(report)

        want_git = check_git if check_git is not None else self.config.output.auto_commit
        if want_git:
            self._check_git_ready(report)
        return report

    # ── Individual checks ───────────────────────────────────────────────────

    def _check_ollama_reachable(self, report: PreflightReport) -> Optional[list[str]]:
        """Probe /api/tags and return the model-name list (or None on failure)."""
        url = f"{self.ollama_host}/api/tags"
        try:
            resp = self.session.get(url, timeout=3)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            report.add(CheckResult(
                name="ollama_reachable",
                ok=False,
                severity="error",
                message=f"Could not contact Ollama at {self.ollama_host} ({exc.__class__.__name__}).",
                fix=(
                    "Start Ollama with `ollama serve`, or adjust the "
                    "`OLLAMA_HOST` variable if it runs on another machine."
                ),
            ))
            return None

        models = [m.get("name", "") for m in payload.get("models", [])]
        report.add(CheckResult(
            name="ollama_reachable",
            ok=True,
            message=f"Ollama responds at {self.ollama_host} ({len(models)} models loaded).",
        ))
        return models

    def _check_models_pulled(
        self, report: PreflightReport, available: Optional[list[str]]
    ) -> None:
        """Ensure both the chat and embedding models appear in /api/tags."""
        if available is None:
            report.add(CheckResult(
                name="models_pulled",
                ok=False,
                severity="error",
                message="Could not verify models (Ollama unreachable).",
                fix="Resolve the `ollama_reachable` check first.",
            ))
            return

        required = {
            "chat": self.config.cognition.model_llm_local,
            "embeddings": self.config.cognition.model_embeddings_local,
        }
        # Ollama exposes models as "name:tag". A user may configure the bare
        # name ("qwen2.5:3b") or just the family ("qwen2.5"); match on either
        # exact or name-prefix so both work.
        missing = []
        for role, name in required.items():
            if not self._model_available(name, available):
                missing.append((role, name))

        if missing:
            pretty = ", ".join(f"{name} ({role})" for role, name in missing)
            commands = "\n".join(f"  ollama pull {name}" for _role, name in missing)
            report.add(CheckResult(
                name="models_pulled",
                ok=False,
                severity="error",
                message=f"Missing pulled models: {pretty}.",
                fix=f"Run:\n{commands}",
            ))
        else:
            report.add(CheckResult(
                name="models_pulled",
                ok=True,
                message=(
                    f"Models present: {required['chat']} (chat), "
                    f"{required['embeddings']} (embeddings)."
                ),
            ))

    @staticmethod
    def _model_available(name: str, available: list[str]) -> bool:
        """
        Match ``name`` against Ollama's tag list. Exact match wins; otherwise
        accept a tagless config value if any pulled tag shares the family
        name (``qwen2.5`` matches ``qwen2.5:3b``).
        """
        if not name:
            return False
        if name in available:
            return True
        if ":" not in name:
            prefix = name + ":"
            return any(m == name or m.startswith(prefix) for m in available)
        return False

    def _check_vault_accessible(self, report: PreflightReport) -> None:
        """The vault path exists, is a directory, and is readable."""
        path = Path(self.config.vault.path)
        if not path.exists():
            report.add(CheckResult(
                name="vault_accessible",
                ok=False,
                severity="error",
                message=f"The vault {path} does not exist.",
                fix=(
                    f"Create the directory (`mkdir -p {path}`) or update "
                    "`[vault].path` in grimore.toml."
                ),
            ))
            return
        if not path.is_dir():
            report.add(CheckResult(
                name="vault_accessible",
                ok=False,
                severity="error",
                message=f"The path {path} exists but is not a directory.",
                fix="Update `[vault].path` to point to a directory.",
            ))
            return
        if not os.access(path, os.R_OK):
            report.add(CheckResult(
                name="vault_accessible",
                ok=False,
                severity="error",
                message=f"The vault {path} is not readable by the current user.",
                fix=f"Check permissions: `chmod -R u+r {path}`.",
            ))
            return
        report.add(CheckResult(
            name="vault_accessible",
            ok=True,
            message=f"Vault accessible at {path}.",
        ))

    def _check_git_ready(self, report: PreflightReport) -> None:
        """When auto_commit is on, confirm the vault is a git repo."""
        guard = GitGuard(self.config.vault.path)
        if guard.is_repo_ready():
            report.add(CheckResult(
                name="git_ready",
                ok=True,
                message="Git repository detected; auto_commit will work.",
            ))
        else:
            report.add(CheckResult(
                name="git_ready",
                ok=False,
                severity="warning",
                message=(
                    "`auto_commit` is enabled but the vault is not a git repository; "
                    "safety snapshots will not be created."
                ),
                fix=(
                    f"Initialise the repo: `git -C {self.config.vault.path} init` "
                    "or disable `[output].auto_commit`."
                ),
            ))
