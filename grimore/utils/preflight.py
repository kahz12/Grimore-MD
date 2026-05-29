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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_which = shutil.which

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
        self._check_adapters(report)
        self._check_ingest_engines(report)

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

    # Per-format adapter health. Keyed by the extension the user puts in
    # ``[vault].formats``; value is (probe_callable, install_hint). The
    # probe imports the underlying library and returns None on success
    # or a short reason string on failure.
    _ADAPTER_PROBES: dict[str, tuple] = {}

    @staticmethod
    def _probe_md() -> Optional[str]:
        try:
            import frontmatter  # noqa: F401
        except ImportError as e:
            return str(e)
        return None

    @staticmethod
    def _probe_txt() -> Optional[str]:
        # Stdlib-only. Always available.
        return None

    @staticmethod
    def _probe_html() -> Optional[str]:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError as e:
            return str(e)
        return None

    @staticmethod
    def _probe_docx() -> Optional[str]:
        # Pure-stdlib adapter (zipfile + xml.etree). No third-party deps.
        return None

    @staticmethod
    def _probe_pdf() -> Optional[str]:
        try:
            import pypdf  # noqa: F401
        except ImportError as e:
            return str(e)
        return None

    @staticmethod
    def _probe_epub() -> Optional[str]:
        # Pure-stdlib adapter (zipfile + xml.etree); only needs bs4 which
        # the HTML probe already covers. Defer to that check via _probe_html.
        return PreflightChecker._probe_html()

    @staticmethod
    def _probe_rtf() -> Optional[str]:
        try:
            from striprtf.striprtf import rtf_to_text  # noqa: F401
        except ImportError as e:
            return str(e)
        return None

    @staticmethod
    def _probe_odt() -> Optional[str]:
        # Pure-stdlib adapter (zipfile + xml.etree). No third-party deps.
        return None

    @staticmethod
    def _probe_doc() -> Optional[str]:
        # Legacy .doc needs the antiword binary on PATH. We re-export the
        # adapter's own resolver so the two paths can't drift apart.
        try:
            from grimore.ingest.adapters.doc import antiword_available
        except ImportError as e:  # pragma: no cover - import path
            return str(e)
        if not antiword_available():
            return "antiword binary not found on PATH"
        return None

    @classmethod
    def _adapter_probes(cls) -> dict[str, tuple]:
        # Lazily populated so the dict literal at class-scope doesn't
        # have to reference unbound classmethods.
        if not cls._ADAPTER_PROBES:
            cls._ADAPTER_PROBES = {
                "md":   (cls._probe_md,   "pip install python-frontmatter"),
                "txt":  (cls._probe_txt,  ""),
                "html": (cls._probe_html, "pip install beautifulsoup4"),
                "htm":  (cls._probe_html, "pip install beautifulsoup4"),
                "docx": (cls._probe_docx, ""),
                "pdf":  (cls._probe_pdf,  "pip install pypdf"),
                "epub": (cls._probe_epub, "pip install beautifulsoup4"),
                "rtf":  (cls._probe_rtf,  "pip install striprtf"),
                "odt":  (cls._probe_odt,  ""),
                "doc":  (
                    cls._probe_doc,
                    "Install antiword (Linux: `apt install antiword`; "
                    "Termux: `pkg install antiword`).",
                ),
            }
        return cls._ADAPTER_PROBES

    def _check_adapters(self, report: PreflightReport) -> None:
        """One ✓/✗ per enabled format in ``[vault].formats``.

        Built-ins with no third-party dep (txt, docx) are always ✓.
        Formats whose adapter isn't shipped yet surface a warning so the
        user knows the extension is configured but the loader will skip
        it.
        """
        formats = list(self.config.vault.formats) or ["md"]
        probes = self._adapter_probes()

        # Import adapters package to populate the registry before we
        # query supported_extensions. Cheap and idempotent.
        try:
            from grimore.ingest.adapters import supported_extensions
            registered = supported_extensions()
        except Exception as exc:  # pragma: no cover - import path is exercised
            report.add(CheckResult(
                name="adapters_registry",
                ok=False,
                severity="error",
                message=f"Could not import the adapter registry: {exc!r}.",
                fix="Reinstall Grimore (`pip install -e .`) and re-run preflight.",
            ))
            return

        for ext in formats:
            key = ext.lower().lstrip(".")
            if not key:
                continue
            check_name = f"adapter:{key}"

            if key not in registered:
                report.add(CheckResult(
                    name=check_name,
                    ok=False,
                    severity="warning",
                    message=(
                        f"No adapter registered for .{key} files. Documents "
                        "with this extension will be skipped during scan."
                    ),
                    fix=(
                        f"Remove '{key}' from `[vault].formats` in grimore.toml, "
                        "or wait for a later phase that ships this adapter."
                    ),
                ))
                continue

            probe, install_hint = probes.get(key, (None, ""))
            if probe is None:
                # Adapter is registered but we have no health probe — assume
                # OK rather than warn (defensive default for future adapters).
                report.add(CheckResult(
                    name=check_name,
                    ok=True,
                    message=f"Adapter for .{key} ready.",
                ))
                continue

            failure = probe()
            if failure is None:
                report.add(CheckResult(
                    name=check_name,
                    ok=True,
                    message=f"Adapter for .{key} ready.",
                ))
            else:
                fix = install_hint or "Reinstall the missing dependency."
                report.add(CheckResult(
                    name=check_name,
                    ok=False,
                    severity="error",
                    message=f"Adapter for .{key} unavailable: {failure}",
                    fix=fix,
                ))

    def _check_ingest_engines(self, report: PreflightReport) -> None:
        """Probe opt-in ingest engines (alternative PDF engines, OCR).

        Only fires when the user has actually selected them in
        ``[ingest]``. Probes that nothing is configured for stay silent
        so a default install doesn't surface irrelevant checks.
        """
        ingest = getattr(self.config, "ingest", None)
        if ingest is None:
            return

        # Alternative PDF engine — pypdf is the always-available default,
        # so we only probe when the user has switched it.
        engine = (getattr(ingest, "pdf_engine", "pypdf") or "pypdf").lower()
        if engine == "pdfplumber":
            self._probe_optional(
                report,
                name="pdf_engine:pdfplumber",
                module="pdfplumber",
                fix="pip install 'grimore[pdf-plumber]'",
            )
        elif engine == "pymupdf":
            # Try both the modern and legacy import names — older releases
            # ship as ``fitz``. ``X and Y`` short-circuits to None on the
            # first success, or to the last error string when both fail.
            failure = self._probe_module("pymupdf") and self._probe_module("fitz")
            report.add(CheckResult(
                name="pdf_engine:pymupdf",
                ok=failure is None,
                severity="error",
                message=(
                    f"PyMuPDF unavailable: {failure}" if failure
                    else "PyMuPDF engine ready (AGPL — verify licence compatibility)."
                ),
                fix=(
                    "pip install 'grimore[pdf-mupdf]'  # AGPL-3.0"
                    if failure else ""
                ),
            ))
        elif engine not in ("pypdf", ""):
            report.add(CheckResult(
                name=f"pdf_engine:{engine}",
                ok=False,
                severity="error",
                message=f"Unknown pdf_engine {engine!r}.",
                fix=(
                    "Set [ingest].pdf_engine to one of: pypdf, pdfplumber, pymupdf."
                ),
            ))

        # OCR fallback — opt-in. Needs both the Python wheels and the
        # tesseract binary on PATH. Either missing is an error because
        # the user explicitly asked for OCR; silent fallback would mask
        # a configuration mistake.
        if bool(getattr(ingest, "ocr", False)):
            wheel_missing = self._probe_module("pytesseract") or self._probe_module("pdf2image")
            binary_missing = None if _which("tesseract") else "tesseract binary not on PATH"
            if wheel_missing or binary_missing:
                fix_parts = []
                if wheel_missing:
                    fix_parts.append("pip install 'grimore[ocr]'")
                if binary_missing:
                    fix_parts.append(
                        "Install tesseract (Linux: `apt install tesseract-ocr`; "
                        "Termux: `pkg install tesseract`)."
                    )
                report.add(CheckResult(
                    name="ocr",
                    ok=False,
                    severity="error",
                    message=(
                        f"OCR enabled but unavailable: "
                        f"{wheel_missing or binary_missing}"
                    ),
                    fix="\n".join(fix_parts),
                ))
            else:
                report.add(CheckResult(
                    name="ocr",
                    ok=True,
                    message="OCR fallback ready (tesseract + pytesseract).",
                ))

        # Magic-byte sniffer — opt-in. Off by default, so absence of the
        # ``python-magic`` extra is silent. When the user has enabled it
        # we want a single actionable ✗/✓ so they know whether libmagic
        # is wired up correctly.
        if bool(getattr(ingest, "sniff_magic", False)):
            try:
                from grimore.ingest.sniffer import sniff_available
            except Exception as exc:  # pragma: no cover - import path
                report.add(CheckResult(
                    name="sniff_magic",
                    ok=False,
                    severity="error",
                    message=f"Could not load the sniffer module: {exc!r}.",
                    fix="pip install 'grimore[sniff]'",
                ))
                return
            if sniff_available():
                report.add(CheckResult(
                    name="sniff_magic",
                    ok=True,
                    message="Magic-byte sniffer ready (python-magic + libmagic).",
                ))
            else:
                report.add(CheckResult(
                    name="sniff_magic",
                    ok=False,
                    severity="error",
                    message=(
                        "sniff_magic is enabled but python-magic / libmagic "
                        "is not available."
                    ),
                    fix=(
                        "pip install 'grimore[sniff]'\n"
                        "and ensure libmagic is installed (Linux: "
                        "`apt install libmagic1`; Termux: `pkg install file`; "
                        "Windows: `pip install python-magic-bin`)."
                    ),
                ))

    @staticmethod
    def _probe_module(module: str) -> Optional[str]:
        """Import probe shared by the optional-engine checks."""
        try:
            __import__(module)
        except ImportError as e:
            return str(e)
        return None

    def _probe_optional(
        self, report: PreflightReport, *, name: str, module: str, fix: str,
    ) -> None:
        failure = self._probe_module(module)
        if failure is None:
            report.add(CheckResult(
                name=name, ok=True,
                message=f"Optional engine {module} ready.",
            ))
        else:
            report.add(CheckResult(
                name=name, ok=False, severity="error",
                message=f"Optional engine {module} unavailable: {failure}",
                fix=fix,
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
