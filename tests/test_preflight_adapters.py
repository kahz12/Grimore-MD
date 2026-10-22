"""
Per-adapter preflight check tests.

The preflight runner now emits one ✓/✗ per format the user has listed
in ``[vault].formats``. We verify: (a) shipped adapters pass, (b)
unsupported extensions surface a warning rather than crash, (c) a
missing optional library surfaces an error with an actionable fix line.
"""
from __future__ import annotations



from grimore.utils.config import Config, VaultConfig
from grimore.utils.preflight import PreflightChecker, PreflightReport


def _checker_with_formats(*formats: str) -> PreflightChecker:
    cfg = Config(vault=VaultConfig(path=".", formats=list(formats)))
    # Ollama probe is irrelevant to adapter checks — short-circuit it.
    return PreflightChecker(cfg, ollama_host="http://127.0.0.1:11434")


class TestAdapterPreflight:
    def test_shipped_adapters_pass(self):
        checker = _checker_with_formats("md", "txt", "html", "docx")
        report = PreflightReport()
        checker._check_adapters(report)
        for name in ("adapter:md", "adapter:txt", "adapter:html", "adapter:docx"):
            row = next(c for c in report.checks if c.name == name)
            assert row.ok, f"{name} should be OK but: {row.message}"

    def test_unknown_extension_is_warning_not_error(self):
        checker = _checker_with_formats("md", "xyz")
        report = PreflightReport()
        checker._check_adapters(report)
        row = next(c for c in report.checks if c.name == "adapter:xyz")
        assert not row.ok
        assert row.severity == "warning"
        assert "xyz" in row.message
        assert row.fix  # actionable hint present

    def test_missing_library_for_html_surfaces_actionable_error(self, monkeypatch):
        # Simulate bs4 not installed by forcing the probe to fail.
        monkeypatch.setattr(
            PreflightChecker, "_probe_html",
            staticmethod(lambda: "No module named 'bs4'"),
        )
        # Reset the cached probe table so the monkeypatched probe wins.
        PreflightChecker._ADAPTER_PROBES = {}

        checker = _checker_with_formats("html")
        report = PreflightReport()
        checker._check_adapters(report)
        row = next(c for c in report.checks if c.name == "adapter:html")
        assert not row.ok
        assert row.severity == "error"
        assert "bs4" in row.message
        assert "pip install" in row.fix

        # Cleanup — clear the cache so subsequent tests get the real probe.
        PreflightChecker._ADAPTER_PROBES = {}

    def test_empty_formats_falls_back_to_md(self):
        checker = _checker_with_formats()  # default applied: ["md"]
        report = PreflightReport()
        checker._check_adapters(report)
        row = next(c for c in report.checks if c.name == "adapter:md")
        assert row.ok


class TestOcrPreflight:
    def test_missing_poppler_and_invalid_timeout_are_actionable(self, monkeypatch):
        from grimore.utils import preflight

        checker = _checker_with_formats("pdf")
        checker.config.ingest.ocr = True
        checker.config.ingest.ocr_timeout_s = 0
        monkeypatch.setattr(checker, "_probe_module", lambda _module: None)
        monkeypatch.setattr(
            preflight, "_which",
            lambda name: "/usr/bin/tesseract" if name == "tesseract" else None,
        )

        report = PreflightReport()
        checker._check_ingest_engines(report)

        timeout = next(row for row in report.checks if row.name == "ocr_timeout_s")
        ocr = next(row for row in report.checks if row.name == "ocr")
        assert not timeout.ok and "positive integer" in timeout.message
        assert not ocr.ok and "Poppler pdfinfo" in ocr.message
        assert "Poppler renderer" in ocr.message
        assert "poppler-utils" in ocr.fix

    def test_ocr_ready_when_all_dependencies_and_timeout_are_valid(self, monkeypatch):
        from grimore.utils import preflight

        checker = _checker_with_formats("pdf")
        checker.config.ingest.ocr = True
        checker.config.ingest.ocr_timeout_s = 15
        monkeypatch.setattr(checker, "_probe_module", lambda _module: None)
        monkeypatch.setattr(preflight, "_which", lambda _name: "/usr/bin/tool")
        monkeypatch.setattr(preflight, "check_ocr_languages", lambda *_args: None)

        report = PreflightReport()
        checker._check_ingest_engines(report)

        assert [(row.name, row.ok) for row in report.checks] == [("ocr", True)]

    def test_requested_language_data_must_be_installed(self, monkeypatch):
        from grimore.utils import preflight

        checker = _checker_with_formats("pdf")
        checker.config.ingest.ocr = True
        checker.config.ingest.ocr_language = "spa+eng"
        monkeypatch.setattr(checker, "_probe_module", lambda _module: None)
        monkeypatch.setattr(preflight, "_which", lambda _name: "/usr/bin/tool")

        def check(executable, language):
            assert language == "spa+eng"
            return "Tesseract language data not installed: spa"

        monkeypatch.setattr(preflight, "check_ocr_languages", check)
        report = PreflightReport()
        checker._check_ingest_engines(report)
        row = next(row for row in report.checks if row.name == "ocr")
        assert not row.ok
        assert "spa" in row.message
        assert "tesseract-ocr-spa" in row.fix

    def test_invalid_language_is_rejected_before_probing(self, monkeypatch):
        from grimore.utils import preflight

        checker = _checker_with_formats("pdf")
        checker.config.ingest.ocr = True
        checker.config.ingest.ocr_language = "spa --psm 6"
        monkeypatch.setattr(checker, "_probe_module", lambda _module: None)
        monkeypatch.setattr(preflight, "_which", lambda _name: "/usr/bin/tool")
        monkeypatch.setattr(preflight, "check_ocr_languages", lambda *_args: 1 / 0)
        report = PreflightReport()
        checker._check_ingest_engines(report)
        assert not report.ok
        assert any(row.name == "ocr_language" and not row.ok for row in report.checks)
