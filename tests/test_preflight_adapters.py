"""
Per-adapter preflight check tests.

The preflight runner now emits one ✓/✗ per format the user has listed
in ``[vault].formats``. We verify: (a) shipped adapters pass, (b)
unsupported extensions surface a warning rather than crash, (c) a
missing optional library surfaces an error with an actionable fix line.
"""
from __future__ import annotations

import sys

import pytest

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
