"""Reproducibility checks for the isolated extraction cost harness."""
from __future__ import annotations

import hashlib
import shutil
import subprocess

import pytest
import yaml

from eval import benchmark_extraction
from eval.check_extraction import load_manifest


def test_pypdf_profile_records_quality_time_and_memory():
    report = benchmark_extraction.benchmark(
        benchmark_extraction.DEFAULT_MANIFEST,
        engines=["pypdf"], repeat=1, include_legacy=True,
        profile_label="test-host",
    )

    assert report["profile"]["label"] == "test-host"
    assert report["profile"]["architecture"]
    assert report["version"] == 2
    assert report["inputs_unchanged"] is True
    assert report["inputs"]["manifest_sha256"] == hashlib.sha256(
        benchmark_extraction.DEFAULT_MANIFEST.read_bytes(),
    ).hexdigest()
    assert report["inputs"]["documents"][0]["id"] == "humedal-pdf"
    assert report["settings"]["engines"] == ["pypdf"]
    assert report["settings"]["ocr_language"] == "eng"
    current, legacy = report["results"]
    assert current["status"] == legacy["status"] == "ok"
    assert current["quality"]["totals"]["reading_order"] == {"passed": 3, "total": 3}
    assert legacy["quality"]["totals"]["reading_order"] == {"passed": 0, "total": 3}
    assert current["wall_s_median"] > 0
    assert current["peak_rss_mb_median"] is None or current["peak_rss_mb_median"] > 0
    assert current["quality_consistent"] is True
    assert current["quality"]["failed_checks"] == []
    assert current["quality"]["page_diagnostics"] == [{
        "document_id": "humedal-pdf", "ocr_pages": [], "extraction_issues": [],
    }]
    assert len(legacy["quality"]["failed_checks"]) == 7
    assert "tesseract_languages" in report["profile"]["external_tools"]


def test_changed_corpus_during_measurement_is_reported(tmp_path, monkeypatch):
    root, documents = load_manifest(benchmark_extraction.DEFAULT_MANIFEST)
    source = tmp_path / documents[0]["path"]
    shutil.copyfile(root / documents[0]["path"], source)
    initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "version": 1, "corpus_root": str(tmp_path), "documents": documents,
    }), encoding="utf-8")
    original_run = benchmark_extraction._run_once

    def measure_then_change(*args):
        sample = original_run(*args)
        source.write_bytes(source.read_bytes() + b"\n% changed after extraction\n")
        return sample

    monkeypatch.setattr(benchmark_extraction, "_run_once", measure_then_change)
    report = benchmark_extraction.benchmark(manifest, engines=["pypdf"], repeat=1)

    assert report["results"][0]["status"] == "ok"
    assert report["inputs_unchanged"] is False
    assert report["inputs"]["documents"][0]["sha256"] == initial_hash


def test_missing_optional_engine_is_explicit_skip(monkeypatch):
    original = benchmark_extraction.importlib.util.find_spec
    monkeypatch.setattr(
        benchmark_extraction.importlib.util, "find_spec",
        lambda name: None if name == "pdfplumber" else original(name),
    )

    report = benchmark_extraction.benchmark(
        benchmark_extraction.DEFAULT_MANIFEST,
        engines=["pdfplumber"], repeat=1,
    )

    assert report["results"] == [
        {"variant": "pdfplumber", "status": "skipped", "reason": "pdfplumber extra"},
    ]


def test_scanned_pdf_quality_failure_does_not_corrupt_worker_json():
    report = benchmark_extraction.benchmark(
        benchmark_extraction.DEFAULT_MANIFEST.with_name("phase4_ocr.yaml"),
        engines=["pypdf"], repeat=1,
    )

    result = report["results"][0]
    assert result["status"] == "ok"
    assert result["quality"]["passed"] is False
    assert result["quality"]["totals"]["page_anchors"] == {
        "passed": 0, "total": 2,
    }
    assert len(result["quality"]["failed_checks"]) == 5
    assert all(check["document_id"] == "humedal-scan-page-1"
               for check in result["quality"]["failed_checks"])


def test_missing_external_tool_is_reported_without_launching(monkeypatch):
    monkeypatch.setattr(benchmark_extraction.shutil, "which", lambda _name: None)
    assert benchmark_extraction._tool_probe("tesseract", ["--version"]) == {
        "path": None, "status": "missing", "output": None,
    }


def test_external_tool_probe_uses_stderr_and_limits_time(monkeypatch):
    monkeypatch.setattr(benchmark_extraction.shutil, "which", lambda _name: "/tool")

    def run(command, **kwargs):
        assert command == ["/tool", "-v"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 5,
                          "check": False}
        return subprocess.CompletedProcess(command, 0, "", "poppler test version\n")

    monkeypatch.setattr(benchmark_extraction.subprocess, "run", run)
    assert benchmark_extraction._tool_probe("pdfinfo", ["-v"]) == {
        "path": "/tool", "status": "ok", "output": "poppler test version",
    }


@pytest.mark.parametrize("failure", ["timeout", "oserror", "exit"])
def test_external_tool_probe_failure_is_not_a_benchmark_failure(monkeypatch, failure):
    monkeypatch.setattr(benchmark_extraction.shutil, "which", lambda _name: "/tool")

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("not executable")
        return subprocess.CompletedProcess(command, 1, "", "cannot run")

    monkeypatch.setattr(benchmark_extraction.subprocess, "run", run)
    probe = benchmark_extraction._tool_probe("pdfinfo", ["-v"])
    assert probe["status"] == "error"
    assert probe["path"] == "/tool"


def test_missing_requested_language_skips_only_ocr_variant(monkeypatch):
    from grimore.utils import ocr

    monkeypatch.setattr(benchmark_extraction.importlib.util, "find_spec", lambda _name: True)
    monkeypatch.setattr(benchmark_extraction.shutil, "which", lambda _name: "/tool")
    monkeypatch.setattr(ocr, "check_ocr_languages", lambda *_args:
                        "Tesseract language data not installed: spa")
    report = benchmark_extraction.benchmark(
        benchmark_extraction.DEFAULT_MANIFEST, engines=["pypdf"], repeat=1,
        include_ocr=True, ocr_language="spa",
    )
    current, skipped = report["results"]
    assert current["status"] == "ok"
    assert skipped["status"] == "skipped"
    assert "spa" in skipped["reason"]
    assert report["settings"]["ocr_language"] == "spa"


@pytest.mark.parametrize("repeat", [0, 21])
def test_repeat_budget_is_validated(repeat):
    with pytest.raises(ValueError, match="repeat"):
        benchmark_extraction.benchmark(
            benchmark_extraction.DEFAULT_MANIFEST,
            engines=["pypdf"], repeat=repeat,
        )
