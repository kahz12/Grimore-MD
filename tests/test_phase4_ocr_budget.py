"""Fixed coverage expectations and subprocess propagation for OCR budgets."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from eval import benchmark_extraction, check_extraction
from eval.build_ocr_budget_pdf import DEFAULT_OUTPUT, SCANNED, build
from grimore.ingest.adapters import pdf

MANIFEST = Path(__file__).resolve().parents[1] / "eval" / "phase4_ocr_budget.yaml"


def test_budget_fixture_repeats_scan_without_text_layer_and_is_reproducible(tmp_path):
    pytest.importorskip("PIL")  # pypdf needs Pillow to decode the page images
    pages = PdfReader(DEFAULT_OUTPUT).pages
    assert len(pages) == 3
    source_image = PdfReader(SCANNED).pages[0].images[0].data
    assert all(page.images[0].data == source_image for page in pages)
    assert not any((page.extract_text() or "").strip() for page in pages)
    first = build(tmp_path / "first.pdf")
    assert first.read_bytes() == DEFAULT_OUTPUT.read_bytes()
    assert build(tmp_path / "second.pdf").read_bytes() == first.read_bytes()
    with pytest.raises(FileExistsError):
        build(first)
    assert first.read_bytes() == DEFAULT_OUTPUT.read_bytes()


def test_budget_keeps_all_fifteen_fixed_expectations_in_denominators(monkeypatch):
    recognise = lambda *_args: pdf._OcrResult(
        "Informe de restauración del Humedal\nResumen ejecutivo\n"
        "El área de intervención aprobada es de 37,5 hectáreas\n"
        "Datos de referencia\nFósforo total 0,18 mg/L Menos de 0,10 mg/L\n"
        "Cobertura de jacinto invasor 8 ha Menos de 0,5 ha\nCorpus de evaluación Grimore"
    )
    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    for limit, count in ((1, 1), (2, 2), (0, 3)):
        report = check_extraction.evaluate(MANIFEST, ocr=True, ocr_max_pages=limit)
        assert report["ocr_max_pages"] == limit
        assert report["totals"] == {
            "reading_order": {"passed": count, "total": 3},
            "table_rows": {"passed": count * 2, "total": 6},
            "page_anchors": {"passed": count * 2, "total": 6},
        }
        assert report["passed"] is (limit == 0)
        doc = report["documents"][0]
        assert doc["ocr_pages"] == list(range(1, count + 1))
        assert [issue["page"] for issue in doc["extraction_issues"]
                if issue.get("reason") == "ocr_page_limit"] == list(range(count + 1, 4))


def test_evaluator_cli_records_limit_and_keeps_json_clean(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "check_extraction", "--manifest", str(MANIFEST), "--ocr", "--ocr-max-pages", "1",
    ])
    monkeypatch.setattr(pdf, "_ocr_page", lambda *_args: pdf._OcrResult("Controlled text"))
    assert check_extraction.main() == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["ocr_max_pages"] == 1
    assert report["documents"][0]["ocr_pages"] == [1]


def test_worker_command_forwards_page_limit(monkeypatch):
    def run(command, **kwargs):
        assert command[command.index("--ocr-max-pages") + 1] == "2"
        assert "--ocr" in command
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess(command, 0, '{"quality": {"ocr_max_pages": 2}}', "")

    monkeypatch.setattr(benchmark_extraction.subprocess, "run", run)
    report = benchmark_extraction._run_once(MANIFEST, "pypdf", True, False, 30, "spa", 2)
    assert report["quality"]["ocr_max_pages"] == 2


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_both_evaluation_entrypoints_reject_invalid_budgets(value):
    with pytest.raises(ValueError, match="ocr_max_pages"):
        check_extraction.evaluate(MANIFEST, ocr_max_pages=value)
    with pytest.raises(ValueError, match="ocr_max_pages"):
        benchmark_extraction.benchmark(MANIFEST, ocr_max_pages=value)


def test_isolated_benchmark_records_budget_and_deliberate_coverage_loss(monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = benchmark_extraction._unavailable("pypdf", True, "spa")
    if missing:
        pytest.skip(", ".join(missing))
    report = benchmark_extraction.benchmark(
        MANIFEST, engines=["pypdf"], repeat=1, include_ocr=True,
        ocr_language="spa", ocr_max_pages=1,
    )
    assert report["settings"]["ocr_max_pages"] == 1
    assert report["inputs_unchanged"] is True
    without_ocr, limited = report["results"]
    assert without_ocr["quality"]["ocr_sections"] == 0
    assert limited["status"] == "ok"
    assert limited["quality"]["passed"] is False
    assert limited["quality"]["ocr_max_pages"] == 1
    assert limited["quality"]["totals"] == {
        "reading_order": {"passed": 1, "total": 3},
        "table_rows": {"passed": 2, "total": 6},
        "page_anchors": {"passed": 2, "total": 6},
    }
    assert limited["quality"]["page_diagnostics"][0]["ocr_pages"] == [1]
    assert len(limited["quality"]["failed_checks"]) == 10
    assert limited["samples"][0]["quality"]["ocr_max_pages"] == 1
