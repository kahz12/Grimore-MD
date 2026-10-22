"""Regression checks for the controlled multi-page PDF extraction fixture."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

from eval.check_extraction import evaluate, load_manifest
from eval.benchmark_extraction import _unavailable
from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.pdf import PdfAdapter


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval" / "phase4_extraction.yaml"
SOURCE = ROOT / "test_vault" / "test_vault" / "informe_humedal_la_esperanza.pdf"
SCAN = ROOT / "test_vault" / "test_vault" / "informe_humedal_escaneado.pdf"
OCR_MANIFEST = ROOT / "eval" / "phase4_ocr.yaml"


def test_pypdf_preserves_visual_order_table_rows_and_page_anchors():
    report = evaluate(MANIFEST)

    assert report["passed"] is True
    assert report["totals"] == {
        "reading_order": {"passed": 3, "total": 3},
        "table_rows": {"passed": 4, "total": 4},
        "page_anchors": {"passed": 6, "total": 6},
    }


def test_legacy_pypdf_baseline_remains_reproducible():
    report = evaluate(MANIFEST, legacy_pypdf_plain=True)

    assert report["passed"] is False
    assert report["totals"] == {
        "reading_order": {"passed": 0, "total": 3},
        "table_rows": {"passed": 0, "total": 4},
        "page_anchors": {"passed": 6, "total": 6},
    }


def test_optional_pdfplumber_preserves_fixed_layout_expectations():
    pytest.importorskip("pdfplumber")

    report = evaluate(MANIFEST, pdf_engine="pdfplumber")

    assert report["passed"] is True
    assert report["totals"] == {
        "reading_order": {"passed": 3, "total": 3},
        "table_rows": {"passed": 4, "total": 4},
        "page_anchors": {"passed": 6, "total": 6},
    }


def test_pdf_page_one_keeps_title_before_footer_and_cells_on_one_line():
    document = PdfAdapter().extract(
        SOURCE, options=AdapterOptions(vault_root=SOURCE.parent),
    )

    assert [section.page for section in document.sections] == [1, 2, 3]
    first_page = document.sections[0].text
    assert first_page.index("Informe de restauración") < first_page.index(
        "Corpus de evaluación Grimore"
    )
    assert any(
        all(cell in line for cell in ("Fósforo total", "0,18 mg/L", "Menos de 0,10 mg/L"))
        for line in first_page.splitlines()
    )


def test_evaluator_fails_when_passage_is_assigned_to_wrong_page(tmp_path):
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest["corpus_root"] = str(SOURCE.parent)
    manifest["documents"][0]["page_anchors"][0]["page"] = 3
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    report = evaluate(altered)

    assert report["passed"] is False
    assert report["totals"]["page_anchors"] == {"passed": 5, "total": 6}


def test_manifest_rejects_document_path_outside_corpus(tmp_path):
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest["corpus_root"] = str(SOURCE.parent)
    manifest["documents"][0]["path"] = "../outside.pdf"
    altered = tmp_path / "outside.yaml"
    altered.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes vault"):
        load_manifest(altered)


def test_scanned_fixture_is_image_only_and_reports_missing_text():
    assert len(PdfReader(SCAN).pages) == 1
    assert not (PdfReader(SCAN).pages[0].extract_text() or "").strip()

    document = PdfAdapter(ocr=False).extract(
        SCAN, options=AdapterOptions(vault_root=SCAN.parent),
    )
    assert document.sections == []
    assert [(issue.code, issue.page, issue.recovered) for issue in document.issues] == [
        ("page_text_unavailable", 1, False),
    ]


def test_scanned_fixture_fails_quality_checks_without_ocr():
    report = evaluate(OCR_MANIFEST, ocr=False)

    assert report["passed"] is False
    assert report["totals"] == {
        "reading_order": {"passed": 0, "total": 1},
        "table_rows": {"passed": 0, "total": 2},
        "page_anchors": {"passed": 0, "total": 2},
    }


def test_optional_real_ocr_recovers_auditable_page_without_claiming_quality(monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = _unavailable("pypdf", True)
    if missing:
        pytest.skip(", ".join(missing))

    document = PdfAdapter(ocr=True).extract(
        SCAN, options=AdapterOptions(vault_root=SCAN.parent),
    )

    assert len(document.sections) == 1
    section = document.sections[0]
    assert section.page == 1
    assert section.heading == "(ocr)"
    assert "Resumen ejecutivo" in section.text
    assert [(issue.code, issue.page, issue.recovered) for issue in document.issues] == [
        ("ocr_used", 1, True),
    ]
    # Non-empty OCR is recovery, not proof that every expected passage or
    # table cell was recognised faithfully. Quality is scored by the manifest.


@pytest.mark.parametrize("engine", ["pypdf", "pdfplumber"])
def test_optional_spanish_ocr_meets_unchanged_controlled_expectations(engine, monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = _unavailable(engine, True, "spa")
    if missing:
        pytest.skip(", ".join(missing))

    report = evaluate(OCR_MANIFEST, pdf_engine=engine, ocr=True, ocr_language="spa")

    assert report["ocr_language"] == "spa"
    assert report["ocr_sections"] == 1
    assert report["passed"] is True
    assert report["totals"] == {
        "reading_order": {"passed": 1, "total": 1},
        "table_rows": {"passed": 2, "total": 2},
        "page_anchors": {"passed": 2, "total": 2},
    }
