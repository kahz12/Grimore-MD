"""Mixed PDFs keep physical anchors and never OCR already-readable pages."""
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

from eval.build_mixed_pdf import DEFAULT_OUTPUT, build
from eval.check_extraction import evaluate
from eval.benchmark_extraction import _unavailable
from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.pdf import PdfAdapter

MANIFEST = Path(__file__).resolve().parents[1] / "eval" / "phase4_mixed.yaml"


def test_mixed_fixture_has_one_scanned_middle_page():
    pages = PdfReader(DEFAULT_OUTPUT).pages
    assert len(pages) == 3
    assert "Diseño de seguimiento" in pages[0].extract_text()
    assert not (pages[1].extract_text() or "").strip()
    assert "Decisiones y alertas de revisión" in pages[2].extract_text()


def test_fixture_builder_is_reproducible_and_refuses_overwrite(tmp_path):
    rebuilt = build(tmp_path / "mixed.pdf")
    expected = rebuilt.read_bytes()
    assert build(tmp_path / "second.pdf").read_bytes() == expected
    assert [bool((page.extract_text() or "").strip())
            for page in PdfReader(rebuilt).pages] == [True, False, True]
    with pytest.raises(FileExistsError):
        build(rebuilt)
    assert rebuilt.read_bytes() == expected


def test_mixed_pdf_without_ocr_preserves_readable_pages_and_reports_gap():
    report = evaluate(MANIFEST)
    assert report["passed"] is False
    assert report["totals"] == {
        "reading_order": {"passed": 2, "total": 3},
        "table_rows": {"passed": 2, "total": 4},
        "page_anchors": {"passed": 4, "total": 6},
    }
    doc = report["documents"][0]
    assert doc["ocr_pages"] == []
    assert doc["extraction_issues"] == [{
        "code": "page_text_unavailable", "page": 2, "engine": "pypdf",
        "ocr_requested": False, "recovered": False,
    }]


@pytest.mark.parametrize("recovered", [True, False])
def test_ocr_only_visits_gap_and_leaves_digital_sections_unchanged(monkeypatch, recovered):
    from grimore.ingest.adapters import pdf

    options = AdapterOptions(vault_root=DEFAULT_OUTPUT.parent)
    original = PdfAdapter().extract(DEFAULT_OUTPUT, options=options)
    calls = []

    def recognise(path, page, timeout, language):
        calls.append((page, language))
        return pdf._OcrResult("Scanned text" if recovered else "",
                              None if recovered else "ocr_recognition_failed")

    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    extracted = PdfAdapter(ocr=True, ocr_language="spa").extract(DEFAULT_OUTPUT, options=options)
    assert calls == [(2, "spa")]
    assert [s for s in extracted.sections if s.page != 2] == original.sections
    assert [s.page for s in extracted.sections] == ([1, 2, 3] if recovered else [1, 3])
    assert extracted.issues[0].page == 2
    assert extracted.issues[0].recovered is recovered
    if recovered:
        assert extracted.sections[1].heading == "(ocr)"
    else:
        assert extracted.issues[0].reason == "ocr_recognition_failed"


@pytest.mark.parametrize("engine", ["pypdf", "pdfplumber"])
def test_optional_real_ocr_meets_mixed_expectations_on_physical_pages(engine, monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = _unavailable(engine, True, "spa")
    if missing:
        pytest.skip(", ".join(missing))
    report = evaluate(MANIFEST, pdf_engine=engine, ocr=True, ocr_language="spa")
    assert report["passed"] is True
    assert report["totals"] == {
        "reading_order": {"passed": 3, "total": 3},
        "table_rows": {"passed": 4, "total": 4},
        "page_anchors": {"passed": 6, "total": 6},
    }
    doc = report["documents"][0]
    assert doc["sections"] == 3
    assert doc["ocr_pages"] == [2]
    assert doc["extraction_issues"] == [{
        "code": "ocr_used", "page": 2, "engine": engine,
        "ocr_requested": True, "recovered": True,
    }]


def test_printed_folio_cannot_override_physical_anchor(tmp_path):
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest["corpus_root"] = str(DEFAULT_OUTPUT.parent)
    # The first PDF page has a printed footer 'Página 2'; its text must
    # nevertheless be anchored to physical page 1, not that printed folio.
    manifest["documents"][0]["page_anchors"][0]["page"] = 2
    altered = tmp_path / "wrong-page.yaml"
    altered.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    report = evaluate(altered)
    assert report["totals"]["page_anchors"] == {"passed": 3, "total": 6}
