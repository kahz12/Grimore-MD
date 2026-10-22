"""Fixed stress controls are separate from the untouched final evaluation set."""
from pathlib import Path

from pypdf import PdfReader
import pytest

from eval.check_extraction import evaluate, load_manifest
from eval.build_difficult_pdf import build
from eval.benchmark_extraction import _unavailable
from grimore.cognition.eval import load_golden

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval" / "phase4_difficult.yaml"


def test_published_fixture_has_two_digital_pages_and_one_degraded_scan():
    root, documents = load_manifest(MANIFEST)
    reader = PdfReader(root / documents[0]["path"])
    assert len(reader.pages) == 3
    assert [bool((page.extract_text() or "").strip()) for page in reader.pages] == [True, True, False]
    assert reader.metadata.title == "Control de extracción compleja del Humedal"
    fonts = [font.get_object() for font in
             reader.pages[0]["/Resources"]["/Font"].get_object().values()
             if "/FontDescriptor" in font.get_object()]
    assert len(fonts) >= 2
    assert all("/FontFile2" in font["/FontDescriptor"].get_object() for font in fonts)


def test_control_expectations_remain_in_denominator_without_ocr():
    report = evaluate(MANIFEST, ocr=False)
    assert report["passed"] is False
    assert {key: value["total"] for key, value in report["totals"].items()} == {
        "reading_order": 3, "table_rows": 5, "page_anchors": 6,
    }
    # The interleaved columns interrupt one digital passage as well.
    assert report["totals"]["page_anchors"] == {"passed": 3, "total": 6}
    assert report["documents"][0]["extraction_issues"]


def test_expected_answers_are_three_separate_tuning_cases_with_physical_pages():
    cases = load_golden(ROOT / "eval" / "phase4_difficult_answers.yaml")
    assert len(cases) == 3
    assert {case.split for case in cases} == {"tuning"}
    assert {case.difficulty for case in cases} == {"multi-column", "wrapped-table", "degraded-scan"}
    assert [case.expected_evidence[0]["locator"] for case in cases] == ["p.1", "p.2", "p.3"]


def test_builder_refuses_existing_files_before_loading_optional_authoring_tools(tmp_path):
    existing = tmp_path / "source.pdf"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(existing)
    assert existing.read_bytes() == b"preserve"


def test_optional_pdfplumber_recovers_all_digital_expectations_without_ocr():
    pytest.importorskip("pdfplumber")
    report = evaluate(MANIFEST, pdf_engine="pdfplumber")
    assert report["totals"] == {
        "reading_order": {"passed": 2, "total": 3},
        "table_rows": {"passed": 3, "total": 5},
        "page_anchors": {"passed": 4, "total": 6},
    }
    assert report["documents"][0]["extraction_issues"][0]["page"] == 3


def test_optional_real_spanish_ocr_and_layout_meet_all_unchanged_expectations(monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    missing = _unavailable("pdfplumber", True, "spa")
    if missing:
        pytest.skip(", ".join(missing))
    report = evaluate(MANIFEST, pdf_engine="pdfplumber", ocr=True, ocr_language="spa")
    assert report["passed"] is True
    assert report["totals"] == {
        "reading_order": {"passed": 3, "total": 3},
        "table_rows": {"passed": 5, "total": 5},
        "page_anchors": {"passed": 6, "total": 6},
    }
    assert report["documents"][0]["ocr_pages"] == [3]
