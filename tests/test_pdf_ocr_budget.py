"""A per-PDF OCR attempt budget preserves visible, physical-page gaps."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from grimore.ingest.adapters import pdf
from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.extraction_profile import pdf_extraction_profile, pdf_index_profile
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.utils.config import Config, load_config
from grimore.utils.ocr import validate_ocr_max_pages
from grimore.utils.preflight import PreflightChecker, PreflightReport


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-1.4\ncontrolled engine input")
    monkeypatch.setitem(pdf._ENGINE_DISPATCH, "pypdf", lambda _path: (
        {}, [(1, "Digital first"), (2, ""), (3, "Digital middle"),
             (4, ""), (5, ""), (6, "Digital last")], [5],
    ))
    return path


@pytest.mark.parametrize("limit,attempted", [(0, [2, 4, 5]), (1, [2]), (2, [2, 4]), (8, [2, 4, 5])])
@pytest.mark.parametrize("outcome", ["recovered", "ocr_no_text", "ocr_recognition_failed"])
def test_budget_counts_attempts_not_digital_pages_or_successes(
    source, monkeypatch, limit, attempted, outcome,
):
    recognise = MagicMock(side_effect=lambda _path, page, *_args: pdf._OcrResult(
        f"Recovered page {page}" if outcome == "recovered" else "",
        None if outcome == "recovered" else outcome,
    ))
    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    document = pdf.PdfAdapter(ocr=True, ocr_max_pages=limit).extract(
        source, options=AdapterOptions(vault_root=source.parent),
    )
    assert [call.args[1] for call in recognise.call_args_list] == attempted
    assert {section.page: section.text for section in document.sections if section.page in (1, 3, 6)} == {
        1: "Digital first", 3: "Digital middle", 6: "Digital last",
    }
    assert [section.page for section in document.sections] == (
        sorted([1, 3, 6] + attempted) if outcome == "recovered" else [1, 3, 6]
    )
    issues = {issue.page: issue for issue in document.issues}
    assert set(issues) == {2, 4, 5}
    assert issues[5].code == "page_extraction_failed"
    for page in (2, 4, 5):
        issue = issues[page]
        assert issue.ocr_requested is True
        if page not in attempted:
            assert issue.reason == "ocr_page_limit"
            assert issue.recovered is False
        else:
            assert issue.recovered is (outcome == "recovered")
            assert issue.reason == (None if outcome == "recovered" else outcome)


def test_options_can_remove_constructor_budget_and_do_not_leak_between_calls(source, monkeypatch):
    recognise = MagicMock(return_value="Recovered")
    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    adapter = pdf.PdfAdapter(ocr=True, ocr_max_pages=1)
    for override, expected in ((2, 2), (0, 3), (None, 1)):
        recognise.reset_mock()
        adapter.extract(source, options=AdapterOptions(ocr_max_pages=override))
        assert recognise.call_count == expected


def test_budget_disabled_ocr_never_attempts_or_reports_a_limit(source, monkeypatch):
    recognise = MagicMock(side_effect=AssertionError("OCR is disabled"))
    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    document = pdf.PdfAdapter(ocr_max_pages=1).extract(source, options=AdapterOptions())
    assert all(issue.reason is None and not issue.ocr_requested for issue in document.issues)
    recognise.assert_not_called()


def test_large_scanned_document_stops_after_configured_attempts(source, monkeypatch):
    monkeypatch.setitem(pdf._ENGINE_DISPATCH, "pypdf", lambda _path: (
        {}, [(page, "") for page in range(1, 501)], [],
    ))
    recognise = MagicMock(return_value=pdf._OcrResult("Recovered"))
    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    document = pdf.PdfAdapter(ocr=True, ocr_max_pages=3).extract(source, options=AdapterOptions())
    assert recognise.call_count == 3
    assert [section.page for section in document.sections] == [1, 2, 3]
    assert [issue.page for issue in document.issues if issue.reason == "ocr_page_limit"] == list(range(4, 501))


def test_loaded_budget_reaches_parser_and_diagnostics_without_perpetual_retry(source, monkeypatch):
    config_path = source.parent / "grimore.toml"
    config_path.write_text("[ingest]\nocr = true\nocr_max_pages = 1\n", encoding="utf-8")
    config = load_config(str(config_path))
    monkeypatch.setattr(pdf, "_ocr_page", lambda *_args: pdf._OcrResult("Recovered"))
    note = MarkdownParser().parse_file(source, vault_root=source.parent, config=config)
    diagnostic = extraction_diagnostics(note)
    assert diagnostic["status"] == "partial"
    assert [issue["page"] for issue in diagnostic["issues"] if issue.get("reason") == "ocr_page_limit"] == [4, 5]
    assert pdf_index_profile(config, diagnostic) == pdf_extraction_profile(config)


def test_budget_profile_retains_legacy_unlimited_and_ignores_inactive_setting():
    config = Config()
    inactive = pdf_extraction_profile(config)
    config.ingest.ocr_max_pages = 2
    assert pdf_extraction_profile(config) == inactive
    config.ingest.ocr = True
    assert pdf_extraction_profile(config).endswith(":max_pages=2")
    config.ingest.ocr_max_pages = 3
    assert pdf_extraction_profile(config).endswith(":max_pages=3")
    config.ingest.ocr_max_pages = 0
    assert pdf_extraction_profile(config) == "pdf-layout-v1:engine=pypdf:ocr=1:timeout=30"


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "2", None])
def test_invalid_budget_rejected_before_extraction(source, value):
    with pytest.raises(ValueError, match="ocr_max_pages"):
        validate_ocr_max_pages(value)
    with pytest.raises(ValueError, match="ocr_max_pages"):
        pdf.PdfAdapter(ocr_max_pages=value)
    # None means inherit only in the per-call options, not as a configured value.
    if value is not None:
        with pytest.raises(ValueError, match="ocr_max_pages"):
            pdf.PdfAdapter().extract(source, options=AdapterOptions(ocr_max_pages=value))


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_preflight_reports_invalid_active_budget(value, monkeypatch):
    from grimore.utils import preflight

    config = Config()
    config.ingest.ocr = True
    config.ingest.ocr_max_pages = value
    checker = PreflightChecker(config, ollama_host="http://127.0.0.1:11434")
    monkeypatch.setattr(checker, "_probe_module", lambda _module: None)
    monkeypatch.setattr(preflight, "_which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(preflight, "check_ocr_languages", lambda *_args: None)
    report = PreflightReport()
    checker._check_ingest_engines(report)
    issue = next(row for row in report.checks if row.name == "ocr_max_pages")
    assert not issue.ok and issue.severity == "error"
    assert "0 (unlimited)" in issue.fix
