"""Versioned PDF extraction settings used to invalidate stale indexed text."""
from __future__ import annotations

from grimore.ingest.adapters.base import AdapterOptions
from grimore.utils.ocr import validate_ocr_language, validate_ocr_max_pages

_RETRYABLE_OCR_REASONS = frozenset({
    "ocr_unavailable", "ocr_render_failed", "ocr_recognition_failed",
})
_RETRY_SUFFIX = ":retry-ocr"


def pdf_extraction_profile(config) -> str:
    """Identify settings that can change PDF text or its page anchors.

    Bump the version when the extraction algorithm changes materially. OCR
    timeout matters only while OCR is enabled, so changing an inactive knob
    does not trigger an unnecessary reindex.
    """
    options = AdapterOptions.from_config(config)
    engine = options.pdf_engine or "pypdf"
    ocr = bool(options.ocr)
    version = "v2" if engine == "pdfplumber" else "v1"
    profile = f"pdf-layout-{version}:engine={engine}:ocr={int(ocr)}"
    if ocr:
        profile += f":timeout={options.ocr_timeout_s or 30}"
        language = validate_ocr_language(
            "eng" if options.ocr_language is None else options.ocr_language,
        )
        # Explicit English is equivalent to the historical Tesseract default.
        if language != "eng":
            profile += f":language={language}"
        max_pages = validate_ocr_max_pages(
            0 if options.ocr_max_pages is None else options.ocr_max_pages,
        )
        # Unlimited keeps the historical profile; changing a real budget
        # must re-extract unchanged PDFs, including previously omitted pages.
        if max_pages:
            profile += f":max_pages={max_pages}"
    return profile


def pdf_index_profile(config, extraction: dict | None) -> str:
    """Keep transient OCR failures eligible for the next unchanged-file scan."""
    profile = pdf_extraction_profile(config)
    issues = extraction.get("issues", []) if isinstance(extraction, dict) else []
    if isinstance(issues, list) and any(
        isinstance(issue, dict)
        and not issue.get("recovered")
        and issue.get("reason") in _RETRYABLE_OCR_REASONS
        for issue in issues
    ):
        return profile + _RETRY_SUFFIX
    return profile


def pdf_retry_pending(stored_profile: str | None, config) -> bool:
    """A retry marker is not an extraction-settings change."""
    return stored_profile == pdf_extraction_profile(config) + _RETRY_SUFFIX
