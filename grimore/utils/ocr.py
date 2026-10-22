"""Validation and bounded checks for OCR settings and language data."""
from __future__ import annotations

import re
import subprocess


def validate_ocr_max_pages(value: int) -> int:
    """Zero preserves unlimited attempts; positive values bound each PDF."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ocr_max_pages must be a non-negative integer (0 means unlimited)")
    return value


def validate_ocr_language(value: str) -> str:
    """Accept installed model identifiers, optionally combined with '+'."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]*(?:\+[A-Za-z][A-Za-z0-9_]*)*", value,
    ):
        raise ValueError("ocr_language must be a language code such as eng, spa or spa+eng")
    return value


def check_ocr_languages(executable: str, language: str) -> str | None:
    """Return an actionable failure instead of hanging on a tool probe."""
    validate_ocr_language(language)
    try:
        result = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Could not inspect Tesseract language data (5 s limit)."
    if result.returncode:
        return "Could not inspect Tesseract language data."
    # Match whole lines; never interpret the human-readable header as a model.
    available = set(result.stdout.splitlines())
    missing = [code for code in language.split("+") if code not in available]
    if missing:
        return "Tesseract language data not installed: " + ", ".join(missing)
    return None
