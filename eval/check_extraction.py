"""Evaluate document extraction against fixed layout and locator expectations.

Run from the repository root with ``python eval/check_extraction.py``. The
manifest describes controlled source documents; no model or index is needed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from pypdf import PdfReader

from grimore.ingest.adapters.base import AdapterOptions, ExtractedSection
from grimore.ingest.adapters.pdf import PdfAdapter
from grimore.ingest.parser import MarkdownParser
from grimore.utils.security import SecurityGuard
from grimore.utils.ocr import validate_ocr_language, validate_ocr_max_pages

DEFAULT_MANIFEST = Path(__file__).with_name("phase4_extraction.yaml")
CHECK_KINDS = ("reading_order", "table_rows", "page_anchors")
EVALUATION_SEMANTICS = "fixed-expectations-v2"


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _section_text(note: Any, page: int) -> str:
    return "\n".join(
        section.text for section in note.sections if section.page == page
    )


def _ordered(text: str, passages: list[str]) -> bool:
    content = _normalise(text)
    cursor = 0
    for passage in passages:
        index = content.find(_normalise(passage), cursor)
        if index < 0:
            return False
        cursor = index + len(_normalise(passage))
    return True


def _table_row(text: str, cells: list[str]) -> bool:
    """All expected cells must occur in order on one extracted line."""
    return any(_ordered(line, cells) for line in text.splitlines())


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_passages(value: Any, label: str, *, minimum: int) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} needs at least {minimum} text entries")
    return [_require_text(item, label) for item in value]


def _require_page(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive page number")
    return value


def load_manifest(path: Path) -> tuple[Path, list[dict]]:
    """Load and validate a small, source-controlled evaluation manifest."""
    if path.stat().st_size > 256_000:
        raise ValueError("extraction manifest exceeds 256 KiB")
    data = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "manifest")
    if set(data) != {"version", "corpus_root", "documents"} or data["version"] != 1:
        raise ValueError("manifest requires version: 1, corpus_root, and documents")
    corpus_root = (path.parent / _require_text(data["corpus_root"], "corpus_root")).resolve()
    if not corpus_root.is_dir():
        raise ValueError(f"corpus directory does not exist: {corpus_root}")
    documents = data["documents"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty list")
    seen_ids: set[str] = set()
    for index, raw in enumerate(documents):
        item = _require_mapping(raw, f"documents[{index}]")
        if set(item) != {"id", "path", "reading_order", "table_rows", "page_anchors"}:
            raise ValueError(f"documents[{index}] has missing or unsupported fields")
        item_id = _require_text(item["id"], f"documents[{index}].id")
        if item_id in seen_ids or not re.fullmatch(r"[a-z0-9-]+", item_id):
            raise ValueError(f"duplicate or invalid document id: {item_id}")
        seen_ids.add(item_id)
        source = corpus_root / _require_text(item["path"], f"documents[{index}].path")
        SecurityGuard.resolve_within_vault(source, corpus_root)
        if not source.is_file():
            raise ValueError(f"source document does not exist: {source}")
        for kind in CHECK_KINDS:
            checks = item[kind]
            if not isinstance(checks, list):
                raise ValueError(f"documents[{index}].{kind} must be a list")
            for check_index, raw_check in enumerate(checks):
                check = _require_mapping(raw_check, f"{kind}[{check_index}]")
                _require_page(check.get("page"), f"{kind}[{check_index}].page")
                if kind == "page_anchors":
                    _require_text(check.get("passage"), f"{kind}[{check_index}].passage")
                else:
                    key = "passages" if kind == "reading_order" else "cells"
                    _require_passages(
                        check.get(key), f"{kind}[{check_index}].{key}", minimum=2,
                    )
    return corpus_root, documents


def evaluate(
    manifest: Path,
    *,
    pdf_engine: str = "pypdf",
    ocr: bool = False,
    ocr_language: str = "eng",
    ocr_max_pages: int = 0,
    legacy_pypdf_plain: bool = False,
) -> dict:
    """Score reading order, complete table rows, and exact page anchors."""
    if legacy_pypdf_plain and (pdf_engine != "pypdf" or ocr):
        raise ValueError("legacy plain mode requires pypdf without OCR")
    validate_ocr_language(ocr_language)
    validate_ocr_max_pages(ocr_max_pages)
    root, documents = load_manifest(manifest)
    totals = {kind: {"passed": 0, "total": 0} for kind in CHECK_KINDS}
    results = []
    ocr_sections = 0
    for item in documents:
        source = root / item["path"]
        note = None
        extraction_error = None
        try:
            if source.suffix.casefold() == ".pdf" and legacy_pypdf_plain:
                # Reproduce plain pypdf text extraction (no layout engine, no
                # OCR) as a comparison baseline, without reverting production
                # code or changing the controlled source document.
                reader = PdfReader(str(source))
                note = SimpleNamespace(
                    format="pdf",
                    sections=[
                        ExtractedSection(
                            text=(page.extract_text() or "").strip(),
                            page=index + 1,
                            order=index,
                        )
                        for index, page in enumerate(reader.pages)
                    ],
                )
            elif source.suffix.casefold() == ".pdf":
                note = PdfAdapter(
                    engine=pdf_engine, ocr=ocr, ocr_language=ocr_language,
                    ocr_max_pages=ocr_max_pages,
                ).extract(
                    source, options=AdapterOptions(vault_root=root),
                )
            else:
                note = MarkdownParser().parse_file(source, vault_root=root)
        except (OSError, ValueError) as exc:
            extraction_error = str(exc)
        checks = []
        for kind in CHECK_KINDS:
            for index, expected in enumerate(item[kind]):
                page = expected["page"]
                # A document-level failure must not remove its expectations
                # from the denominator or accidentally reuse the previous note.
                text = _section_text(note, page) if note is not None else ""
                if kind == "reading_order":
                    passed = _ordered(text, expected["passages"])
                elif kind == "table_rows":
                    passed = _table_row(text, expected["cells"])
                else:
                    passed = _normalise(expected["passage"]) in _normalise(text)
                totals[kind]["total"] += 1
                totals[kind]["passed"] += int(passed)
                checks.append({
                    "kind": kind, "index": index, "page": page, "passed": passed,
                })
        if extraction_error is not None:
            results.append({"id": item["id"], "error": extraction_error, "checks": checks})
            continue
        document_ocr_sections = sum(
            section.heading == "(ocr)" for section in note.sections
        )
        ocr_sections += document_ocr_sections
        results.append({
            "id": item["id"], "format": note.format, "checks": checks,
            "sections": len(note.sections),
            "ocr_sections": document_ocr_sections,
            "ocr_pages": [section.page for section in note.sections
                          if section.heading == "(ocr)"],
            "extraction_issues": [issue.as_dict() for issue in getattr(note, "issues", [])],
        })
    return {
        "evaluation_semantics": EVALUATION_SEMANTICS,
        "manifest": str(manifest),
        "pdf_engine": pdf_engine,
        "ocr": ocr,
        "ocr_language": ocr_language,
        "ocr_max_pages": ocr_max_pages,
        "legacy_pypdf_plain": legacy_pypdf_plain,
        "totals": totals,
        "documents": results,
        "ocr_sections": ocr_sections,
        "passed": all(
            score["passed"] == score["total"] and score["total"] > 0
            for score in totals.values()
        ) and all("error" not in result for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-engine", default="pypdf")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr-language", default="eng", help="Tesseract language, e.g. spa+eng")
    parser.add_argument("--ocr-max-pages", type=int, default=0,
                        help="OCR attempts per PDF; 0 means unlimited")
    parser.add_argument(
        "--legacy-pypdf-plain", action="store_true",
        help="Repeat the pre-Phase-4 pypdf extraction for comparison.",
    )
    args = parser.parse_args()
    try:
        # Partial extraction emits warnings; stdout must remain valid JSON
        # for callers comparing explicit OCR budgets programmatically.
        with redirect_stdout(sys.stderr):
            report = evaluate(
                args.manifest.resolve(), pdf_engine=args.pdf_engine, ocr=args.ocr,
                ocr_language=args.ocr_language,
                ocr_max_pages=args.ocr_max_pages,
                legacy_pypdf_plain=args.legacy_pypdf_plain,
            )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
