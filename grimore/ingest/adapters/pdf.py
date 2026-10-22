"""
PDF adapter — pure-Python via ``pypdf`` by default, with optional
``pdfplumber`` and ``pymupdf`` engines selectable through
``[ingest].pdf_engine`` in ``grimore.toml``.

Per-page extraction lets every emitted section carry its 1-based page
number, which the embedding layer threads through into ``embeddings.page``
so the Oracle can render citations as ``[[Title#p.42]]`` rather than a
bare ``[[Title]]``.

Engine selection:

* ``"pypdf"`` (default) — pure-Python, MIT-licensed. Always available.
* ``"pdfplumber"``      — opt-in extra (``pip install grimore[pdf-plumber]``).
                           Better column / table handling for non-prose PDFs.
* ``"pymupdf"``         — opt-in extra (``pip install grimore[pdf-mupdf]``).
                           Best extraction quality but AGPL-3.0; users must
                           acknowledge the license implications.

OCR fallback: when ``[ingest].ocr = true`` and the ``tesseract`` binary
is on PATH (plus the ``ocr`` extra installed), pages whose text layer
extracts to empty are rasterised and run through OCR. Sections produced
by OCR are tagged with ``heading="(ocr)"`` so a reviewer can audit
extracted text for typical OCR artefacts.
``[ingest].ocr_max_pages`` optionally bounds attempts per PDF (0 is unlimited).
Skipped pages retain physical anchors and explicit partial-coverage issues;
readable digital pages never consume the OCR budget.

Limitations (documented, not failures):

* Encrypted PDFs are skipped with a structured ``ValueError`` so the
  daemon never sits idle waiting for a passphrase that won't come.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import ClassVar, NamedTuple, Optional, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
    ExtractionIssue,
    ExtractedSection,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.ocr import validate_ocr_language, validate_ocr_max_pages
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_PDF_BYTES = 100_000_000

_OCR_HEADING_MARKER = "(ocr)"
_SUPPORTED_ENGINES = ("pypdf", "pdfplumber", "pymupdf")

# A "heading-ish" line on page 1: short, mostly capitalised, no trailing
# punctuation. Used only as a title fallback when ``info.title`` is empty.
_HEADING_LINE_RE = re.compile(r"^[A-Z0-9][^.!?]{2,80}$")


def _extract_first_heading(page_text: str) -> Optional[str]:
    """Best-effort heading hint from the first non-empty line of page 1."""
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADING_LINE_RE.match(line):
            return line
        # The first non-empty line, even if not a "heading", beats nothing
        # — but trimmed to a sane length so a 500-char paragraph doesn't
        # become the document title.
        return line[:120]
    return None


def _read_pdf_metadata(reader: PdfReader) -> dict[str, str]:
    """Pull the standard XMP / info-dict fields. Best-effort: corrupt
    metadata never blocks ingest, it just yields an empty dict."""
    out: dict[str, str] = {}
    try:
        info = reader.metadata
    except Exception:
        return out
    if info is None:
        return out
    for key, target in (("/Title", "title"), ("/Author", "author"),
                        ("/Subject", "subject"), ("/Keywords", "keywords"),
                        ("/CreationDate", "created"), ("/ModDate", "modified")):
        val = info.get(key)
        if val:
            try:
                out[target] = str(val).strip()
            except Exception:
                continue
    return out


# ── Per-engine page extractors ────────────────────────────────────────────
#
# Each engine returns ``(metadata, list[(page_no, text)], failed_pages)`` so the main
# ``extract`` body never has to branch on engine while assembling
# ExtractedSections. The 1-based page numbering lives here so every engine
# produces the same anchor shape. Page text is yielded only when non-empty;
# empty pages become OCR candidates in the caller.
# pdfplumber adds a fourth mapping for auditable layout fallback reasons.


def _extract_with_pypdf(
    file_path: Path,
) -> tuple[dict[str, str], list[tuple[int, str]], list[int]]:
    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as e:
        raise ValueError(f"unreadable pdf {file_path}: {e}") from e
    except Exception as e:
        # pypdf can raise plain Exceptions on really mangled files; we
        # swallow them here so one bad PDF doesn't take the scan down.
        raise ValueError(f"failed to open pdf {file_path}: {e}") from e

    if getattr(reader, "is_encrypted", False):
        try:
            if not reader.decrypt(""):
                raise ValueError(f"encrypted pdf {file_path}: passphrase required")
        except Exception as e:
            raise ValueError(f"encrypted pdf {file_path}: {e}") from e

    metadata = _read_pdf_metadata(reader)
    pages: list[tuple[int, str]] = []
    failed_pages: list[int] = []
    for idx, page in enumerate(reader.pages):
        try:
            # Layout mode follows rendered lines instead of content-stream
            # insertion order. It keeps table cells on their visual row and
            # leaves the footer after the body in the controlled PDF corpus.
            # Suppress vertical spacer lines so a long page does not flood
            # the chunker with blank text. Rotated labels remain available.
            text = (page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
                layout_mode_strip_rotated=False,
            ) or "").strip()
        except Exception as e:
            failed_pages.append(idx + 1)
            logger.warning(
                "pdf_page_extract_failed",
                path=str(file_path), page=idx + 1, engine="pypdf", error=str(e),
            )
            text = ""
        pages.append((idx + 1, text))
    return metadata, pages, failed_pages


def _extract_with_pdfplumber(
    file_path: Path,
) -> tuple[dict[str, str], list[tuple[int, str]], list[int], dict[int, str]]:
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError as e:
        raise ValueError(
            f"pdf_engine='pdfplumber' requires the 'pdf-plumber' extra: "
            f"pip install 'grimore[pdf-plumber]' ({e})"
        ) from e

    metadata: dict[str, str] = {}
    pages: list[tuple[int, str]] = []
    failed_pages: list[int] = []
    layout_fallbacks: dict[int, str] = {}
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            info = getattr(pdf, "metadata", None) or {}
            for key, target in (("Title", "title"), ("Author", "author"),
                                ("Subject", "subject"), ("Keywords", "keywords")):
                val = info.get(key)
                if val:
                    metadata[target] = str(val).strip()
            for idx, page in enumerate(pdf.pages):
                try:
                    text = (page.extract_text() or "").strip()
                    if text:
                        from grimore.ingest.pdf_layout import extract_page, LayoutUnavailable

                        try:
                            candidate = extract_page(page, baseline_text=text)
                            if not candidate.strip():
                                raise LayoutUnavailable("layout_text_mismatch")
                            text = candidate
                        except Exception as e:
                            # Never throw away readable text because layout
                            # detection failed. Keep the fallback auditable.
                            reason = str(e) if isinstance(e, LayoutUnavailable) else "layout_failed"
                            layout_fallbacks[idx + 1] = reason
                            logger.warning("pdf_layout_fallback", page=idx + 1,
                                           engine="pdfplumber", reason=reason)
                except Exception as e:
                    failed_pages.append(idx + 1)
                    logger.warning(
                        "pdf_page_extract_failed",
                        path=str(file_path), page=idx + 1,
                        engine="pdfplumber", error=str(e),
                    )
                    text = ""
                pages.append((idx + 1, text))
    except Exception as e:
        raise ValueError(f"pdfplumber failed on {file_path}: {e}") from e
    return metadata, pages, failed_pages, layout_fallbacks


def _extract_with_pymupdf(
    file_path: Path,
) -> tuple[dict[str, str], list[tuple[int, str]], list[int]]:
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        # Older versions ship as `fitz`; try that as a fallback.
        try:
            import fitz as pymupdf  # type: ignore[import-not-found]
        except ImportError as e:
            raise ValueError(
                f"pdf_engine='pymupdf' requires the 'pdf-mupdf' extra (AGPL): "
                f"pip install 'grimore[pdf-mupdf]' ({e})"
            ) from e

    metadata: dict[str, str] = {}
    pages: list[tuple[int, str]] = []
    failed_pages: list[int] = []
    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        raise ValueError(f"pymupdf failed to open {file_path}: {e}") from e

    try:
        if getattr(doc, "is_encrypted", False) and not doc.authenticate(""):
            raise ValueError(f"encrypted pdf {file_path}: passphrase required")
        info = getattr(doc, "metadata", None) or {}
        for key, target in (("title", "title"), ("author", "author"),
                            ("subject", "subject"), ("keywords", "keywords")):
            val = info.get(key)
            if val:
                metadata[target] = str(val).strip()
        for idx, page in enumerate(doc):
            try:
                text = (page.get_text() or "").strip()
            except Exception as e:
                failed_pages.append(idx + 1)
                logger.warning(
                    "pdf_page_extract_failed",
                    path=str(file_path), page=idx + 1,
                    engine="pymupdf", error=str(e),
                )
                text = ""
            pages.append((idx + 1, text))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return metadata, pages, failed_pages


_ENGINE_DISPATCH = {
    "pypdf":      _extract_with_pypdf,
    "pdfplumber": _extract_with_pdfplumber,
    "pymupdf":    _extract_with_pymupdf,
}


class _OcrResult(NamedTuple):
    text: str
    reason: Optional[str] = None


def _ocr_page(file_path: Path, page_no: int, timeout_s: int,
              language: str = "eng") -> _OcrResult:
    """Best-effort OCR for one page with a stable failure reason.

    The function never raises — OCR failures are warnings, not file-level
    errors. Tesseract or its Python bridge missing simply skips the page.
    Both rasterisation and recognition have their own timeout so neither
    external process can wait indefinitely on a malformed page.
    """
    if shutil.which("tesseract") is None:
        return _OcrResult("", "ocr_unavailable")
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
    except ImportError:
        return _OcrResult("", "ocr_unavailable")
    try:
        # pdf2image needs poppler-utils. We constrain to a single page so
        # one bad page in a 500-page scan can't tie up the daemon.
        images = convert_from_path(
            str(file_path), first_page=page_no, last_page=page_no,
            thread_count=1, timeout=timeout_s,
            use_pdftocairo=shutil.which("pdftoppm") is None,
        )
    except Exception as e:
        logger.warning(
            "pdf_ocr_rasterise_failed",
            path=str(file_path), page=page_no, error=str(e),
        )
        return _OcrResult("", "ocr_render_failed")
    if not images:
        return _OcrResult("", "ocr_render_failed")
    try:
        text = pytesseract.image_to_string(images[0], timeout=timeout_s, lang=language)
    except Exception as e:
        logger.warning(
            "pdf_ocr_failed",
            path=str(file_path), page=page_no, error=str(e),
        )
        return _OcrResult("", "ocr_recognition_failed")
    clean_text = (text or "").strip()
    return _OcrResult(clean_text, None if clean_text else "ocr_no_text")


def _positive_ocr_timeout(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("ocr_timeout_s must be a positive integer")
    return value


class PdfAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("pdf",)
    binary: ClassVar[bool] = True
    mutable_frontmatter: ClassVar[bool] = False

    def __init__(self, *, engine: str = "pypdf", ocr: bool = False,
                 ocr_timeout_s: int = 30, ocr_language: str = "eng",
                 ocr_max_pages: int = 0):
        """The adapter is normally instantiated once at import time with
        defaults; the section-aware scan path resolves the per-run engine
        through ``AdapterOptions.pdf_engine`` so config swaps take
        effect on the next scan without re-importing.
        """
        if engine not in _SUPPORTED_ENGINES:
            raise ValueError(
                f"unknown pdf_engine {engine!r}; expected one of "
                f"{_SUPPORTED_ENGINES}"
            )
        self.engine = engine
        self.ocr = ocr
        self.ocr_timeout_s = _positive_ocr_timeout(ocr_timeout_s)
        self.ocr_language = validate_ocr_language(ocr_language)
        self.ocr_max_pages = validate_ocr_max_pages(ocr_max_pages)

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        file_path = Path(path)
        if options.vault_root is not None:
            SecurityGuard.resolve_within_vault(file_path, options.vault_root)

        try:
            stat = file_path.stat()
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        size = stat.st_size
        if size > _MAX_PDF_BYTES:
            logger.warning(
                "pdf_too_large", path=str(file_path), size=size, max=_MAX_PDF_BYTES,
            )
            raise ValueError(
                f"pdf file exceeds {_MAX_PDF_BYTES} bytes: {file_path} ({size} bytes)"
            )

        # Engine + OCR are resolved per-call from AdapterOptions when set so
        # config changes on disk don't require re-instantiating the adapter.
        engine = (
            getattr(options, "pdf_engine", None) or self.engine
        )
        ocr_enabled = (
            getattr(options, "ocr", None)
            if getattr(options, "ocr", None) is not None
            else self.ocr
        )
        requested_timeout = getattr(options, "ocr_timeout_s", None)
        ocr_timeout = _positive_ocr_timeout(
            self.ocr_timeout_s if requested_timeout is None else requested_timeout
        )
        requested_language = getattr(options, "ocr_language", None)
        ocr_language = validate_ocr_language(
            self.ocr_language if requested_language is None else requested_language
        ) if ocr_enabled else self.ocr_language
        requested_max_pages = getattr(options, "ocr_max_pages", None)
        ocr_max_pages = validate_ocr_max_pages(
            self.ocr_max_pages if requested_max_pages is None else requested_max_pages
        )

        extractor = _ENGINE_DISPATCH.get(engine)
        if extractor is None:
            raise ValueError(
                f"unknown pdf_engine {engine!r}; expected one of "
                f"{_SUPPORTED_ENGINES}"
            )
        extracted = extractor(file_path)
        layout_fallbacks = {}
        if len(extracted) == 2:
            # Keep small embedders/test doubles using the original engine
            # contract valid; real engines now also return failed page IDs.
            metadata, raw_pages = extracted
            failed_pages: set[int] = set()
        else:
            metadata, raw_pages, failures = extracted[:3]
            failed_pages = set(failures)
            if len(extracted) > 3:
                layout_fallbacks = extracted[3]

        sections: list[ExtractedSection] = []
        issues: list[ExtractionIssue] = []
        first_page_text: Optional[str] = None
        ocr_attempts = 0
        for page_no, text in raw_pages:
            if page_no in layout_fallbacks:
                issues.append(ExtractionIssue(
                    code="page_layout_fallback", page=page_no, engine=engine,
                    ocr_requested=bool(ocr_enabled), recovered=bool(text),
                    reason=layout_fallbacks[page_no],
                ))
            heading: Optional[str] = None
            ocr_reason: Optional[str] = None
            if not text and ocr_enabled:
                if ocr_max_pages and ocr_attempts >= ocr_max_pages:
                    # This is an explicit coverage limit, not a transient
                    # failure: don't repeatedly OCR the same prefix forever.
                    ocr_reason = "ocr_page_limit"
                else:
                    ocr_attempts += 1
                    ocr_result = _ocr_page(file_path, page_no, ocr_timeout, ocr_language)
                    # Preserve small test doubles that return only a string.
                    if isinstance(ocr_result, str):
                        ocr_text = ocr_result
                    else:
                        ocr_text = ocr_result.text
                        ocr_reason = ocr_result.reason
                    if ocr_text:
                        text = ocr_text
                        heading = _OCR_HEADING_MARKER
            if page_no in failed_pages or heading == _OCR_HEADING_MARKER or not text:
                issue = ExtractionIssue(
                    code=(
                        "page_extraction_failed" if page_no in failed_pages else
                        "ocr_used" if heading == _OCR_HEADING_MARKER else
                        "page_text_unavailable"
                    ),
                    page=page_no,
                    engine=engine,
                    ocr_requested=bool(ocr_enabled),
                    recovered=bool(text),
                    reason=ocr_reason,
                )
                issues.append(issue)
                if not issue.recovered:
                    logger.warning(
                        "pdf_page_text_unavailable", path=str(file_path),
                        page=page_no, engine=engine, code=issue.code,
                        ocr_requested=bool(ocr_enabled),
                    )
            if page_no == 1:
                first_page_text = text
            if not text:
                continue
            sections.append(ExtractedSection(
                text=text, page=page_no, heading=heading, order=page_no - 1,
            ))

        title = (
            metadata.get("title")
            or (first_page_text and _extract_first_heading(first_page_text))
            or file_path.stem
        )
        body = "\n\n".join(s.text for s in sections)

        return ExtractedDocument(
            source_path=file_path,
            format="pdf",
            title=title,
            text=body,
            content_hash=calculate_content_hash(body),
            file_hash=sha256_file(file_path),
            metadata=metadata,
            sections=sections,
            size_bytes=size,
            issues=issues,
        )


register(PdfAdapter())
