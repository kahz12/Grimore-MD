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

Limitations (documented, not failures):

* Encrypted PDFs are skipped with a structured ``ValueError`` so the
  daemon never sits idle waiting for a passphrase that won't come.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import ClassVar, Optional, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
    ExtractedSection,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
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
                        ("/Subject", "subject"), ("/Keywords", "keywords")):
        val = info.get(key)
        if val:
            try:
                out[target] = str(val).strip()
            except Exception:
                continue
    return out


# ── Per-engine page extractors ────────────────────────────────────────────
#
# Each engine returns ``(metadata, list[(page_no, text)])`` so the main
# ``extract`` body never has to branch on engine while assembling
# ExtractedSections. The 1-based page numbering lives here so every engine
# produces the same anchor shape. Page text is yielded only when non-empty;
# empty pages become OCR candidates in the caller.


def _extract_with_pypdf(file_path: Path) -> tuple[dict[str, str], list[tuple[int, str]]]:
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
    for idx, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            logger.warning(
                "pdf_page_extract_failed",
                path=str(file_path), page=idx + 1, engine="pypdf", error=str(e),
            )
            text = ""
        pages.append((idx + 1, text))
    return metadata, pages


def _extract_with_pdfplumber(file_path: Path) -> tuple[dict[str, str], list[tuple[int, str]]]:
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError as e:
        raise ValueError(
            f"pdf_engine='pdfplumber' requires the 'pdf-plumber' extra: "
            f"pip install 'grimore[pdf-plumber]' ({e})"
        ) from e

    metadata: dict[str, str] = {}
    pages: list[tuple[int, str]] = []
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
                except Exception as e:
                    logger.warning(
                        "pdf_page_extract_failed",
                        path=str(file_path), page=idx + 1,
                        engine="pdfplumber", error=str(e),
                    )
                    text = ""
                pages.append((idx + 1, text))
    except Exception as e:
        raise ValueError(f"pdfplumber failed on {file_path}: {e}") from e
    return metadata, pages


def _extract_with_pymupdf(file_path: Path) -> tuple[dict[str, str], list[tuple[int, str]]]:
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
    return metadata, pages


_ENGINE_DISPATCH = {
    "pypdf":      _extract_with_pypdf,
    "pdfplumber": _extract_with_pdfplumber,
    "pymupdf":    _extract_with_pymupdf,
}


def _ocr_page(file_path: Path, page_no: int, timeout_s: int) -> Optional[str]:
    """Best-effort OCR for one page. Returns the text (possibly empty) or
    None when OCR machinery is unavailable / fails.

    The function never raises — OCR failures are warnings, not file-level
    errors. Tesseract or its Python bridge missing simply skips the page.
    """
    if shutil.which("tesseract") is None:
        return None
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        # pdf2image needs poppler-utils. We constrain to a single page so
        # one bad page in a 500-page scan can't tie up the daemon.
        images = convert_from_path(
            str(file_path), first_page=page_no, last_page=page_no,
        )
    except Exception as e:
        logger.warning(
            "pdf_ocr_rasterise_failed",
            path=str(file_path), page=page_no, error=str(e),
        )
        return None
    if not images:
        return None
    try:
        text = pytesseract.image_to_string(images[0], timeout=timeout_s)
    except Exception as e:
        logger.warning(
            "pdf_ocr_failed",
            path=str(file_path), page=page_no, error=str(e),
        )
        return None
    return (text or "").strip()


class PdfAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("pdf",)
    binary: ClassVar[bool] = True
    mutable_frontmatter: ClassVar[bool] = False

    def __init__(self, *, engine: str = "pypdf", ocr: bool = False,
                 ocr_timeout_s: int = 30):
        """The adapter is normally instantiated once at import time with
        defaults; the section-aware scan path resolves the per-run engine
        through ``AdapterOptions.ingest_engine`` so config swaps take
        effect on the next scan without re-importing.
        """
        if engine not in _SUPPORTED_ENGINES:
            raise ValueError(
                f"unknown pdf_engine {engine!r}; expected one of "
                f"{_SUPPORTED_ENGINES}"
            )
        self.engine = engine
        self.ocr = ocr
        self.ocr_timeout_s = ocr_timeout_s

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
        ocr_timeout = (
            getattr(options, "ocr_timeout_s", None) or self.ocr_timeout_s
        )

        extractor = _ENGINE_DISPATCH.get(engine)
        if extractor is None:
            raise ValueError(
                f"unknown pdf_engine {engine!r}; expected one of "
                f"{_SUPPORTED_ENGINES}"
            )
        metadata, raw_pages = extractor(file_path)

        sections: list[ExtractedSection] = []
        first_page_text: Optional[str] = None
        for page_no, text in raw_pages:
            heading: Optional[str] = None
            if not text and ocr_enabled:
                ocr_text = _ocr_page(file_path, page_no, ocr_timeout)
                if ocr_text:
                    text = ocr_text
                    heading = _OCR_HEADING_MARKER
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
        )


register(PdfAdapter())
