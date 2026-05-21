"""
PDF adapter — pure-Python via ``pypdf``.

Per-page extraction lets every emitted section carry its 1-based page
number, which the embedding layer threads through into ``embeddings.page``
so the Oracle can render citations as ``[[Title#p.42]]`` rather than a
bare ``[[Title]]``.

Limitations (documented, not failures):

* Scanned PDFs with no text layer produce empty pages and therefore no
  sections — Grimore will warn in the scan summary but not error out.
  OCR is gated behind an opt-in extra in a later phase.
* Encrypted PDFs are skipped with a structured ``ValueError`` so the
  daemon never sits idle waiting for a passphrase that won't come.
"""
from __future__ import annotations

import re
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


class PdfAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("pdf",)
    binary: ClassVar[bool] = True
    mutable_frontmatter: ClassVar[bool] = False

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

        try:
            reader = PdfReader(str(file_path))
        except PdfReadError as e:
            raise ValueError(f"unreadable pdf {file_path}: {e}") from e
        except Exception as e:
            # pypdf can raise plain Exceptions on really mangled files; we
            # swallow them here so one bad PDF doesn't take the scan down.
            raise ValueError(f"failed to open pdf {file_path}: {e}") from e

        if getattr(reader, "is_encrypted", False):
            # Some PDFs are "encrypted" with an empty password — try once
            # before giving up so we don't skip the trivial case.
            try:
                if not reader.decrypt(""):
                    raise ValueError(f"encrypted pdf {file_path}: passphrase required")
            except Exception as e:
                raise ValueError(f"encrypted pdf {file_path}: {e}") from e

        metadata = _read_pdf_metadata(reader)

        sections: list[ExtractedSection] = []
        first_page_text: Optional[str] = None
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                # Per-page failure is logged and skipped; the rest of the
                # document still indexes.
                logger.warning(
                    "pdf_page_extract_failed",
                    path=str(file_path), page=idx + 1, error=str(e),
                )
                continue
            text = text.strip()
            if idx == 0:
                first_page_text = text
            if not text:
                continue
            sections.append(ExtractedSection(
                text=text, page=idx + 1, heading=None, order=idx,
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
