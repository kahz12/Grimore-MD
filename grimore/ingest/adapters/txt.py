"""
Plain-text adapter — the simplest non-Markdown format.

Reads the file as UTF-8 with ``errors="replace"`` so a stray byte at the
end of an otherwise readable log won't fail the whole scan. There is no
metadata to parse, no sections to anchor, and no fallback chain for the
title beyond the filename stem.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Union

from grimore.ingest.adapters.base import AdapterOptions, ExtractedDocument
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Per-format cap (blueprint §6.6). Larger than Markdown's 2 MB because
# plain-text logs and source dumps regularly run a few megabytes.
_MAX_TXT_BYTES = 5_000_000


class TxtAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("txt",)
    binary: ClassVar[bool] = False
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
        if size > _MAX_TXT_BYTES:
            logger.warning(
                "txt_too_large", path=str(file_path), size=size, max=_MAX_TXT_BYTES,
            )
            raise ValueError(
                f"text file exceeds {_MAX_TXT_BYTES} bytes: {file_path} ({size} bytes)"
            )

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        return ExtractedDocument(
            source_path=file_path,
            format="txt",
            title=file_path.stem,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata={},
            sections=[],
            size_bytes=size,
        )


register(TxtAdapter())
