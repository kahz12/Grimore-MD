"""
RTF adapter — pure-Python via ``striprtf``.

RTF (Rich Text Format) has no reliable section structure once you strip
the formatting codes, so the adapter emits a single anchor-free section
holding the entire body. That matches the Markdown / TXT shape and means
the section-aware embedding path stays unused for this format (anchors
would all be ``None`` anyway).

striprtf is pure-Python with no native deps, so it sits in the hard
dependency set — Termux installs cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Union

from striprtf.striprtf import rtf_to_text

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_RTF_BYTES = 25_000_000


class RtfAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("rtf",)
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
        if size > _MAX_RTF_BYTES:
            logger.warning(
                "rtf_too_large", path=str(file_path), size=size, max=_MAX_RTF_BYTES,
            )
            raise ValueError(
                f"rtf file exceeds {_MAX_RTF_BYTES} bytes: {file_path} ({size} bytes)"
            )

        # RTF is ASCII-ish on the wire; striprtf accepts either str or bytes
        # but expects to see the leading `{\rtf1` marker. Reading as text with
        # errors='replace' keeps a malformed file from blowing up the scan.
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            text = rtf_to_text(raw, errors="ignore")
        except Exception as e:
            # striprtf raises on truly malformed input. Treat as adapter-level
            # failure so the daemon logs it and moves on.
            raise ValueError(f"failed to parse rtf {file_path}: {e}") from e

        text = text.strip()
        title = file_path.stem
        return ExtractedDocument(
            source_path=file_path,
            format="rtf",
            title=title,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata={},
            sections=[],
            size_bytes=size,
        )


register(RtfAdapter())
