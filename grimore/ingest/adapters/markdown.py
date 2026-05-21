"""
Markdown adapter — the original Grimore ingest path, now behind the
multi-format dispatcher.

Behaviour is byte-for-byte identical to the pre-multiformat
:class:`grimore.ingest.parser.MarkdownParser`: YAML frontmatter is
separated from body, the title falls back through frontmatter → first
H1 → filename stem, and a 2 MB hard cap protects against accidental
binary blobs hiding under a ``.md`` extension.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Union

import frontmatter

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Same cap the old MarkdownParser enforced. Per-format caps come in Phase 2
# via [ingest.max_bytes] — for now we keep the historical 2 MB ceiling so
# Phase 1 stays a pure refactor.
_MAX_MD_BYTES = 2_000_000

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class MarkdownAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("md",)
    binary: ClassVar[bool] = False
    mutable_frontmatter: ClassVar[bool] = True

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        file_path = Path(path)

        if options.vault_root is not None:
            # Defence-in-depth — even if the caller already filtered, re-check
            # here so the adapter is safe to drop into a new call site.
            SecurityGuard.resolve_within_vault(file_path, options.vault_root)

        try:
            stat = file_path.stat()
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        size = stat.st_size
        if size > _MAX_MD_BYTES:
            logger.warning(
                "note_too_large",
                path=str(file_path),
                size=size,
                max=_MAX_MD_BYTES,
            )
            raise ValueError(
                f"note exceeds {_MAX_MD_BYTES} bytes: {file_path} ({size} bytes)"
            )

        with open(file_path, "r", encoding="utf-8") as fh:
            post = frontmatter.load(fh)

        content = post.content
        metadata = dict(post.metadata)

        title = metadata.get("title")
        if not title:
            h1 = _H1_RE.search(content)
            title = h1.group(1) if h1 else file_path.stem

        return ExtractedDocument(
            source_path=file_path,
            format="md",
            title=title,
            text=content,
            content_hash=calculate_content_hash(content),
            file_hash=sha256_file(file_path),
            metadata=metadata,
            sections=[],          # Phase 3 may revisit MD heading segmentation.
            size_bytes=size,
        )


# Side-effect: register on import.
register(MarkdownAdapter())
