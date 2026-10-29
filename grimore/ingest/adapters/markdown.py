"""
Markdown adapter — the native Grimore format, the only one whose source file
is edited in place (see ``mutable_frontmatter``).

YAML frontmatter is separated from the body, the title falls back through
frontmatter → first H1 → filename stem, and a 2 MB hard cap protects against
accidental binary blobs hiding under a ``.md`` extension. The body is split
into one section per ATX heading so chunks cite ``[[Title#Heading]]``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Union

import frontmatter

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

# Hard cap on a Markdown note, in bytes: a larger file is rejected rather than
# read into memory and sent through tagging and embedding. Each adapter defines
# its own limit (plain text, HTML, DOCX... allow more).
_MAX_MD_BYTES = 2_000_000

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def markdown_sections(content: str, title: str) -> list[ExtractedSection]:
    """One section per ATX heading, outside fenced code blocks.

    A section's text excludes its heading line; ``heading`` is the section's
    own heading and ``breadcrumb`` the title plus the headings above it. Text
    before the first heading, or directly under an H1 that repeats the title,
    belongs to the whole note: it has no heading, so it cites as ``[[Title]]``.
    """
    title = " ".join(title.split())
    sections: list[ExtractedSection] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    heading: str | None = None
    in_fence = False

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            path = [h for _, h in stack]
            if path and path[0] == title:
                path = path[1:]
            sections.append(ExtractedSection(
                text=text, heading=heading, order=len(sections),
                breadcrumb=(title, *path),
            ))

    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            lines.append(line)
            continue
        flush()
        lines = []
        level, text = len(match.group(1)), " ".join(match.group(2).split())
        stack = [(lv, h) for lv, h in stack if lv < level] + [(level, text)]
        heading = None if level == 1 and text == title else text
    flush()
    return sections


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
            sections=markdown_sections(content, str(title)),
            size_bytes=size,
        )


# Side-effect: register on import.
register(MarkdownAdapter())
