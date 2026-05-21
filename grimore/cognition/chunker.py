"""
Semantically-aware document chunker.

Originally Markdown-only. As of v2.1 this module also exposes a
section-aware path: :func:`chunk_sections` takes a list of
:class:`ExtractedSection` (the format-neutral output of every adapter)
and emits :class:`Chunk` instances that carry the parent section's
page / heading anchors, so the embedding layer can persist them and
the Oracle can render precise citations like ``[[Title#p.42]]``.

The text-only :func:`chunk_markdown` keeps its signature and behaviour
so the existing Markdown call path is untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from grimore.ingest.adapters.base import ExtractedSection

# Regex to split text by blank lines (paragraphs)
_PARA_SPLIT = re.compile(r"\n\s*\n")

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    """A piece of body text with the source anchors that produced it.

    ``page`` is set for paginated formats (PDF); ``heading`` for
    structured formats (DOCX, HTML, EPUB). Both are ``None`` for
    Markdown / TXT so the v2.0 storage shape stays exactly what it was.
    """
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None


def chunk_markdown(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Splits markdown text into chunks.
    
    Strategy:
    1. Split by blank-line paragraphs.
    2. Pack whole paragraphs into chunks until the character budget (max_chars) is met.
    3. If a single paragraph is larger than max_chars, it's split using a sliding window.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush():
        """Joins current buffer into a single chunk and resets state."""
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_len = 0

    for para in paragraphs:
        # Handling oversized paragraphs
        if len(para) > max_chars:
            flush()
            chunks.extend(_sliding_window(para, max_chars, overlap))
            continue

        # Check if adding this paragraph exceeds the budget
        sep = 2 if buffer else 0 # Accounting for "\n\n" separator
        if buffer_len + sep + len(para) > max_chars and buffer:
            flush()
            sep = 0

        buffer.append(para)
        buffer_len += sep + len(para)

    flush()
    return chunks


def chunk_sections(
    sections: Iterable[ExtractedSection],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split each section's text via :func:`chunk_markdown`, stamping
    every produced piece with the parent section's anchors.

    A section that exceeds ``max_chars`` is broken with the existing
    sliding-window strategy — each resulting chunk inherits the section's
    page / heading, so a PDF page that overflows the budget still cites
    back to the same page number.

    Sections with empty / whitespace-only text are silently dropped.
    """
    chunks: list[Chunk] = []
    for section in sections:
        pieces = chunk_markdown(section.text, max_chars, overlap)
        for piece in pieces:
            chunks.append(Chunk(
                text=piece, page=section.page, heading=section.heading,
            ))
    return chunks


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """
    Splits a large block of text into overlapping pieces of a fixed size.
    Used as a fallback for paragraphs that exceed the maximum chunk size.
    """
    if size <= 0 or len(text) <= size:
        return [text] if text else []
    # Clamp overlap so step is always meaningful progress. If overlap >= size,
    # step would degenerate to 1 and emit len(text) near-identical windows —
    # trivial OOM path for any hostile or mis-configured note. Capping at
    # size // 2 keeps the "overlap" semantics without the runaway.
    if overlap >= size:
        overlap = size // 2
    step = max(1, size - overlap)
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
    return out
