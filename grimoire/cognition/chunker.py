"""
Semantically-aware Markdown Chunker.
This module splits large Markdown files into smaller, manageable pieces (chunks)
while respecting the logical structure of the document (paragraphs).
"""
import re

# Regex to split text by blank lines (paragraphs)
_PARA_SPLIT = re.compile(r"\n\s*\n")

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150


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


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """
    Splits a large block of text into overlapping pieces of a fixed size.
    Used as a fallback for paragraphs that exceed the maximum chunk size.
    """
    if size <= 0 or len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
    return out
