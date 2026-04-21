"""
Semantically-aware markdown chunker.

Strategy:
- Split by blank-line paragraphs (markdown's logical unit).
- Pack whole paragraphs into chunks until the budget is exceeded.
- Paragraphs larger than the budget fall back to a fixed-width sliding
  window with overlap, so nothing gets silently dropped.
"""
import re

_PARA_SPLIT = re.compile(r"\n\s*\n")

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150


def chunk_markdown(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush():
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            chunks.extend(_sliding_window(para, max_chars, overlap))
            continue

        sep = 2 if buffer else 0
        if buffer_len + sep + len(para) > max_chars and buffer:
            flush()
            sep = 0

        buffer.append(para)
        buffer_len += sep + len(para)

    flush()
    return chunks


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
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
