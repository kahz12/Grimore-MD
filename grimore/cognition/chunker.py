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
from grimore.utils.logger import get_logger

logger = get_logger(__name__)

# Regex to split text by blank lines (paragraphs)
_PARA_SPLIT = re.compile(r"\n\s*\n")

# Sentence splitter for the semantic chunker. We deliberately keep this
# simple: split after ``.`` ``!`` ``?`` when followed by whitespace,
# guarding against common single-letter abbreviations (``Mr.`` ``Dr.``
# ``vs.``) and decimals (``3.14``). It's good enough for the docs we
# index — perfect sentence segmentation is a research problem and the
# hard cap (``chunk_max_chars``) catches anything that runs away.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'(\[])"
)
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "vs", "etc", "e.g", "i.e", "no", "fig", "vol", "cf",
}

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150
DEFAULT_SEMANTIC_THRESHOLD = 0.55


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


def build_candidate_chunks(
    note,
    body_text: str,
    embedder,
    config,
) -> list[Chunk]:
    """Single entry point shared by scan + daemon for chunking a note.

    Dispatches on ``config.ingest.chunker`` (``"markdown"`` or
    ``"semantic"``) and routes around section anchors when the adapter
    produced them. Keeps cli.py and daemon.py from drifting apart —
    the same input produces the same chunks regardless of which path
    enters the function.

    Markdown is the default and matches v2.1 byte-for-byte. Semantic
    needs the embedder because it scores sentence similarity to find
    topic boundaries.
    """
    engine = getattr(config.ingest, "chunker", "markdown")
    max_chars = getattr(config.ingest, "chunk_max_chars", DEFAULT_MAX_CHARS)
    threshold = getattr(config.ingest, "semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)

    use_sections = bool(getattr(note, "sections", None))
    if engine == "semantic":
        if use_sections:
            return chunk_semantic_sections(
                note.sections, embedder, max_chars=max_chars, threshold=threshold,
            )
        pieces = chunk_semantic(
            body_text, embedder, max_chars=max_chars, threshold=threshold,
        )
        return [Chunk(text=t) for t in pieces]

    # Markdown path (default).
    if use_sections:
        return chunk_sections(note.sections, max_chars=max_chars)
    return [Chunk(text=t) for t in chunk_markdown(body_text, max_chars=max_chars)]


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


def split_sentences(text: str) -> list[str]:
    """Heuristic sentence split for the semantic chunker.

    Splits on ``.!?`` followed by whitespace + an uppercase / quote /
    paren, then merges any chunk whose previous sentence ends with a
    known abbreviation (``Dr.`` ``e.g.``) back onto its neighbour. Not
    perfect — that's why :func:`chunk_semantic` caps chunk size as a
    safety net.
    """
    if not text or not text.strip():
        return []
    raw = _SENTENCE_SPLIT.split(text.strip())
    merged: list[str] = []
    for piece in raw:
        if merged:
            prev = merged[-1].rstrip()
            # Check the final dotted token of the previous chunk against
            # the abbreviation list. ``Dr.`` ``e.g.`` ``Mr.`` etc. all
            # end mid-thought and the split shouldn't have fired.
            tail = prev.split()[-1].rstrip(".").lower() if prev else ""
            if tail in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
        merged.append(piece)
    return [m.strip() for m in merged if m.strip()]


def chunk_semantic(
    text: str,
    embedder,
    max_chars: int = DEFAULT_MAX_CHARS,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split ``text`` on topic shifts measured by embedding similarity.

    Greedy: walk sentences in order, embed each, and keep adding to the
    current chunk while the candidate sentence's similarity to the
    running mean of the chunk stays at or above ``threshold``. When it
    drops below — that's a topic shift — flush and start a new chunk.

    Two safety nets keep the output well-behaved even when the embedder
    is misconfigured or the heuristic mis-splits:

    * **Hard cap** at ``max_chars``: a chunk that would exceed it
      flushes regardless of similarity. Matches the markdown chunker's
      :func:`_sliding_window` overflow path so downstream callers see
      the same size envelope.
    * **Embedder fallback**: if any sentence embed returns ``None``
      (Ollama down, empty payload) we degrade to the markdown chunker
      for the whole ``text`` rather than producing a degenerate split.
      Retrieval recall is more important than chunker fidelity when
      the model is flaky.
    """
    if not text or not text.strip():
        return []
    if max_chars <= 0:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return []

    # Single sentence — nothing to score; return as-is, capped.
    if len(sentences) == 1:
        only = sentences[0]
        if len(only) <= max_chars:
            return [only]
        return _sliding_window(only, max_chars, overlap)

    # Embed every sentence up front. Prefer the embedder's batch API so the
    # whole sentence list is one round-trip instead of N; fall back to the
    # per-sentence call for minimal embedder stand-ins that only implement
    # embed(). Resolved by duck-typing rather than import because embedder.py
    # imports this module — a hard import here would be circular.
    _embed_batch = getattr(embedder, "embed_batch", None)
    if callable(_embed_batch):
        vectors = _embed_batch(sentences)
    else:
        vectors = [embedder.embed(s) for s in sentences]
    if any(v is None for v in vectors):
        logger.warning("semantic_chunker_embed_failed", fallback="markdown")
        return chunk_markdown(text, max_chars, overlap)
    embeddings = [list(v) for v in vectors]

    chunks: list[str] = []
    cur_sents: list[str] = [sentences[0]]
    cur_sum: list[float] = list(embeddings[0])  # running sum (cheap mean)
    cur_len = 1

    def _flush():
        joined = " ".join(cur_sents).strip()
        if not joined:
            return
        if len(joined) > max_chars:
            # Last-line defence: a single sentence that exceeds the cap.
            chunks.extend(_sliding_window(joined, max_chars, overlap))
        else:
            chunks.append(joined)

    for i in range(1, len(sentences)):
        sent = sentences[i]
        emb = embeddings[i]

        # Mean of the running chunk: divide by count, then cosine.
        mean = [v / cur_len for v in cur_sum]
        sim = _cosine(mean, emb)

        candidate_len = sum(len(s) for s in cur_sents) + len(cur_sents) + len(sent)
        too_big = candidate_len > max_chars
        drift = sim < threshold

        if drift or too_big:
            _flush()
            cur_sents = [sent]
            cur_sum = list(emb)
            cur_len = 1
        else:
            cur_sents.append(sent)
            cur_sum = [a + b for a, b in zip(cur_sum, emb)]
            cur_len += 1

    _flush()
    return chunks


def chunk_semantic_sections(
    sections: Iterable[ExtractedSection],
    embedder,
    max_chars: int = DEFAULT_MAX_CHARS,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Semantic-chunk per section, propagating each section's anchors.

    Topic boundaries are computed *within* a section — we don't want a
    PDF page break or DOCX heading to merge into the next one even if
    the embeddings look similar, because the citation anchors would
    then point to the wrong page/heading.
    """
    out: list[Chunk] = []
    for section in sections:
        pieces = chunk_semantic(section.text, embedder, max_chars, threshold, overlap)
        for piece in pieces:
            out.append(Chunk(text=piece, page=section.page, heading=section.heading))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inlined to avoid an Embedder import cycle.

    Ollama embeddings are already unit-normalised, so for stored
    vectors this reduces to a dot product — but ``a`` here is the
    running *mean* (not normalised), so we compute the magnitude path
    for correctness.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    ma = sum(x * x for x in a) ** 0.5
    mb = sum(y * y for y in b) ** 0.5
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


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
