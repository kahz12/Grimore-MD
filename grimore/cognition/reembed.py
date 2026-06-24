"""
Chunk-level incremental re-embedding.

Both ``grimore scan`` and the file-watcher daemon used to delete every
embedding for a note and re-embed it from scratch on any change. For a
200-page PDF where one paragraph was tweaked that's hundreds of needless
Ollama round-trips. This module shares one diff-and-embed routine
between both call sites so a re-scan only re-embeds what actually
changed, keyed on ``Embedder.chunk_hash(text, model)``.

Legacy rows (pre-v2.3) have ``chunk_hash = NULL``. They're treated as
unconditionally stale so the column back-fills naturally on the first
re-scan after upgrade — no explicit migration command needed for the
typical user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from grimore.cognition.chunker import Chunk
from grimore.cognition.embedder import Embedder
from grimore.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReembedResult:
    """Outcome counts returned from :func:`reembed_note`.

    Stats consumers (the scan summary, the daemon event log) read
    ``stored`` for "chunks this note contributes" and ``embedded`` to
    show how many Ollama calls actually happened — the delta between
    the two is the cache hit count.
    """
    kept: int
    embedded: int
    removed: int
    stored: int   # total chunks present after the operation


def reembed_note(
    db,
    embedder: Embedder,
    note_id: int,
    chunks: Iterable[Chunk],
    *,
    text_truncation: int = 500,
) -> ReembedResult:
    """Replace ``note_id``'s embeddings only where the chunk text changed.

    ``chunks`` is the section-aware (or markdown-body wrapped) list
    produced by :func:`chunk_sections` / :func:`chunk_markdown`. Order
    matters: ``chunk_index`` is positional, so passing the same chunker
    output between scans is what keeps unchanged rows in place.

    Behaviour:

    * Compute :py:meth:`Embedder.chunk_hash` for every candidate.
    * Keep rows whose stored hash matches the candidate at the same
      ``chunk_index``.
    * Delete rows for indices that no longer exist (the note shrank) and
      for indices whose content changed.
    * Embed only the new / changed chunks and insert them.

    A failed ``embed()`` (Ollama down, etc.) skips that chunk — same
    policy as the legacy "delete all then re-embed" path. ``text_truncation``
    matches the historical 500-char column store; the *hash* is computed
    over the full chunk text because that's what gets sent to Ollama.
    """
    chunk_list = list(chunks)
    existing = db.get_chunk_records(note_id)   # {idx: (hash, page, heading)}
    model = embedder.model

    # Plan each candidate: keep vs re-embed, by index.
    plan: list[tuple[int, Chunk, str, bool]] = []
    for idx, c in enumerate(chunk_list):
        h = Embedder.chunk_hash(c.text, model)
        prior = existing.get(idx)
        prior_hash = prior[0] if prior is not None else None
        # Legacy NULL hashes count as stale — re-embed once to back-fill.
        is_stale = prior_hash is None or prior_hash != h
        plan.append((idx, c, h, is_stale))

    # Indices to drop: anything past the new length (note shrank) plus
    # in-range indices whose content actually changed.
    surplus = [i for i in existing.keys() if i >= len(chunk_list)]
    changed = [idx for idx, _, _, stale in plan if stale and idx in existing]
    to_delete = sorted(set(surplus) | set(changed))
    removed = db.delete_chunks(note_id, to_delete) if to_delete else 0

    kept = 0
    embedded = 0
    reanchored = 0
    stale_items: list[tuple[int, Chunk, str]] = []
    for idx, c, h, stale in plan:
        if not stale:
            kept += 1
            # Text (and so the embedding) is unchanged, but an edit earlier in
            # the document may have shifted this chunk's page/heading. Refresh
            # the stored anchors cheaply — no re-embed — so citations stay
            # correct even when only pagination moved.
            prior = existing.get(idx)
            if prior is not None and (prior[1], prior[2]) != (c.page, c.heading):
                db.update_chunk_anchors(note_id, idx, c.page, c.heading)
                reanchored += 1
            continue
        stale_items.append((idx, c, h))

    # Embed every changed chunk in as few round-trips as possible. Prefer the
    # embedder's batch API (one request for the whole note) and fall back to
    # per-chunk embed() for minimal stand-ins that only implement embed().
    if stale_items:
        texts = [c.text for _, c, _ in stale_items]
        _embed_batch = getattr(embedder, "embed_batch", None)
        vectors = _embed_batch(texts) if callable(_embed_batch) \
            else [embedder.embed(t) for t in texts]
        for (idx, c, h), vector in zip(stale_items, vectors, strict=True):
            if vector is None:
                # Embed failure: skip — the next scan will retry. We've already
                # deleted any prior row for this index, so nothing inconsistent
                # is left behind.
                continue
            db.store_embedding(
                note_id,
                idx,
                c.text[:text_truncation],
                embedder.serialize_vector(vector),
                page=c.page,
                heading=c.heading,
                chunk_hash=h,
            )
            embedded += 1

    stored = kept + embedded
    if removed or embedded or reanchored:
        logger.info(
            "reembed_note_done",
            note_id=note_id,
            kept=kept,
            embedded=embedded,
            removed=removed,
            reanchored=reanchored,
        )
    return ReembedResult(kept=kept, embedded=embedded, removed=removed, stored=stored)
