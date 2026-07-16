"""
Duplicate-note detection.

Two independent signals, both deterministic and LLM-free so the command
works even when Ollama is busy or offline:

* **Exact** — notes sharing the same ``content_hash``. Byte-identical
  bodies (re-imports, stray copies); a single SQL GROUP BY.
* **Near** — note pairs whose mean-pooled chunk vectors score above a
  cosine threshold. Catches rewrites, partial copies and re-embedded
  duplicates that the hash misses.

Report-only: nothing here touches the vault or the index.
"""
from __future__ import annotations

from dataclasses import dataclass

from grimore.cognition.embedder import Embedder, _np


@dataclass
class ExactGroup:
    """Notes whose bodies hash identically."""
    content_hash: str
    notes: list[tuple[int, str, str]]  # (note_id, path, title)


@dataclass
class NearDuplicate:
    """A pair of notes above the similarity threshold, ``a_id < b_id``."""
    a_id: int
    a_path: str
    a_title: str
    b_id: int
    b_path: str
    b_title: str
    score: float


def find_exact_duplicates(db) -> list[ExactGroup]:
    """Groups of 2+ notes sharing a non-empty ``content_hash``.

    Largest groups first; within a group, notes keep insertion (id) order
    so the oldest copy is listed first.
    """
    with db._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT n.content_hash, n.id, n.path, n.title
            FROM notes n
            JOIN (
                SELECT content_hash FROM notes
                WHERE content_hash IS NOT NULL AND content_hash != ''
                GROUP BY content_hash HAVING COUNT(*) > 1
            ) dupes ON dupes.content_hash = n.content_hash
            ORDER BY n.content_hash, n.id
            """
        ).fetchall()

    groups: dict[str, list[tuple[int, str, str]]] = {}
    for content_hash, note_id, path, title in rows:
        groups.setdefault(content_hash, []).append((note_id, path, title or ""))
    return sorted(
        (ExactGroup(h, notes) for h, notes in groups.items()),
        key=lambda g: (-len(g.notes), g.notes[0][0]),
    )


def _note_mean_vectors(db) -> dict[int, list[float]]:
    """Mean-pooled, unit-normalized vector per note.

    Same trick as ``graph._suggested_edges`` but batched into one query.
    Notes with ragged chunk dims (model swapped without a re-scan) are
    skipped rather than aborting the whole report.
    """
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT note_id, vector FROM embeddings WHERE vector IS NOT NULL"
        ).fetchall()

    by_note: dict[int, list[list[float]]] = {}
    for note_id, blob in rows:
        try:
            by_note.setdefault(note_id, []).append(Embedder.deserialize_vector(blob))
        except Exception:
            by_note[note_id] = []  # poison the note; dropped below

    means: dict[int, list[float]] = {}
    for note_id, vectors in by_note.items():
        if not vectors:
            continue
        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            continue  # ragged → skip the note, keep the report alive
        avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        means[note_id] = Embedder.normalize(avg)
    return means


def _scored_pairs(
    ids: list[int], means: dict[int, list[float]], threshold: float
) -> list[tuple[int, int, float]]:
    """Every ``(a_id, b_id, score)`` with ``score >= threshold``, a < b.

    Fast path: one ``M @ M.T`` matmul over the note-mean matrix; N is
    note count (not chunks), so N² stays trivial for real vaults. The
    pure-Python fallback is pairwise dot products with the same
    semantics. Cross-dim pairs (different embedding models coexisting)
    are silently incomparable and skipped.
    """
    pairs: list[tuple[int, int, float]] = []
    dims = {len(means[i]) for i in ids}
    if _np is not None and len(dims) == 1:
        matrix = _np.asarray([means[i] for i in ids], dtype=_np.float32)
        sims = matrix @ matrix.T
        iu_a, iu_b = _np.triu_indices(len(ids), k=1)
        keep = sims[iu_a, iu_b] >= threshold
        for a, b in zip(iu_a[keep].tolist(), iu_b[keep].tolist(), strict=True):
            pairs.append((ids[a], ids[b], float(sims[a, b])))
        return pairs

    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            if len(means[a_id]) != len(means[b_id]):
                continue
            score = Embedder.dot_product(means[a_id], means[b_id])
            if score >= threshold:
                pairs.append((a_id, b_id, score))
    return pairs


def find_near_duplicates(
    db, *, threshold: float = 0.90, limit: int = 30
) -> list[NearDuplicate]:
    """Note pairs whose mean vectors score ``>= threshold``, best first.

    Pairs that already share a ``content_hash`` are left to
    :func:`find_exact_duplicates` — reporting them twice would just be
    noise. Truncated to ``limit`` pairs after sorting.
    """
    means = _note_mean_vectors(db)
    if len(means) < 2:
        return []

    ids = sorted(means)
    pairs = _scored_pairs(ids, means, threshold)
    if not pairs:
        return []

    involved = {i for a, b, _ in pairs for i in (a, b)}
    placeholders = ",".join("?" * len(involved))
    with db._get_connection() as conn:
        rows = conn.execute(
            f"SELECT id, path, title, content_hash FROM notes WHERE id IN ({placeholders})",
            sorted(involved),
        ).fetchall()
    meta = {nid: (path, title or "", content_hash) for nid, path, title, content_hash in rows}

    out: list[NearDuplicate] = []
    for a_id, b_id, score in sorted(pairs, key=lambda p: (-p[2], p[0], p[1])):
        a = meta.get(a_id)
        b = meta.get(b_id)
        if a is None or b is None:
            continue  # embedding rows orphaned from their note
        if a[2] and a[2] == b[2]:
            continue  # exact duplicate — reported by find_exact_duplicates
        out.append(NearDuplicate(
            a_id=a_id, a_path=a[0], a_title=a[1],
            b_id=b_id, b_path=b[0], b_title=b[1],
            score=score,
        ))
        if len(out) >= limit:
            break
    return out
