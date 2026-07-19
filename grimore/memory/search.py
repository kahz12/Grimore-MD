"""
Chunk-level retrieval primitives: BM25 full-text search (FTS5) and
sqlite-vec k-NN search, plus the availability flags the connector reads
to pick between them and the numpy fallback.
"""
import sqlite3
from typing import Optional

from grimore.memory._base import DbBase

from grimore.utils.logger import get_logger

logger = get_logger(__name__)

# Upper bound on the number of OR-ed terms in an FTS5 MATCH expression, so a
# pathological (e.g. multi-kilobyte) query can't build an unbounded query tree.
_FTS_MAX_TERMS = 50


class SearchMixin(DbBase):
    """FTS5 + sqlite-vec search surface for :class:`Database`."""

    @property
    def fts_available(self) -> bool:
        """Whether FTS5 is compiled in and wired up on this database."""
        return bool(getattr(self, "_fts_available", False))

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[int, int, str, float]]:
        """
        BM25 full-text search over embeddings.text_content.

        Returns ``[(embedding_id, note_id, text_content, bm25_score), …]``
        sorted by relevance (lower BM25 = more relevant; SQLite's bm25()
        returns negative values where smaller is better). Safe no-op when
        FTS5 isn't available.
        """
        if not self.fts_available or not query or not query.strip():
            return []
        # FTS5 match string. Tokenise on whitespace and OR the terms so we get
        # bag-of-words BM25 recall — a document that contains the query words
        # in any order/position matches, instead of only the exact phrase a
        # single quoted string would demand. Each token is individually
        # double-quoted (with internal ``"`` doubled) so FTS5 operators or
        # punctuation in user input can't be parsed as query syntax — same
        # injection safety as before, wider recall. Capped so a pathological
        # query can't build a giant MATCH expression.
        tokens = [t for t in query.split() if t][:_FTS_MAX_TERMS]
        if not tokens:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        with self._get_connection() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.note_id, e.text_content, bm25(embeddings_fts) AS score
                    FROM embeddings_fts
                    JOIN embeddings e ON e.id = embeddings_fts.rowid
                    WHERE embeddings_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("fts_query_failed", error=str(e))
                return []
        return [(r[0], r[1], r[2], float(r[3])) for r in rows]

    # ── sqlite-vec search ─────────────────────────────────────────────────

    @property
    def vec_available(self) -> bool:
        """True only when the extension loaded AND a vec table is ready.

        The connector reads this to decide whether to skip the numpy
        matrix build entirely; if it ever returns True transiently and
        flips to False (extension misbehaves mid-session), the connector
        falls back to numpy on the next call without raising.
        """
        return bool(self._vec_available and self._vec_dim is not None)

    @property
    def vec_dim(self) -> Optional[int]:
        """Embedding dim currently held by ``embeddings_vec`` (None if absent)."""
        return self._vec_dim

    def vec_search(
        self,
        query_vector: bytes | list[float],
        limit: int,
        exclude_note_id: Optional[int] = None,
    ) -> list[tuple[int, int, str, float]]:
        """k-NN search against ``embeddings_vec``.

        Returns ``[(embedding_id, note_id, text_content, similarity), …]``
        sorted by descending similarity. ``similarity = 1 - cosine_distance``
        so the score range matches the numpy dot-product path and the
        connector can fuse with BM25 the same way.

        Caller filters ``exclude_note_id`` post-hoc by widening ``limit``
        a touch — same pattern as the numpy fast path. Safe no-op (empty
        list) when the vec table isn't available.
        """
        if not self.vec_available or limit <= 0:
            return []
        if isinstance(query_vector, list):
            import struct
            qbytes = struct.pack(f"{len(query_vector)}f", *query_vector)
        else:
            qbytes = bytes(query_vector)
        with self._get_connection() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.note_id, e.text_content, v.distance
                    FROM embeddings_vec v
                    JOIN embeddings e ON e.id = v.rowid
                    WHERE v.embedding MATCH ?
                      AND k = ?
                    ORDER BY v.distance ASC
                    """,
                    (qbytes, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("vec_query_failed", error=str(e))
                return []
        out: list[tuple[int, int, str, float]] = []
        for eid, nid, text, dist in rows:
            if exclude_note_id is not None and nid == exclude_note_id:
                continue
            out.append((int(eid), int(nid), text, 1.0 - float(dist)))
        return out
