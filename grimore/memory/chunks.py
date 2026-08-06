"""
Chunk/embedding storage: the ``embeddings`` table, its sqlite-vec mirror,
the embedding cache, and the change-detection signatures the connector
uses to cache its scoring matrix.
"""
import sqlite3
from typing import Iterable, Optional

from grimore.memory._base import DbBase

from grimore.utils.logger import get_logger

logger = get_logger(__name__)

# Host parameters per statement; see the note on notes._MAX_SQL_VARS.
_MAX_SQL_VARS = 900


def _batched(items: list, size: int) -> Iterable[list]:
    """Yield ``items`` in consecutive slices of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


class ChunksMixin(DbBase):
    """Embedding rows, vec mirror, and embedding cache for :class:`Database`."""

    def get_chunk_anchors(
        self, note_id: int, text_content: str,
    ) -> tuple[Optional[int], Optional[str]]:
        """Return ``(page, heading)`` for the chunk in ``note_id`` whose
        stored text matches ``text_content`` exactly.

        Used by the Oracle to render anchor-aware citations
        (``[[Title#p.42]]``). The match is exact rather than fuzzy
        because the connector and the embeddings table share the same
        500-char truncation, so a literal compare is safe. Returns
        ``(None, None)`` when no row matches or both anchors are unset
        (Markdown / TXT path, where there is nothing to anchor on).
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings "
                "WHERE note_id = ? AND text_content = ? LIMIT 1",
                (note_id, text_content),
            ).fetchone()
        if not row:
            return (None, None)
        return (row[0], row[1])

    def get_chunk_anchors_bulk(
        self, pairs: Iterable[tuple[int, str]],
    ) -> dict[tuple[int, str], tuple[Optional[int], Optional[str]]]:
        """Batch form of :py:meth:`get_chunk_anchors`.

        Takes ``(note_id, text_content)`` pairs and returns them mapped to
        ``(page, heading)``. The Oracle resolves one anchor per cited chunk;
        done singly that is a query each, and each one scans ``text_content``,
        which has no index of its own (only ``idx_embeddings_note_id`` bounds
        it). One statement for the whole citation set costs the same scan work
        but pays the per-statement overhead once.

        Selection is ``note_id IN (...) AND text_content IN (...)`` — a
        superset of the requested pairs, since it also matches cross
        combinations — and the pairing is re-established in Python. That keeps
        the note_id index usable; restricting to exact pairs would need an
        ``OR`` chain that SQLite plans far worse.

        Tie-breaking matches the singular method: ``LIMIT 1`` with no
        ``ORDER BY`` returns the first row the scan reaches, which for this
        index-bounded plan is the lowest ``id``. Ordering by ``id`` and
        keeping the first occurrence per key reproduces that. Pairs with no
        matching row are absent from the result, so ``.get(key, (None, None))``
        gives the singular contract.
        """
        wanted = {(int(n), t) for n, t in pairs}
        if not wanted:
            return {}

        note_ids = sorted({n for n, _ in wanted})
        texts = sorted({t for _, t in wanted})
        anchors: dict[tuple[int, str], tuple[Optional[int], Optional[str]]] = {}

        with self._get_connection() as conn:
            # Both IN lists ride in the same statement, so the chunk size has
            # to bound their sum, not each separately.
            for n_batch in _batched(note_ids, _MAX_SQL_VARS // 2):
                for t_batch in _batched(texts, _MAX_SQL_VARS // 2):
                    rows = conn.execute(
                        "SELECT note_id, text_content, page, heading "
                        "FROM embeddings "
                        f"WHERE note_id IN ({','.join('?' * len(n_batch))}) "
                        f"AND text_content IN ({','.join('?' * len(t_batch))}) "
                        "ORDER BY id",
                        tuple(n_batch) + tuple(t_batch),
                    ).fetchall()
                    for note_id, text, page, heading in rows:
                        key = (int(note_id), text)
                        # Cross-product rows for pairs nobody asked about, and
                        # the 2nd+ duplicate of a key, are both dropped here.
                        if key in wanted and key not in anchors:
                            anchors[key] = (page, heading)
        return anchors

    def store_embedding(
        self,
        note_id: int,
        chunk_index: int,
        text_content: str,
        vector_blob: bytes,
        *,
        page: Optional[int] = None,
        heading: Optional[str] = None,
        chunk_hash: Optional[str] = None,
    ):
        """Stores a vector embedding for a specific note chunk.

        ``page`` / ``heading`` are the multi-format anchors used by the
        Oracle when rendering citations (``[[Title#p.42]]``). ``chunk_hash``
        is the content+model fingerprint used by the incremental re-embed
        path; legacy callers that omit it write NULL and back-fill on the
        next scan. All three are keyword-only so the v2.0 four-arg call
        shape keeps working.
        """
        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO embeddings (note_id, chunk_index, text_content, vector,
                                        page, heading, chunk_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (note_id, chunk_index, text_content, vector_blob, page, heading, chunk_hash))
            new_rowid = cur.lastrowid
            assert new_rowid is not None  # always set after INSERT
            self._mirror_vec_insert(conn, new_rowid, vector_blob)
            conn.commit()

    def store_embeddings_bulk(self, note_id: int, rows: list[dict]) -> int:
        """Insert every chunk of a note in ONE transaction. Returns rows written.

        The per-chunk :meth:`store_embedding` is kept for the callers (and the
        many tests) that write a single row, and as the rollback path; this is
        purely additive.

        Why it matters: in WAL mode each commit costs an fsync, and the old
        re-embed loop committed once per chunk. Measured at 7.17 ms per
        connection-and-commit across a 2000-note scan, that fsync -- not the
        SQL -- was the dominant cost of indexing. Collapsing N transactions
        into one is the whole optimisation; ``executemany`` on top of it just
        removes the per-row Python overhead.

        Each row is a dict with chunk_index, text_content, vector, page,
        heading and chunk_hash. ``note_id`` is passed once rather than repeated
        in every row.
        """
        if not rows:
            return 0
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (note_id, chunk_index, text_content, vector,
                                        page, heading, chunk_hash)
                VALUES (:note_id, :chunk_index, :text_content, :vector,
                        :page, :heading, :chunk_hash)
                """,
                [{"note_id": note_id, **row} for row in rows],
            )
            self._mirror_vec_insert_many(conn, note_id, rows)
        return len(rows)

    def _mirror_vec_insert_many(self, conn, note_id: int, rows: list[dict]) -> None:
        """Mirror a bulk insert into ``embeddings_vec`` on the same transaction.

        The rowids are read back rather than derived from ``cursor.lastrowid``.
        Deriving them (``lastrowid - len(rows) + 1``) assumes the AUTOINCREMENT
        ids landed contiguously, which holds for one uninterrupted executemany
        but silently produces wrong mappings the moment anything else writes to
        the table -- and a wrong mapping here means a vector filed under
        another chunk's id, which retrieval would surface as a citation
        pointing at the wrong note.

        The read-back selects by ``note_id`` alone instead of an
        ``IN (chunk_index, ...)`` list: a large PDF can exceed SQLite's host
        parameter limit, and the extra rows for chunks that were kept rather
        than re-embedded are simply ignored by the lookup below.
        """
        if not self._vec_available or not rows:
            return

        dim = len(rows[0]["vector"]) // 4
        if dim <= 0:
            return
        if self._vec_dim is None:
            self._create_vec_table(conn, dim)
            self._vec_dim = dim

        id_by_index = {
            int(chunk_index): int(rowid)
            for chunk_index, rowid in conn.execute(
                "SELECT chunk_index, id FROM embeddings WHERE note_id = ?",
                (note_id,),
            )
        }
        payload = []
        for row in rows:
            rowid = id_by_index.get(row["chunk_index"])
            # Same per-row dim guard as the singular path: a model swap without
            # a migration must still land the source-of-truth embeddings row,
            # leaving the connector on numpy until migrate-embeddings runs.
            if rowid is None or len(row["vector"]) // 4 != self._vec_dim:
                continue
            payload.append((rowid, row["vector"]))
        if not payload:
            return
        self._vec_write(
            conn,
            "INSERT INTO embeddings_vec(rowid, embedding) VALUES (?, ?)",
            payload,
            dim,
            "vec_bulk_insert_failed",
            note_id=note_id,
        )

    def _vec_write(self, conn, sql: str, payload: list, dim: int,
                   event: str, **context) -> None:
        """Write to ``embeddings_vec``, recreating the table once if it vanished.

        ``_vec_dim`` is Python state set inside the transaction that creates
        ``embeddings_vec``. If that transaction later rolls back, the DDL is
        undone but the attribute survives, so every subsequent write skips
        creation and inserts into a table that no longer exists. Previously
        that surfaced as a warning and nothing else: the mirror stayed silently
        empty for the rest of the process while ``embeddings`` kept filling up,
        leaving a vault whose vec index answers with a fraction of its chunks.

        Recovery happens inside the same call rather than on the next write,
        because the vectors of the failing batch would otherwise be lost for
        good -- their ``embeddings`` rows commit either way, so nothing would
        ever come back to mirror them. Clearing ``_vec_dim`` first restores the
        invariant "set only when the table exists"; ``_create_vec_table`` is
        ``IF NOT EXISTS`` so the retry is safe even when the table was fine and
        the failure was something else.
        """
        try:
            conn.executemany(sql, payload)
            return
        except sqlite3.OperationalError as first_error:
            self._vec_dim = None
            try:
                self._create_vec_table(conn, dim)
                conn.executemany(sql, payload)
            except sqlite3.OperationalError as retry_error:
                # Genuinely unwritable (not just a rolled-back table). Leave
                # _vec_dim cleared so a later write can try again, and let the
                # source-of-truth embeddings row stand.
                logger.warning(event, error=str(retry_error), **context)
                return
            self._vec_dim = dim
            logger.info("vec_table_recreated", dim=dim, after=str(first_error), **context)

    def _mirror_vec_insert(self, conn, rowid: int, vector_blob: bytes) -> None:
        """Insert the freshly-stored vector into ``embeddings_vec``.

        Skips silently when sqlite-vec isn't available or the vector dim
        doesn't match the table — the connector will keep using numpy in
        that case. Lazily creates the vec table on the first insert if
        the vault was empty at startup (so the dim is finally known).
        """
        if not self._vec_available:
            return
        dim = len(vector_blob) // 4
        if dim <= 0:
            return
        if self._vec_dim is None:
            self._create_vec_table(conn, dim)
            self._vec_dim = dim
        if dim != self._vec_dim:
            # Mid-flight dim mismatch (model swap without migration). Skip
            # the vec write so the source-of-truth ``embeddings`` row still
            # lands; the connector falls back to numpy until migrate-embeddings
            # rebuilds the vec table.
            return
        self._vec_write(
            conn,
            "INSERT INTO embeddings_vec(rowid, embedding) VALUES (?, ?)",
            [(rowid, vector_blob)],
            dim,
            "vec_insert_failed",
            rowid=rowid,
        )

    def _delete_embeddings_for_note(self, conn, note_id: int) -> None:
        """Delete a note's embeddings (and their vec mirror) on an existing
        connection, *without* committing.

        The FTS index follows automatically via the ``embeddings_ad`` AFTER
        DELETE trigger, but ``embeddings_vec`` is kept in sync by explicit
        dual-write (it has no trigger), so its rows must be cleared here too.
        Callers that forget this leave orphaned vectors that still occupy
        KNN slots in :py:meth:`vec_search` and silently shrink results.
        The vec rows are deleted first, while the source rows they select
        from still exist.
        """
        if self._vec_available and self._vec_dim is not None:
            conn.execute(
                "DELETE FROM embeddings_vec WHERE rowid IN "
                "(SELECT id FROM embeddings WHERE note_id = ?)",
                (note_id,),
            )
        conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))

    def delete_note_embeddings(self, note_id: int):
        """Delete all embeddings (and their vec mirror) for a note.

        Used before a full re-index and by the prune path. The vec mirror is
        cleared alongside the source rows — see
        :py:meth:`_delete_embeddings_for_note`.
        """
        with self._get_connection() as conn:
            self._delete_embeddings_for_note(conn, note_id)
            conn.commit()

    def get_chunk_records(
        self, note_id: int
    ) -> dict[int, tuple[Optional[str], Optional[int], Optional[str]]]:
        """Return ``{chunk_index: (chunk_hash, page, heading)}`` for a note.

        The incremental re-embed path uses the hash to detect changed
        *content* and the ``(page, heading)`` anchors to detect when a
        chunk's *position* moved — a pagination reflow earlier in the
        document — even though its text (and therefore its hash) is
        unchanged. Rows that pre-date the chunk_hash migration carry
        ``None`` for the hash and are treated as always-stale.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT chunk_index, chunk_hash, page, heading "
                "FROM embeddings WHERE note_id = ?",
                (note_id,),
            ).fetchall()
        return {int(r[0]): (r[1], r[2], r[3]) for r in rows}

    def get_chunk_hashes(self, note_id: int) -> dict[int, Optional[str]]:
        """Return ``{chunk_index: chunk_hash}`` for a note's stored chunks.

        Thin view over :py:meth:`get_chunk_records` for callers that only
        need the content fingerprint. Rows that pre-date the chunk_hash
        migration carry ``None`` here — the caller treats those as
        always-stale, which back-fills the column on first re-scan without
        an explicit migration pass.
        """
        return {idx: rec[0] for idx, rec in self.get_chunk_records(note_id).items()}

    def update_chunk_anchors(
        self,
        note_id: int,
        chunk_index: int,
        page: Optional[int],
        heading: Optional[str],
    ) -> None:
        """Refresh a kept chunk's citation anchors without re-embedding.

        The chunk_hash is computed over text only, so a chunk whose text is
        unchanged keeps its embedding across scans. But if an edit earlier in
        the document reflowed pagination, its stored ``page`` / ``heading`` can
        go stale and a citation like ``[[Title#p.42]]`` points at the wrong
        place. This updates just those two columns — the vector is untouched,
        so the vec / FTS mirrors need no sync.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE embeddings SET page = ?, heading = ? "
                "WHERE note_id = ? AND chunk_index = ?",
                (page, heading, note_id, chunk_index),
            )
            conn.commit()

    def delete_chunks(self, note_id: int, chunk_indices: Iterable[int]) -> int:
        """Drop a subset of a note's chunks by ``chunk_index``.

        Surgical complement to :py:meth:`delete_note_embeddings`: the
        incremental re-embed path uses this to evict only stale + surplus
        chunks while leaving unchanged rows (and their AUTOINCREMENT ids)
        in place. Returns the number of rows removed.
        """
        indices = list(chunk_indices)
        if not indices:
            return 0
        with self._get_connection() as conn:
            # SQLite's parameter limit is generous; even a 10k-chunk doc
            # comfortably fits one IN-list.
            placeholders = ",".join("?" * len(indices))
            if self._vec_available and self._vec_dim is not None:
                conn.execute(
                    f"DELETE FROM embeddings_vec WHERE rowid IN "
                    f"(SELECT id FROM embeddings WHERE note_id = ? "
                    f"AND chunk_index IN ({placeholders}))",
                    (note_id, *indices),
                )
            cur = conn.execute(
                f"DELETE FROM embeddings WHERE note_id = ? AND chunk_index IN ({placeholders})",
                (note_id, *indices),
            )
            conn.commit()
            return cur.rowcount or 0

    def drop_vec_table(self) -> None:
        """Tear down ``embeddings_vec`` (e.g. before a migrate-embeddings rebuild).

        Idempotent — silently does nothing when sqlite-vec is unavailable or
        the table doesn't exist. Resets ``_vec_dim`` so the next insert
        recreates the table at the new dim.
        """
        if not self._vec_available:
            return
        with self._get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS embeddings_vec")
            conn.commit()
        self._vec_dim = None

    def embeddings_total_bytes(self) -> int:
        """Approximate on-disk size of the ``embeddings.vector`` column.

        Used as a disk-pressure preflight in the migrate command: a
        rough doubling of this number must fit on the filesystem
        before we kick off a full re-embed.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(vector)), 0) FROM embeddings"
            ).fetchone()
        return int(row[0]) if row else 0

    def get_all_embeddings(self):
        """Returns all embeddings in the database for connection discovery."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT note_id, text_content, vector FROM embeddings")
            return cursor.fetchall()

    def get_all_embeddings_with_id(self):
        """
        Like :py:meth:`get_all_embeddings` but also returns the embedding's
        primary-key id, which is needed to align dense and FTS5 rankings.
        Rows: ``(id, note_id, text_content, vector)``.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, note_id, text_content, vector FROM embeddings"
            )
            return cursor.fetchall()

    def embeddings_signature(self) -> tuple[int, int]:
        """Cheap change-detection key for the embeddings table.

        Returns ``(row_count, max_id)``. ``id`` is an AUTOINCREMENT PK, so any
        insert bumps ``max_id`` and any delete changes the count — together
        they let :class:`~grimore.cognition.connector.Connector` cache its
        scoring matrix across queries in the warm shell session and skip a
        rebuild when the vault hasn't changed since the last ask.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM embeddings"
            ).fetchone()
            return (int(row[0]), int(row[1]))

    def get_cached_embedding(self, key: str) -> Optional[bytes]:
        """Retrieves a vector from the embedding cache if present."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT vector FROM embedding_cache WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def store_cached_embedding(self, key: str, vector_blob: bytes) -> None:
        """Stores a vector in the embedding cache."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)",
                (key, vector_blob),
            )
            conn.commit()
