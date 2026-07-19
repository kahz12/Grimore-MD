"""
Embedding-model migration workflow: the resumable shadow-table dance
that re-embeds every chunk under a new model and atomically swaps the
result in. See ``grimore migrate-embeddings``.
"""
from datetime import datetime
from typing import Optional

from grimore.memory._base import DbBase


class EmbeddingMigrationMixin(DbBase):
    """Resumable embedding-model swap machinery for :class:`Database`."""

    def get_active_embedding_migration(self) -> Optional[dict]:
        """Return the in-flight migration row, if any.

        Resume relies on this: rerunning ``migrate-embeddings`` with the
        same target picks up the existing row instead of starting over.
        A target-mismatch (user changed their mind mid-migration) is
        treated as a hard error by the caller — they have to ``--abort``
        first.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, kind, started_at, target_model, total, done, status
                FROM migrations
                WHERE status = 'running' AND kind = 'embedding'
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]), "kind": row[1], "started_at": row[2],
            "target_model": row[3], "total": int(row[4]),
            "done": int(row[5]), "status": row[6],
        }

    def begin_embedding_migration(self, target_model: str) -> dict:
        """Start (or no-op resume) an embedding-model migration.

        On a fresh start: creates the shadow ``embeddings_migration``
        table mirroring the source shape, stamps a ``running`` row in
        ``migrations`` with ``total = COUNT(*) FROM embeddings``, and
        returns the new row. If a ``running`` row already exists for the
        same target, this is a silent no-op and returns the existing
        row (used for resume).
        """
        existing = self.get_active_embedding_migration()
        if existing is not None:
            if existing["target_model"] != target_model:
                raise ValueError(
                    f"another embedding migration is in flight for "
                    f"{existing['target_model']!r}; abort it first"
                )
            return existing

        with self._get_connection() as conn:
            # Shadow table mirrors the source — same anchor + chunk_hash columns.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_migration (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id       INTEGER,
                    chunk_index   INTEGER,
                    text_content  TEXT,
                    vector        BLOB,
                    page          INTEGER,
                    heading       TEXT,
                    chunk_hash    TEXT
                )
                """
            )
            total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            now = datetime.now().isoformat()
            cur = conn.execute(
                """
                INSERT INTO migrations (kind, started_at, target_model, total, done, status)
                VALUES ('embedding', ?, ?, ?, 0, 'running')
                """,
                (now, target_model, int(total)),
            )
            conn.commit()
            assert cur.lastrowid is not None  # always set after INSERT
            mid = int(cur.lastrowid)
        return {
            "id": mid, "kind": "embedding", "started_at": now,
            "target_model": target_model, "total": int(total),
            "done": 0, "status": "running",
        }

    def iter_pending_migration_rows(self) -> list[tuple[int, int, int, str, Optional[int], Optional[str]]]:
        """Source rows the worker still needs to re-embed.

        Returns ``(id, note_id, chunk_index, text_content, page, heading)``
        for every ``embeddings`` row whose primary key is greater than the
        max ``id`` already mirrored into ``embeddings_migration`` — so
        resume after an interrupted run skips finished work without a
        secondary "done" set lookup.
        """
        with self._get_connection() as conn:
            already = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM embeddings_migration"
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, note_id, chunk_index, text_content, page, heading
                FROM embeddings
                WHERE id > ?
                ORDER BY id ASC
                """,
                (int(already),),
            ).fetchall()
        return [(int(r[0]), int(r[1]), int(r[2]), r[3], r[4], r[5]) for r in rows]

    def append_migration_row(
        self,
        source_id: int,
        note_id: int,
        chunk_index: int,
        text_content: str,
        vector_blob: bytes,
        page: Optional[int],
        heading: Optional[str],
        chunk_hash: Optional[str],
    ) -> None:
        """Insert a re-embedded row into the shadow table at the source id.

        Reusing the source ``id`` is what makes resume cheap: the next
        :py:meth:`iter_pending_migration_rows` call simply picks up where
        the max(id) left off. ``done`` is advanced in the same
        transaction so an interrupted process can't double-count.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings_migration
                    (id, note_id, chunk_index, text_content, vector, page, heading, chunk_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, note_id, chunk_index, text_content,
                 vector_blob, page, heading, chunk_hash),
            )
            conn.execute(
                "UPDATE migrations SET done = done + 1 "
                "WHERE status = 'running' AND kind = 'embedding'"
            )
            conn.commit()

    def swap_embedding_migration(self) -> dict:
        """Atomically replace ``embeddings`` with the shadow contents.

        Single transaction so a crash mid-swap leaves the original table
        intact. The vec table (sized for the old vectors' rowids + dim) is
        dropped and rebuilt only *after* the swap commits, so a rolled-back
        swap leaves the existing vec index valid rather than missing.

        Returns the migration row in its new ``complete`` state.
        """
        active = self.get_active_embedding_migration()
        if active is None:
            raise RuntimeError("no in-flight embedding migration to swap")
        if active["done"] < active["total"]:
            raise RuntimeError(
                f"migration not done yet: {active['done']}/{active['total']}"
            )

        with self._get_connection() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM embeddings")
                conn.execute(
                    """
                    INSERT INTO embeddings
                        (id, note_id, chunk_index, text_content, vector,
                         page, heading, chunk_hash)
                    SELECT id, note_id, chunk_index, text_content, vector,
                           page, heading, chunk_hash
                    FROM embeddings_migration
                    """
                )
                conn.execute("DROP TABLE embeddings_migration")
                conn.execute(
                    "UPDATE migrations SET status = 'complete' WHERE id = ?",
                    (active["id"],),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        # Rebuild the FTS index and (if available) the vec table from the
        # newly-installed embeddings rows.
        with self._get_connection() as conn:
            if self.fts_available:
                conn.execute("INSERT INTO embeddings_fts(embeddings_fts) VALUES ('rebuild')")
                conn.commit()
        if self._vec_available:
            # Sized for the old vectors, so drop and rebuild from the
            # swapped-in embeddings. Only reached after COMMIT above, so a
            # rolled-back swap never leaves the vec index torn down.
            self.drop_vec_table()
            with self._get_connection() as conn:
                self._migrate_vec_table(conn)
                conn.commit()

        active["status"] = "complete"
        return active

    def abort_embedding_migration(self) -> Optional[dict]:
        """Tear down any in-flight migration: drop the shadow table and
        mark the row aborted. Returns the (now aborted) row, or ``None``
        if there was nothing to abort.
        """
        active = self.get_active_embedding_migration()
        if active is None:
            return None
        with self._get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS embeddings_migration")
            conn.execute(
                "UPDATE migrations SET status = 'aborted' WHERE id = ?",
                (active["id"],),
            )
            conn.commit()
        active["status"] = "aborted"
        return active
