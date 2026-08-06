"""
Note-level metadata access: upserts, lookups, dashboard counters,
category assignment, and the prune path that clears records for files
deleted from the vault.
"""
from datetime import datetime
from typing import Iterable, Optional

from grimore.memory._base import DbBase

# Host parameters per statement. SQLite's own limit is 999 on builds older
# than 3.32 and 32766 after; 900 clears the lower bound with room for the
# non-placeholder parameters a query may also carry.
_MAX_SQL_VARS = 900


def _escape_like(text: str) -> str:
    """
    Escape SQLite LIKE wildcards (``%``, ``_``) and the escape char itself
    (``\\``) so a literal user-supplied prefix matches only that prefix.

    Without this, a category named e.g. ``"50_off"`` in taxonomy.yml would
    also match ``"50aoff/sub"`` because the bare ``_`` is a one-character
    wildcard. Use together with ``LIKE ? ESCAPE '\\'`` in the query.
    """
    # Backslash first — otherwise we'd double-escape the % and _ we just
    # added a backslash in front of.
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class NotesMixin(DbBase):
    """Note metadata, categories, and prune for :class:`Database`."""

    def get_content_hash_by_path(self, path: str) -> Optional[str]:
        """
        Returns the stored content hash for ``path`` (or None if unknown).
        Named columns keep idempotency checks robust against schema drift.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT content_hash FROM notes WHERE path = ?", (path,)
            ).fetchone()
        return row[0] if row else None

    def get_note_location(self, note_id: int) -> Optional[tuple[str, str]]:
        """Returns ``(path, title)`` for the given note_id, or None if absent."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT path, title FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def get_note_writeback_target(
        self, note_id: int,
    ) -> Optional[tuple[str, str, Optional[str]]]:
        """Returns ``(source_path, format, sidecar_path)`` for ``note_id``.

        Used by the connector when deciding where to inject the suggested
        connections section: Markdown notes get the original path, every
        other format gets the sidecar path (or ``None`` if the user runs
        with ``write_sidecars = false`` and never materialised one).
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT path, format, sidecar_path FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        if not row:
            return None
        # `format` may be NULL on rows that pre-date the multiformat
        # migration's default — treat that as Markdown, the historical
        # behaviour.
        fmt = row[1] or "md"
        return (row[0], fmt, row[2])

    def get_note_title(self, note_id: int) -> Optional[str]:
        """Returns the title for the given note_id, or None if absent."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT title FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        return row[0] if row else None

    def get_note_titles(self, note_ids: Iterable[int]) -> dict[int, str]:
        """Batch form of :py:meth:`get_note_title`: ``{note_id: title}``.

        The Oracle used to call ``get_note_title`` once per retrieved chunk,
        which is one query (and, before the per-thread connection, one
        connection) per result. This collapses the whole pool into a single
        ``IN (...)``.

        Ids missing from the ``notes`` table are simply absent from the result
        rather than mapping to ``None`` — callers already treat a falsy title
        as an orphan embedding, so ``.get(note_id)`` reproduces the singular
        method's contract exactly. Duplicate ids are de-duplicated; order is
        irrelevant because the result is a mapping.
        """
        unique = {int(n) for n in note_ids}
        if not unique:
            return {}
        titles: dict[int, str] = {}
        ids = list(unique)
        with self._get_connection() as conn:
            # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER (999 on builds
            # predating 3.32). Retrieval pools are far smaller than this, but
            # eval sweeps can widen the pool arbitrarily.
            for start in range(0, len(ids), _MAX_SQL_VARS):
                batch = ids[start:start + _MAX_SQL_VARS]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT id, title FROM notes WHERE id IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
                titles.update({int(i): t for i, t in rows if t is not None})
        return titles

    def get_dashboard_stats(self) -> dict:
        """
        Aggregates the counters shown on the ``grimore status`` screen in a
        single round-trip. Keeps the CLI out of the raw connection.
        """
        with self._get_connection() as conn:
            total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            tagged_notes = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE last_tagged IS NOT NULL"
            ).fetchone()[0]
            total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            cached_embeddings = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
            categorised_notes = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE category IS NOT NULL AND category <> ''"
            ).fetchone()[0]
        return {
            "total_notes": int(total_notes),
            "tagged_notes": int(tagged_notes),
            "total_embeddings": int(total_embeddings),
            "cached_embeddings": int(cached_embeddings),
            "categorised_notes": int(categorised_notes),
        }

    def upsert_note(
        self,
        path: str,
        title: str,
        content_hash: str,
        *,
        format: str = "md",
        file_hash: Optional[str] = None,
        sidecar_path: Optional[str] = None,
        size_bytes: Optional[int] = None,
    ) -> Optional[int]:
        """Inserts or updates a note record. Returns the internal note ID.

        The multi-format kwargs are keyword-only with safe defaults so v2.0
        callers (``upsert_note(path, title, hash)``) keep working unchanged
        — they implicitly tag their notes as Markdown, which is what they
        always were.
        """
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            cursor = conn.execute("""
                INSERT INTO notes (path, title, content_hash, last_seen,
                                   format, file_hash, sidecar_path, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title        = excluded.title,
                    content_hash = excluded.content_hash,
                    last_seen    = excluded.last_seen,
                    format       = excluded.format,
                    file_hash    = COALESCE(excluded.file_hash, notes.file_hash),
                    sidecar_path = COALESCE(excluded.sidecar_path, notes.sidecar_path),
                    size_bytes   = COALESCE(excluded.size_bytes, notes.size_bytes)
                RETURNING id
            """, (path, title, content_hash, now,
                  format, file_hash, sidecar_path, size_bytes))
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None

    def update_last_tagged(self, path: str):
        """Updates the last_tagged timestamp for a note."""
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            conn.execute("UPDATE notes SET last_tagged = ? WHERE path = ?", (now, path))
            conn.commit()

    # ── Multi-format helpers ───────────────────────────────────────────────

    def get_file_hash(self, path: str) -> Optional[str]:
        """Raw-bytes SHA-256 for ``path`` (None if unknown).

        Cheap fast-skip key for ingest: if this matches what's on disk
        right now, the file genuinely hasn't changed and we can skip
        even the extraction step. See blueprint §6.4.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_hash FROM notes WHERE path = ?", (path,)
            ).fetchone()
        return row[0] if row and row[0] else None

    def update_file_hash(self, path: str, file_hash: str) -> None:
        """Refresh just the file_hash for an otherwise-unchanged note.

        Used when a save bumps the mtime + bytes but the extracted text
        is identical (e.g. a PDF re-exported with the same content).
        Keeps the fast-skip honest on the next scan.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET file_hash = ? WHERE path = ?",
                (file_hash, path),
            )
            conn.commit()

    # ── Categories ─────────────────────────────────────────────────────────

    def set_note_category(self, note_id: int, category: Optional[str]) -> None:
        """Set (or clear with ``None``) the canonical category path of a note."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET category = ? WHERE id = ?",
                (category, note_id),
            )
            conn.commit()

    def get_category_frequency(self) -> list[tuple[str, int]]:
        """All categories currently in use, sorted by number of notes (desc)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT category, COUNT(*) AS n
                FROM notes
                WHERE category IS NOT NULL AND category <> ''
                GROUP BY category
                ORDER BY n DESC, category ASC
                """
            ).fetchall()
        return [(name, count) for name, count in rows]

    def count_notes_under_category(self, category: str) -> int:
        """Count notes whose category is ``category`` or a descendant of it."""
        if not category:
            return 0
        prefix_pattern = _escape_like(category + "/") + "%"
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM notes
                WHERE category = ? OR category LIKE ? ESCAPE '\\'
                """,
                (category, prefix_pattern),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_notes_by_category(self, category: str, recursive: bool = True) -> list[tuple[int, str, str]]:
        """
        Return ``(id, path, title)`` for notes assigned to ``category``.
        When ``recursive`` is True (default) descendants are included too.
        """
        with self._get_connection() as conn:
            if recursive:
                prefix_pattern = _escape_like(category + "/") + "%"
                rows = conn.execute(
                    """
                    SELECT id, path, title FROM notes
                    WHERE category = ? OR category LIKE ? ESCAPE '\\'
                    ORDER BY category, title
                    """,
                    (category, prefix_pattern),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, path, title FROM notes WHERE category = ? ORDER BY title",
                    (category,),
                ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    # ── Prune ───────────────────────────────────────────────────────────────

    def find_stale_notes(self, existing_paths: Iterable[str]) -> list[tuple[int, str]]:
        """
        Identifies note records in the DB that no longer correspond to a file on disk.
        """
        existing = set(existing_paths)
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, path FROM notes").fetchall()
        return [(nid, path) for nid, path in rows if path not in existing]

    def prune_missing_notes(self, existing_paths: Iterable[str]) -> int:
        """
        Removes all database records (notes, tags association, embeddings)
        for files that have been deleted from the vault.
        """
        stale = self.find_stale_notes(existing_paths)
        if not stale:
            return 0
        with self._get_connection() as conn:
            for note_id, _ in stale:
                conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
                # Use the shared helper so the embeddings_vec mirror is cleared
                # too — a raw ``DELETE FROM embeddings`` would orphan vec rows.
                self._delete_embeddings_for_note(conn, note_id)
                conn.execute("DELETE FROM freshness WHERE note_path = (SELECT path FROM notes WHERE id = ?)", (note_id,))
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
        return len(stale)
