"""
Note-level metadata access: upserts, lookups, dashboard counters,
category assignment, and the prune path that clears records for files
deleted from the vault.
"""
import json
from datetime import datetime, timezone
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

    def get_note_id_by_path(self, path: str) -> Optional[int]:
        """Return an indexed note's ID without mutating its stored state."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM notes WHERE path = ?", (path,)
            ).fetchone()
        return int(row[0]) if row else None

    def resolve_note_filter(
        self,
        *,
        category: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        formats: Optional[Iterable[str]] = None,
    ) -> Optional[set[int]]:
        """Note ids matching every supplied criterion, or ``None`` for no filter.

        ``None`` and an empty set mean different things and callers depend on
        the difference: ``None`` is "no filter was asked for, search
        everything", while an empty set is "a filter was asked for and nothing
        matched", which must yield no results rather than silently searching
        the whole vault.

        Criteria combine with AND, and so do multiple ``tags`` -- the point of
        the feature is to narrow a pool, so ``--tag a --tag b`` means notes
        carrying both. Category matching is recursive: ``infra`` also selects
        ``infra/networking``, matching :py:meth:`get_notes_by_category`.
        Formats are compared lowercase, and a NULL ``format`` column (rows
        predating the multiformat migration) counts as ``md``.

        There is deliberately no date criterion. The only timestamps on a note
        are ``last_seen`` and ``last_tagged``, both of which record when the
        *scanner* touched the row, not anything about the document: after one
        scan every note in the vault shares a timestamp to the minute. A
        "modified after" filter built on that would rank by scan order, so it
        needs a real document date on the schema first.
        """
        clauses: list[str] = []
        params: list = []

        if category:
            clauses.append("(category = ? OR category LIKE ? ESCAPE '\\')")
            params += [category, _escape_like(category + "/") + "%"]

        fmt_list = [str(f).lower().lstrip(".") for f in (formats or []) if f]
        if fmt_list:
            placeholders = ",".join("?" * len(fmt_list))
            # COALESCE so legacy NULL rows behave as the Markdown they were.
            clauses.append(f"LOWER(COALESCE(format, 'md')) IN ({placeholders})")
            params += fmt_list

        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        for tag in tag_list:
            # One EXISTS per tag rather than IN + HAVING COUNT: it expresses
            # the AND directly and needs no grouping over the outer query.
            clauses.append(
                "EXISTS (SELECT 1 FROM note_tags nt JOIN tags t ON t.id = nt.tag_id "
                "WHERE nt.note_id = notes.id AND LOWER(t.name) = ?)"
            )
            params.append(tag.lower())

        if not clauses:
            return None

        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT id FROM notes WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchall()
        return {int(r[0]) for r in rows}

    def get_note_targets(
        self, note_ids: Iterable[int],
    ) -> dict[int, tuple[str, str, str, Optional[str]]]:
        """``{note_id: (path, title, format, sidecar_path)}`` in one query.

        ``connect`` needs all four per note and used to take them from
        :py:meth:`get_note_location` and
        :py:meth:`get_note_writeback_target` -- two statements per note over
        the whole vault, both keyed on the same id and both selecting ``path``.
        The singular methods stay for the callers that want one note.

        ``format`` is normalised to ``"md"`` when NULL, matching
        :py:meth:`get_note_writeback_target`: rows predating the multiformat
        migration have no format and were always Markdown.
        """
        unique = {int(n) for n in note_ids}
        if not unique:
            return {}
        ids = list(unique)
        out: dict[int, tuple[str, str, str, Optional[str]]] = {}
        with self._get_connection() as conn:
            for start in range(0, len(ids), _MAX_SQL_VARS):
                batch = ids[start:start + _MAX_SQL_VARS]
                rows = conn.execute(
                    "SELECT id, path, title, format, sidecar_path FROM notes "
                    f"WHERE id IN ({','.join('?' * len(batch))})",
                    tuple(batch),
                ).fetchall()
                for note_id, path, title, fmt, sidecar in rows:
                    out[int(note_id)] = (path, title, fmt or "md", sidecar)
        return out

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
        mtime_ns: Optional[int] = None,
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
                                   format, file_hash, sidecar_path, size_bytes,
                                   mtime_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title        = excluded.title,
                    content_hash = excluded.content_hash,
                    last_seen    = excluded.last_seen,
                    format       = excluded.format,
                    file_hash    = COALESCE(excluded.file_hash, notes.file_hash),
                    sidecar_path = COALESCE(excluded.sidecar_path, notes.sidecar_path),
                    size_bytes   = COALESCE(excluded.size_bytes, notes.size_bytes),
                    mtime_ns     = COALESCE(excluded.mtime_ns, notes.mtime_ns)
                RETURNING id
            """, (path, title, content_hash, now,
                  format, file_hash, sidecar_path, size_bytes, mtime_ns))
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None

    def update_last_tagged(self, path: str):
        """Updates the last_tagged timestamp for a note."""
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            conn.execute("UPDATE notes SET last_tagged = ? WHERE path = ?", (now, path))
            conn.commit()

    def complete_note_indexing(
        self, note_id: int, content_hash: str, file_hash: str,
        size_bytes: int, mtime_ns: int,
        *,
        extraction_profile: Optional[str] = None,
        extraction: Optional[dict] = None,
        published_at: Optional[str] = None,
        published_source: Optional[str] = None,
        modified_at: Optional[str] = None,
        modified_source: Optional[str] = None,
    ) -> None:
        """Publish change fingerprints only after all chunks were indexed.

        Ingest first upserts with empty hashes and negative stat values. Those
        sentinels cannot match a real file, so a failed or interrupted pass is
        retried even when its source has not changed.
        """
        indexed_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        ).replace("+00:00", "Z")
        extraction_json = json.dumps(
            extraction, ensure_ascii=False, separators=(",", ":"),
        ) if extraction is not None else None
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET content_hash = ?, file_hash = ?, "
                "size_bytes = ?, mtime_ns = ?, extraction_profile = ?, "
                "extraction_json = ? WHERE id = ?",
                (content_hash, file_hash, size_bytes, mtime_ns,
                 extraction_profile, extraction_json, note_id),
            )
            conn.execute(
                """
                INSERT INTO note_temporal
                    (note_id, published_at, published_source,
                     modified_at, modified_source, indexed_at, indexed_source)
                VALUES (?, ?, ?, ?, ?, ?, 'grimore:index')
                ON CONFLICT(note_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    published_source = excluded.published_source,
                    modified_at = excluded.modified_at,
                    modified_source = excluded.modified_source,
                    indexed_at = excluded.indexed_at,
                    indexed_source = excluded.indexed_source
                """,
                (
                    note_id, published_at, published_source,
                    modified_at, modified_source, indexed_at,
                ),
            )

    # ── Multi-format helpers ───────────────────────────────────────────────

    def get_file_hash(self, path: str) -> Optional[str]:
        """Raw-bytes SHA-256 for ``path`` (None if unknown).

        This supports the ingest fast path: matching raw bytes mean extraction
        and downstream processing can be skipped safely.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_hash FROM notes WHERE path = ?", (path,)
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_file_state(self, path: str) -> Optional[tuple[str, int, int]]:
        """Return ``(sha256, size, mtime_ns)`` when all values are known."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_hash, size_bytes, mtime_ns FROM notes WHERE path = ?",
                (path,),
            ).fetchone()
        if not row or row[0] is None or row[1] is None or row[2] is None:
            return None
        return str(row[0]), int(row[1]), int(row[2])

    def get_ingest_state(
        self, path: str,
    ) -> Optional[tuple[Optional[str], Optional[int], Optional[int], str, Optional[str]]]:
        """File fingerprint plus stored format and PDF extraction profile."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_hash, size_bytes, mtime_ns, format, "
                "extraction_profile FROM notes WHERE path = ?", (path,),
            ).fetchone()
        if not row:
            return None
        return (
            str(row[0]) if row[0] is not None else None,
            int(row[1]) if row[1] is not None else None,
            int(row[2]) if row[2] is not None else None,
            row[3] or "md", row[4],
        )

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

    def update_file_state(
        self, path: str, file_hash: str, size_bytes: int, mtime_ns: int,
    ) -> None:
        """Refresh the complete cheap-change fingerprint for ``path``."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET file_hash = ?, size_bytes = ?, mtime_ns = ? "
                "WHERE path = ?",
                (file_hash, size_bytes, mtime_ns, path),
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

    def _delete_note_row(self, conn, note_id: int, path: str) -> None:
        """Delete one note and every path/id keyed dependent record."""
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
        self._delete_embeddings_for_note(conn, note_id)
        conn.execute("DELETE FROM freshness WHERE note_path = ?", (path,))
        # Claims use note_path rather than a note FK. Enabling foreign keys
        # above makes their contradiction rows cascade away as intended.
        conn.execute("DELETE FROM claims WHERE note_path = ?", (path,))
        conn.execute("DELETE FROM note_temporal WHERE note_id = ?", (note_id,))
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def delete_note_by_path(self, path: str) -> bool:
        """Delete one indexed note by source path; return whether it existed."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, path FROM notes WHERE path = ?", (path,),
            ).fetchone()
            if row is None:
                return False
            self._delete_note_row(conn, int(row[0]), str(row[1]))
            conn.commit()
        return True

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
            for note_id, path in stale:
                self._delete_note_row(conn, note_id, path)
            conn.commit()
        return len(stale)
