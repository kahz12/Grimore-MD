"""
Persistence Layer (SQLite).
This module manages the SQLite database, handling note metadata, tags,
and vector embeddings. It uses WAL mode to allow concurrent access.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


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

class Database:
    """
    Manages all database operations for Project Grimoire.
    Ensures the schema is initialized and provides high-level methods for data access.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        """Creates a new SQLite connection with optimized settings for concurrency."""
        conn = sqlite3.connect(self.db_path)
        # WAL (Write-Ahead Logging) lets the daemon write while the CLI reads.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            # Table for note metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    title TEXT,
                    content_hash TEXT,
                    last_seen DATETIME,
                    last_tagged DATETIME
                )
            """)
            # Table for unique tag names
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE
                )
            """)
            # Junction table for many-to-many relationship between notes and tags
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_tags (
                    note_id INTEGER,
                    tag_id INTEGER,
                    FOREIGN KEY(note_id) REFERENCES notes(id),
                    FOREIGN KEY(tag_id) REFERENCES tags(id),
                    PRIMARY KEY(note_id, tag_id)
                )
            """)
            # Table for chunked vector embeddings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id INTEGER,
                    chunk_index INTEGER,
                    text_content TEXT,
                    vector BLOB,
                    FOREIGN KEY(note_id) REFERENCES notes(id)
                )
            """)
            # Cache table to avoid re-embedding the same text multiple times
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Speed up retrieval by note_id
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_note_id ON embeddings(note_id)"
            )
            self._migrate_category_column(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)"
            )
            self._fts_available = self._migrate_fts_index(conn)
            self._migrate_freshness_table(conn)
            conn.commit()

    @staticmethod
    def _migrate_category_column(conn) -> None:
        """Add ``notes.category`` if it's missing (idempotent upgrade path)."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        if "category" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN category TEXT")

    @staticmethod
    def _migrate_freshness_table(conn) -> None:
        """Create the Chronicler freshness table (idempotent).

        Only notes with a finite category window land here; exempt
        categories ("never stale") have no row at all.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS freshness (
                note_path       TEXT PRIMARY KEY,
                last_verified   TEXT NOT NULL,
                window_days     INTEGER NOT NULL,
                decay_check_at  TEXT,
                likely_stale    INTEGER
            )
            """
        )

    @staticmethod
    def _migrate_fts_index(conn) -> bool:
        """
        Create the FTS5 full-text index over ``embeddings.text_content`` plus
        synchronisation triggers. Uses external-content mode so FTS doesn't
        duplicate the chunk text. Returns False when the SQLite build lacks
        FTS5 support (the caller will then fall back to pure-vector search).

        Idempotent: safe to call on every startup.
        """
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
                    text_content,
                    content='embeddings',
                    content_rowid='id',
                    tokenize = "unicode61 remove_diacritics 2"
                )
                """
            )
        except sqlite3.OperationalError as e:
            logger.warning("fts5_unavailable", error=str(e))
            return False

        # Keep the FTS index in sync with the embeddings table.
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS embeddings_ai AFTER INSERT ON embeddings BEGIN
                INSERT INTO embeddings_fts(rowid, text_content)
                VALUES (new.id, new.text_content);
            END;
            CREATE TRIGGER IF NOT EXISTS embeddings_ad AFTER DELETE ON embeddings BEGIN
                INSERT INTO embeddings_fts(embeddings_fts, rowid, text_content)
                VALUES ('delete', old.id, old.text_content);
            END;
            CREATE TRIGGER IF NOT EXISTS embeddings_au AFTER UPDATE ON embeddings BEGIN
                INSERT INTO embeddings_fts(embeddings_fts, rowid, text_content)
                VALUES ('delete', old.id, old.text_content);
                INSERT INTO embeddings_fts(rowid, text_content)
                VALUES (new.id, new.text_content);
            END;
            """
        )

        # One-time rebuild when the FTS table exists but is empty and the
        # source table already has rows (fresh upgrade path).
        fts_count = conn.execute("SELECT COUNT(*) FROM embeddings_fts").fetchone()[0]
        src_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if fts_count == 0 and src_count > 0:
            conn.execute("INSERT INTO embeddings_fts(embeddings_fts) VALUES ('rebuild')")
            logger.info("fts_rebuilt", chunks=src_count)
        return True

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
        # FTS5 match string: quote to avoid operator parsing on user input.
        match = '"' + query.replace('"', '""') + '"'
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

    def get_note_by_path(self, path: str):
        """Retrieves a note record by its file path."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE path = ?", (path,))
            return cursor.fetchone()

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

    def get_note_title(self, note_id: int) -> Optional[str]:
        """Returns the title for the given note_id, or None if absent."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT title FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        return row[0] if row else None

    def get_dashboard_stats(self) -> dict:
        """
        Aggregates the counters shown on the ``grimoire status`` screen in a
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

    def upsert_note(self, path: str, title: str, content_hash: str) -> int:
        """Inserts or updates a note record. Returns the internal note ID."""
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            cursor = conn.execute("""
                INSERT INTO notes (path, title, content_hash, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    last_seen = excluded.last_seen
                RETURNING id
            """, (path, title, content_hash, now))
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None

    def update_last_tagged(self, path: str):
        """Updates the last_tagged timestamp for a note."""
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            conn.execute("UPDATE notes SET last_tagged = ? WHERE path = ?", (now, path))
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

    def store_embedding(self, note_id: int, chunk_index: int, text_content: str, vector_blob: bytes):
        """Stores a vector embedding for a specific note chunk."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO embeddings (note_id, chunk_index, text_content, vector)
                VALUES (?, ?, ?, ?)
            """, (note_id, chunk_index, text_content, vector_blob))
            conn.commit()

    def delete_note_embeddings(self, note_id: int):
        """Deletes all embeddings associated with a note (usually before re-indexing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            conn.commit()

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

    # ── Tags ────────────────────────────────────────────────────────────────

    def upsert_tags(self, note_id: int, tag_names: list[str]) -> None:
        """
        Syncs a note's tags with the database. 
        Ensures the 'tags' table has the tag names and updates 'note_tags' association.
        """
        with self._get_connection() as conn:
            # Clear old associations
            conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
            for name in tag_names:
                if not name:
                    continue
                # Ensure the tag exists globally
                conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
                row = conn.execute(
                    "SELECT id FROM tags WHERE name = ?", (name,)
                ).fetchone()
                if row is None:
                    continue
                # Associate tag with this note
                conn.execute(
                    "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                    (note_id, row[0]),
                )
            conn.commit()

    def get_tag_frequency(self, limit: int = 50) -> list[tuple[str, int]]:
        """Returns the most used tags and their frequencies."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT t.name, COUNT(nt.note_id) AS freq
                FROM tags t
                JOIN note_tags nt ON nt.tag_id = t.id
                GROUP BY t.id
                ORDER BY freq DESC, t.name ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(name, count) for name, count in rows]

    def get_tag_count(self) -> int:
        """Returns the number of distinct tags currently in use."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT nt.tag_id)
                FROM note_tags nt
                """
            ).fetchone()
        return int(row[0]) if row else 0

    def get_notes_by_tag(self, tag_name: str) -> list[tuple[int, str, str]]:
        """Retrieves all notes that have a specific tag."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.id, n.path, n.title
                FROM notes n
                JOIN note_tags nt ON nt.note_id = n.id
                JOIN tags t ON t.id = nt.tag_id
                WHERE t.name = ?
                ORDER BY n.title
                """,
                (tag_name,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def purge_unused_tags(self) -> int:
        """Deletes tags that are not associated with any notes."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM note_tags)"
            )
            conn.commit()
            return cursor.rowcount or 0

    # ── Maintenance primitives ──────────────────────────────────────────────

    def file_size_bytes(self) -> int:
        """On-disk byte size of the main DB file (0 if missing)."""
        try:
            return Path(self.db_path).stat().st_size
        except OSError:
            return 0

    def vacuum(self) -> dict:
        """
        Rewrite the DB file to reclaim free pages. Returns bytes freed and the
        pre/post sizes. VACUUM can't run inside a transaction and briefly holds
        an exclusive lock — the daemon schedules it during the low-traffic tick.
        """
        before = self.file_size_bytes()
        # autocommit (isolation_level=None) — VACUUM refuses to run otherwise.
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        after = self.file_size_bytes()
        return {"before_bytes": before, "after_bytes": after, "reclaimed_bytes": max(before - after, 0)}

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> dict:
        """
        Fold the WAL back into the main DB and optionally truncate the -wal
        sidecar. ``mode`` can be PASSIVE, FULL, RESTART, or TRUNCATE (default —
        the only mode that actually shrinks the sidecar on disk).

        Returns ``{busy, log_frames, checkpointed_frames}`` from SQLite.
        """
        mode_up = mode.upper()
        if mode_up not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError(f"invalid wal_checkpoint mode: {mode!r}")
        with self._get_connection() as conn:
            row = conn.execute(f"PRAGMA wal_checkpoint({mode_up})").fetchone()
        busy, log_frames, ckpt = (row or (0, 0, 0))
        return {"busy": int(busy), "log_frames": int(log_frames), "checkpointed_frames": int(ckpt)}

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
                conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
                conn.execute("DELETE FROM freshness WHERE note_path = (SELECT path FROM notes WHERE id = ?)", (note_id,))
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
        return len(stale)

    # ── Freshness (Chronicler) ─────────────────────────────────────────────

    def list_freshness(self) -> list[tuple[str, int]]:
        """``[(note_path, window_days), …]`` for every freshness row."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT note_path, window_days FROM freshness"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def upsert_freshness(self, path: str, last_verified: str, window_days: int) -> None:
        """Insert if missing; on conflict only the window updates.

        ``last_verified`` is preserved across re-seeds — that's the whole
        point of the verification timestamp. The user must call
        :py:meth:`touch_freshness_verified` to advance it explicitly.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO freshness (note_path, last_verified, window_days)
                VALUES (?, ?, ?)
                ON CONFLICT(note_path) DO UPDATE SET window_days = excluded.window_days
                """,
                (path, last_verified, window_days),
            )
            conn.commit()

    def delete_freshness(self, path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM freshness WHERE note_path = ?", (path,))
            conn.commit()

    def get_freshness_with_notes(self) -> list[tuple[str, str, Optional[str], str, int, Optional[int]]]:
        """Join used by ``chronicler list``.

        Returns rows of ``(path, title, category, last_verified,
        window_days, likely_stale)``.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.path, n.title, n.category,
                       f.last_verified, f.window_days, f.likely_stale
                FROM freshness f
                JOIN notes n ON n.path = f.note_path
                """
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def get_freshness_row(self, path: str) -> Optional[tuple[str, int, Optional[str], Optional[int]]]:
        """``(last_verified, window_days, decay_check_at, likely_stale)``."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT last_verified, window_days, decay_check_at, likely_stale
                FROM freshness
                WHERE note_path = ?
                """,
                (path,),
            ).fetchone()
        return (row[0], row[1], row[2], row[3]) if row else None

    def touch_freshness_verified(self, path: str) -> bool:
        """Advance ``last_verified`` to now and clear any decay verdict.

        Returns False if no freshness row exists for ``path`` — callers
        treat that as a silent no-op (the user verified a note that
        Chronicler doesn't track because of its category).
        """
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE freshness
                SET last_verified = ?, decay_check_at = NULL, likely_stale = NULL
                WHERE note_path = ?
                """,
                (now, path),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_freshness_decay(self, path: str, likely_stale: bool) -> None:
        """Persist the LLM decay verdict for ``path``."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE freshness
                SET decay_check_at = ?, likely_stale = ?
                WHERE note_path = ?
                """,
                (now, 1 if likely_stale else 0, path),
            )
            conn.commit()

    def get_notes_for_freshness_seed(self) -> list[tuple[str, Optional[str], Optional[str]]]:
        """All notes with the columns needed to seed freshness.

        Returns ``(path, category, last_tagged_or_seen)``. We prefer
        ``last_tagged`` over ``last_seen`` because tagging is the
        first time the note's content was understood; if a note hasn't
        been tagged yet, ``last_seen`` is the next-best anchor.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, category, COALESCE(last_tagged, last_seen) FROM notes"
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
