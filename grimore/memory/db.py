"""
Persistence Layer (SQLite).
This module manages the SQLite database, handling note metadata, tags,
and vector embeddings. It uses WAL mode to allow concurrent access.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional
from grimore.utils.logger import get_logger

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
    Manages all database operations for Project Grimore.
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
            self._migrate_mirror_tables(conn)
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
    def _migrate_mirror_tables(conn) -> None:
        """Create the Black Mirror tables (idempotent).

        Three tables:

        * ``claims``         — atomic factual claims extracted from notes,
                               one row per claim with its embedding.
        * ``contradictions`` — claim-pairs the LLM flagged as conflicting.
                               Status is 'open' / 'dismissed' / 'resolved';
                               dismissed pairs persist across re-scans so
                               we don't re-flag what the user already saw.
        * ``mirror_runs``    — audit trail for incremental scans.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                note_path     TEXT NOT NULL,
                claim_text    TEXT NOT NULL,
                char_start    INTEGER,
                char_end      INTEGER,
                embedding     BLOB,
                extracted_at  TEXT NOT NULL,
                UNIQUE(note_path, claim_text)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_note_path ON claims(note_path)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contradictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_a_id    INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
                claim_b_id    INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
                severity      TEXT CHECK(severity IN ('low','medium','high')),
                explanation   TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'open'
                              CHECK(status IN ('open','dismissed','resolved')),
                detected_at   TEXT NOT NULL,
                resolved_at   TEXT,
                UNIQUE(claim_a_id, claim_b_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions(status)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_runs (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                ran_at                TEXT NOT NULL,
                notes_scanned         INTEGER,
                claims_extracted      INTEGER,
                pairs_checked         INTEGER,
                contradictions_found  INTEGER
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

    # ── Black Mirror: claims ─────────────────────────────────────────────

    def replace_claims_for_note(
        self,
        note_path: str,
        claims: list[tuple[str, Optional[int], Optional[int], Optional[bytes]]],
        extracted_at: str,
    ) -> int:
        """Replace this note's claim set, preserving claim ids for
        identical text across re-extractions.

        This is what makes the "dismissed contradictions stay dismissed"
        invariant work: cascading FK deletes only fire for claims whose
        text actually disappeared from the note. Claims whose text is
        unchanged keep their id (and therefore keep any contradiction
        rows that reference them, including dismissed ones).

        Each entry is ``(text, char_start, char_end, embedding_blob)``.
        ``embedding_blob`` may be None for claims whose embedding failed.
        Returns the *new* row count — i.e. claims that didn't exist for
        this note before this call.
        """
        new_count = 0
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            existing = {
                row[1]: row[0]
                for row in conn.execute(
                    "SELECT id, claim_text FROM claims WHERE note_path = ?",
                    (note_path,),
                )
            }
            new_texts = {text for text, _, _, _ in claims}
            for text, old_id in existing.items():
                if text not in new_texts:
                    conn.execute("DELETE FROM claims WHERE id = ?", (old_id,))
            for text, cs, ce, blob in claims:
                if text in existing:
                    conn.execute(
                        """
                        UPDATE claims
                        SET char_start = ?, char_end = ?, embedding = ?, extracted_at = ?
                        WHERE id = ?
                        """,
                        (cs, ce, blob, extracted_at, existing[text]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO claims
                            (note_path, claim_text, char_start, char_end, embedding, extracted_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (note_path, text, cs, ce, blob, extracted_at),
                    )
                    new_count += 1
            conn.commit()
        return new_count

    def get_claim_extraction_state(self) -> list[tuple[str, str]]:
        """``[(note_path, max_extracted_at), …]`` over every note that has
        at least one claim row. Used by Mirror.scan to decide which notes
        need re-extraction (mtime > extracted_at)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT note_path, MAX(extracted_at)
                FROM claims
                GROUP BY note_path
                """
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_claim_by_id(self, claim_id: int):
        """``(id, note_path, claim_text, char_start, char_end)`` or None."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, note_path, claim_text, char_start, char_end
                FROM claims WHERE id = ?
                """,
                (claim_id,),
            ).fetchone()
        return tuple(row) if row else None

    def get_all_claims_with_vectors(self) -> list[tuple[int, str, str, bytes]]:
        """``[(claim_id, note_path, claim_text, embedding_blob), …]``.

        Returns rows with non-NULL embeddings only — claim rows whose
        embedding failed to compute are excluded from neighbor search.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, note_path, claim_text, embedding
                FROM claims
                WHERE embedding IS NOT NULL
                """
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def count_claims(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
        return int(row[0]) if row else 0

    # ── Black Mirror: contradictions ─────────────────────────────────────

    def contradiction_pair_exists(self, claim_a_id: int, claim_b_id: int) -> bool:
        """Whether any row already references this canonical claim pair.

        Used by Mirror.scan to avoid re-running the LLM contradiction
        check for pairs we've already seen — including dismissed and
        resolved pairs (that's the whole point of dismissal-persistence).
        """
        a, b = (claim_a_id, claim_b_id) if claim_a_id < claim_b_id else (claim_b_id, claim_a_id)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM contradictions WHERE claim_a_id = ? AND claim_b_id = ? LIMIT 1",
                (a, b),
            ).fetchone()
        return row is not None

    def insert_contradiction(
        self,
        claim_a_id: int,
        claim_b_id: int,
        severity: str,
        explanation: str,
        detected_at: str,
    ) -> Optional[int]:
        """Insert a contradiction with canonical order ``a < b``.

        Returns the new row id, or ``None`` when the pair already exists
        (UNIQUE on (claim_a_id, claim_b_id)) — that's the dismissal-persists
        invariant: dismissed pairs aren't re-inserted as 'open'.
        """
        a, b = (claim_a_id, claim_b_id) if claim_a_id < claim_b_id else (claim_b_id, claim_a_id)
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO contradictions
                    (claim_a_id, claim_b_id, severity, explanation, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (a, b, severity, explanation, detected_at),
            )
            conn.commit()
            return cur.lastrowid if cur.rowcount else None

    def list_contradictions(self, status: Optional[str] = "open") -> list[tuple]:
        """Rows for ``mirror`` listing.

        Returns ``(id, severity, explanation, status, detected_at,
        note_a, note_b, claim_a_text, claim_b_text)`` joined with the
        claims table. ``status=None`` returns rows in every state.
        """
        sql = """
            SELECT c.id, c.severity, c.explanation, c.status, c.detected_at,
                   a.note_path, b.note_path,
                   a.claim_text, b.claim_text
            FROM contradictions c
            JOIN claims a ON a.id = c.claim_a_id
            JOIN claims b ON b.id = c.claim_b_id
        """
        params: tuple = ()
        if status is not None:
            sql += " WHERE c.status = ?"
            params = (status,)
        sql += """
            ORDER BY
              CASE c.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
              c.detected_at DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [tuple(r) for r in rows]

    def get_contradiction(self, contradiction_id: int):
        """Full row for ``mirror show <id>``.

        ``(id, severity, explanation, status, detected_at, resolved_at,
        claim_a_id, claim_b_id, note_a, note_b, claim_a_text,
        claim_b_text, char_start_a, char_end_a, char_start_b, char_end_b)``.
        """
        sql = """
            SELECT c.id, c.severity, c.explanation, c.status, c.detected_at,
                   c.resolved_at,
                   c.claim_a_id, c.claim_b_id,
                   a.note_path, b.note_path,
                   a.claim_text, b.claim_text,
                   a.char_start, a.char_end,
                   b.char_start, b.char_end
            FROM contradictions c
            JOIN claims a ON a.id = c.claim_a_id
            JOIN claims b ON b.id = c.claim_b_id
            WHERE c.id = ?
        """
        with self._get_connection() as conn:
            row = conn.execute(sql, (contradiction_id,)).fetchone()
        return tuple(row) if row else None

    def set_contradiction_status(self, contradiction_id: int, status: str) -> bool:
        """Update ``status`` and (when terminal) ``resolved_at``.

        Returns False if no row exists with that id, so the surface can
        say "no such contradiction" instead of silently doing nothing.
        """
        if status not in {"open", "dismissed", "resolved"}:
            raise ValueError(f"invalid contradiction status: {status!r}")
        now = datetime.now().isoformat() if status in {"dismissed", "resolved"} else None
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE contradictions
                SET status = ?, resolved_at = ?
                WHERE id = ?
                """,
                (status, now, contradiction_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def count_open_contradictions(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM contradictions WHERE status = 'open'"
            ).fetchone()
        return int(row[0]) if row else 0

    # ── Black Mirror: runs ───────────────────────────────────────────────

    def record_mirror_run(
        self,
        ran_at: str,
        notes_scanned: int,
        claims_extracted: int,
        pairs_checked: int,
        contradictions_found: int,
    ) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO mirror_runs
                    (ran_at, notes_scanned, claims_extracted,
                     pairs_checked, contradictions_found)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ran_at, notes_scanned, claims_extracted, pairs_checked, contradictions_found),
            )
            conn.commit()
            return int(cur.lastrowid)

    def latest_mirror_run_time(self) -> Optional[str]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(ran_at) FROM mirror_runs"
            ).fetchone()
        return row[0] if row and row[0] else None
