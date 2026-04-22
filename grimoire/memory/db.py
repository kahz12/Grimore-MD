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
            conn.commit()

    @staticmethod
    def _migrate_category_column(conn) -> None:
        """Add ``notes.category`` if it's missing (idempotent upgrade path)."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        if "category" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN category TEXT")

    def get_note_by_path(self, path: str):
        """Retrieves a note record by its file path."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE path = ?", (path,))
            return cursor.fetchone()

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
        prefix = category + "/"
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM notes
                WHERE category = ? OR category LIKE ?
                """,
                (category, prefix + "%"),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_notes_by_category(self, category: str, recursive: bool = True) -> list[tuple[int, str, str]]:
        """
        Return ``(id, path, title)`` for notes assigned to ``category``.
        When ``recursive`` is True (default) descendants are included too.
        """
        with self._get_connection() as conn:
            if recursive:
                prefix = category + "/"
                rows = conn.execute(
                    """
                    SELECT id, path, title FROM notes
                    WHERE category = ? OR category LIKE ?
                    ORDER BY category, title
                    """,
                    (category, prefix + "%"),
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
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
        return len(stale)
