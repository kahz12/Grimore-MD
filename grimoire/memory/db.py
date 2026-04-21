import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        # WAL lets the daemon write while the CLI reads.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_tags (
                    note_id INTEGER,
                    tag_id INTEGER,
                    FOREIGN KEY(note_id) REFERENCES notes(id),
                    FOREIGN KEY(tag_id) REFERENCES tags(id),
                    PRIMARY KEY(note_id, tag_id)
                )
            """)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_note_id ON embeddings(note_id)"
            )
            conn.commit()

    def get_note_by_path(self, path: str):
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE path = ?", (path,))
            return cursor.fetchone()

    def upsert_note(self, path: str, title: str, content_hash: str) -> int:
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
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            conn.execute("UPDATE notes SET last_tagged = ? WHERE path = ?", (now, path))
            conn.commit()

    def store_embedding(self, note_id: int, chunk_index: int, text_content: str, vector_blob: bytes):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO embeddings (note_id, chunk_index, text_content, vector)
                VALUES (?, ?, ?, ?)
            """, (note_id, chunk_index, text_content, vector_blob))
            conn.commit()

    def delete_note_embeddings(self, note_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            conn.commit()

    def get_all_embeddings(self):
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT note_id, text_content, vector FROM embeddings")
            return cursor.fetchall()

    def get_cached_embedding(self, key: str) -> Optional[bytes]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT vector FROM embedding_cache WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def store_cached_embedding(self, key: str, vector_blob: bytes) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)",
                (key, vector_blob),
            )
            conn.commit()

    # ── Tags ────────────────────────────────────────────────────────────────

    def upsert_tags(self, note_id: int, tag_names: list[str]) -> None:
        """
        Replace the association set of tags for a note. Creates new tag rows
        as needed; old rows are left in the ``tags`` table (they surface with
        zero frequency and can be purged separately).
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
            for name in tag_names:
                if not name:
                    continue
                conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
                row = conn.execute(
                    "SELECT id FROM tags WHERE name = ?", (name,)
                ).fetchone()
                if row is None:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                    (note_id, row[0]),
                )
            conn.commit()

    def get_tag_frequency(self, limit: int = 50) -> list[tuple[str, int]]:
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
        """Number of distinct tags currently associated with at least one note."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT nt.tag_id)
                FROM note_tags nt
                """
            ).fetchone()
        return int(row[0]) if row else 0

    def get_notes_by_tag(self, tag_name: str) -> list[tuple[int, str, str]]:
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
        """Remove tag rows that no longer belong to any note."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM note_tags)"
            )
            conn.commit()
            return cursor.rowcount or 0

    # ── Prune ───────────────────────────────────────────────────────────────

    def find_stale_notes(self, existing_paths: Iterable[str]) -> list[tuple[int, str]]:
        """
        Return (note_id, path) for every note whose ``path`` is not in
        ``existing_paths``. The caller supplies the live filesystem set.
        """
        existing = set(existing_paths)
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, path FROM notes").fetchall()
        return [(nid, path) for nid, path in rows if path not in existing]

    def prune_missing_notes(self, existing_paths: Iterable[str]) -> int:
        """
        Delete notes whose path is no longer on disk, along with their
        cascading rows in note_tags and embeddings.
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
