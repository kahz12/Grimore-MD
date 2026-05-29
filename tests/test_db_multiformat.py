"""
Multi-format DB migration tests.

The migration must be idempotent: running it on a v2.0 schema (notes
table without ``format`` / ``file_hash`` / ``sidecar_path`` / ``size_bytes``,
embeddings without ``page`` / ``heading``) must add the columns without
losing data, and running it again on the upgraded schema must be a no-op.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimore.memory.db import Database


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _make_v2_schema(db_path: Path) -> None:
    """Re-create the v2.0 schema by hand so the migration has work to do."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                title TEXT,
                content_hash TEXT,
                last_seen DATETIME,
                last_tagged DATETIME,
                category TEXT
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
            CREATE TABLE note_tags (
                note_id INTEGER, tag_id INTEGER,
                PRIMARY KEY(note_id, tag_id)
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER, chunk_index INTEGER,
                text_content TEXT, vector BLOB
            );
            CREATE TABLE embedding_cache (
                key TEXT PRIMARY KEY, vector BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO notes (path, title, content_hash)
            VALUES ('/vault/old.md', 'Old Note', 'deadbeef');
            """
        )
        conn.commit()


class TestMigration:
    def test_fresh_db_has_all_multiformat_columns(self, tmp_path):
        db = Database(str(tmp_path / "fresh.db"))
        with sqlite3.connect(db.db_path) as conn:
            notes_cols = _columns(conn, "notes")
            emb_cols = _columns(conn, "embeddings")
        for col in ("format", "file_hash", "sidecar_path", "size_bytes"):
            assert col in notes_cols, f"notes missing {col!r}"
        for col in ("page", "heading"):
            assert col in emb_cols, f"embeddings missing {col!r}"

    def test_upgrade_from_v2_preserves_rows_and_defaults_format(self, tmp_path):
        db_path = tmp_path / "v2.db"
        _make_v2_schema(db_path)

        # Trigger the migration.
        db = Database(str(db_path))

        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT title, format, file_hash, sidecar_path, size_bytes "
                "FROM notes WHERE path = ?",
                ("/vault/old.md",),
            ).fetchone()
        assert row is not None
        title, fmt, file_hash, sidecar, size = row
        assert title == "Old Note"
        # Backfilled rows inherit the column default 'md'.
        assert fmt == "md"
        # The other new columns are NULL until something fills them in.
        assert file_hash is None
        assert sidecar is None
        assert size is None

    def test_migration_is_idempotent(self, tmp_path):
        path = str(tmp_path / "twice.db")
        Database(path)
        # Second invocation must not raise (duplicate ALTER would error).
        Database(path)


class TestUpsertNoteKwargs:
    def test_v2_call_shape_still_works(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        nid = db.upsert_note("/vault/a.md", "A", "h1")
        assert nid is not None
        assert db.get_file_hash("/vault/a.md") is None  # not provided

    def test_multiformat_kwargs_persist(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        nid = db.upsert_note(
            "/vault/book.pdf",
            "Book",
            "ch1",
            format="pdf",
            file_hash="f1",
            sidecar_path="/vault/.grimore/sidecars/book.pdf.md",
            size_bytes=1234,
        )
        assert nid is not None
        assert db.get_file_hash("/vault/book.pdf") == "f1"

        # Re-upsert with a new content_hash but no file_hash override —
        # the stored file_hash must be preserved via COALESCE.
        db.upsert_note("/vault/book.pdf", "Book", "ch2", format="pdf")
        assert db.get_file_hash("/vault/book.pdf") == "f1"

    def test_update_file_hash_round_trip(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        db.upsert_note("/vault/c.md", "C", "h", file_hash="aaa")
        db.update_file_hash("/vault/c.md", "bbb")
        assert db.get_file_hash("/vault/c.md") == "bbb"


class TestStoreEmbeddingKwargs:
    def test_v2_call_shape_still_works(self, tmp_path):
        db = Database(str(tmp_path / "e.db"))
        nid = db.upsert_note("/vault/x.md", "X", "h")
        # The historical positional signature must keep working.
        db.store_embedding(nid, 0, "chunk", b"\x00" * 8)
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings WHERE note_id = ?",
                (nid,),
            ).fetchone()
        assert row == (None, None)

    def test_page_and_heading_persist(self, tmp_path):
        db = Database(str(tmp_path / "e.db"))
        nid = db.upsert_note("/vault/book.pdf", "Book", "h", format="pdf")
        db.store_embedding(nid, 0, "chunk", b"\x01" * 4, page=42, heading="Ch 3")
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings WHERE note_id = ?",
                (nid,),
            ).fetchone()
        assert row == (42, "Ch 3")
