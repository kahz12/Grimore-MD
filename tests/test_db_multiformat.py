"""
Multi-format DB migration tests.

The migration must be idempotent: running it on a v2.0 schema (notes
table without ``format`` / ``file_hash`` / ``sidecar_path`` / ``size_bytes``,
embeddings without ``page`` / ``heading``) must add the columns without
losing data, and running it again on the upgraded schema must be a no-op.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


from grimore.memory.db import Database


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _make_v2_schema(db_path: Path) -> None:
    """Re-create the v2.0 schema by hand so the migration has work to do."""
    with closing(sqlite3.connect(db_path)) as conn, conn:
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
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            notes_cols = _columns(conn, "notes")
            emb_cols = _columns(conn, "embeddings")
        for col in ("format", "file_hash", "sidecar_path", "size_bytes", "mtime_ns",
                    "extraction_profile", "extraction_json"):
            assert col in notes_cols, f"notes missing {col!r}"
        for col in ("page", "heading"):
            assert col in emb_cols, f"embeddings missing {col!r}"

    def test_upgrade_from_v2_preserves_rows_and_defaults_format(self, tmp_path):
        db_path = tmp_path / "v2.db"
        _make_v2_schema(db_path)

        # Trigger the migration.
        db = Database(str(db_path))

        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            row = conn.execute(
                "SELECT title, format, file_hash, sidecar_path, size_bytes, mtime_ns, "
                "extraction_profile, extraction_json "
                "FROM notes WHERE path = ?",
                ("/vault/old.md",),
            ).fetchone()
        assert row is not None
        title, fmt, file_hash, sidecar, size, mtime_ns, extraction_profile, extraction_json = row
        assert title == "Old Note"
        # Backfilled rows inherit the column default 'md'.
        assert fmt == "md"
        # The other new columns are NULL until something fills them in.
        assert file_hash is None
        assert sidecar is None
        assert size is None
        assert mtime_ns is None
        assert extraction_profile is None
        assert extraction_json is None

    def test_migration_is_idempotent(self, tmp_path):
        path = str(tmp_path / "twice.db")
        Database(path)
        # Repeating the migration leaves the schema unchanged.
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
            mtime_ns=5678,
        )
        assert nid is not None
        assert db.get_file_hash("/vault/book.pdf") == "f1"
        assert db.get_file_state("/vault/book.pdf") == ("f1", 1234, 5678)

        # Re-upsert with a new content_hash but no file_hash override —
        # the stored file_hash must be preserved via COALESCE.
        db.upsert_note("/vault/book.pdf", "Book", "ch2", format="pdf")
        assert db.get_file_hash("/vault/book.pdf") == "f1"

    def test_update_file_hash_round_trip(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        db.upsert_note("/vault/c.md", "C", "h", file_hash="aaa")
        db.update_file_hash("/vault/c.md", "bbb")
        assert db.get_file_hash("/vault/c.md") == "bbb"

    def test_update_file_state_round_trip(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        db.upsert_note("/vault/c.md", "C", "h")
        db.update_file_state("/vault/c.md", "abc", 42, 123456)
        assert db.get_file_state("/vault/c.md") == ("abc", 42, 123456)

    def test_pdf_extraction_profile_is_published_after_indexing(self, tmp_path):
        db = Database(str(tmp_path / "u.db"))
        note_id = db.upsert_note("/vault/book.pdf", "Book", "", format="pdf")
        assert db.get_ingest_state("/vault/book.pdf") == (None, None, None, "pdf", None)

        db.complete_note_indexing(
            note_id, "content", "file", 42, 123456,
            extraction_profile="pdf-layout-v1:engine=pypdf:ocr=0",
        )

        assert db.get_ingest_state("/vault/book.pdf") == (
            "file", 42, 123456, "pdf", "pdf-layout-v1:engine=pypdf:ocr=0",
        )


class TestStoreEmbeddingKwargs:
    def test_v2_call_shape_still_works(self, tmp_path):
        db = Database(str(tmp_path / "e.db"))
        nid = db.upsert_note("/vault/x.md", "X", "h")
        # The historical positional signature must keep working.
        db.store_embedding(nid, 0, "chunk", b"\x00" * 8)
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings WHERE note_id = ?",
                (nid,),
            ).fetchone()
        assert row == (None, None)

    def test_page_and_heading_persist(self, tmp_path):
        db = Database(str(tmp_path / "e.db"))
        nid = db.upsert_note("/vault/book.pdf", "Book", "h", format="pdf")
        db.store_embedding(nid, 0, "chunk", b"\x01" * 4, page=42, heading="Ch 3")
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings WHERE note_id = ?",
                (nid,),
            ).fetchone()
        assert row == (42, "Ch 3")


class TestDeleteNoteByPath:
    def test_removes_path_and_all_dependent_records(self, tmp_path):
        db = Database(str(tmp_path / "delete.db"))
        path = "/vault/gone.md"
        nid = db.upsert_note(path, "Gone", "hash")
        db.upsert_tags(nid, ["old"])
        db.store_embedding(nid, 0, "chunk", b"\x00" * 8)
        db.upsert_freshness(path, "2026-01-01T00:00:00", 30)
        db.replace_claims_for_note(
            path, [("A claim", 0, 7, b"\x00" * 8)],
            "2026-01-01T00:00:00Z",
        )

        assert db.delete_note_by_path(path) is True
        assert db.delete_note_by_path(path) is False
        with closing(sqlite3.connect(db.db_path)) as conn:
            for table in ("notes", "note_tags", "embeddings", "freshness", "claims"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
