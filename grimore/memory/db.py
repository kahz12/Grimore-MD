"""
Persistence Layer (SQLite).
This module manages the SQLite database, handling note metadata, tags,
and vector embeddings. It uses WAL mode to allow concurrent access.
"""
import re
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
        # sqlite-vec capability is probed once at startup. ``_vec_available``
        # gates every other vec-aware code path so a missing extension
        # degrades silently to the numpy fast path.
        self._vec_available: bool = self._probe_vec_extension()
        self._vec_dim: Optional[int] = None
        self._init_db()

    @staticmethod
    def _probe_vec_extension() -> bool:
        """One-shot check that the sqlite-vec extension loads on this Python.

        Two things can fail here: the stdlib sqlite3 was built without
        ``enable_load_extension`` (some distro packages), or the
        ``sqlite_vec`` wheel isn't installed. Both are silent fallbacks —
        the caller (Connector) treats the absence as "use numpy" without
        warning so a default Grimore install stays Termux-friendly.
        """
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except ImportError:
            return False
        probe = sqlite3.connect(":memory:")
        try:
            probe.enable_load_extension(True)
            sqlite_vec.load(probe)
        except (AttributeError, sqlite3.OperationalError, sqlite3.NotSupportedError):
            return False
        finally:
            probe.close()
        return True

    def _get_connection(self):
        """Creates a new SQLite connection with optimized settings for concurrency."""
        conn = sqlite3.connect(self.db_path)
        # WAL (Write-Ahead Logging) lets the daemon write while the CLI reads.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Load sqlite-vec on every connection so its virtual tables and
        # MATCH operator are visible. Cheap (~tens of µs) once the .so is
        # cached in the OS page cache.
        if self._vec_available:
            try:
                import sqlite_vec  # type: ignore[import-not-found]
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception as e:  # pragma: no cover - defensive
                # Flip the flag off so we don't keep retrying every call.
                logger.warning("sqlite_vec_load_failed", error=str(e))
                self._vec_available = False
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
            self._migrate_multiformat_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_format ON notes(format)"
            )
            self._fts_available = self._migrate_fts_index(conn)
            self._migrate_freshness_table(conn)
            self._migrate_mirror_tables(conn)
            if self._vec_available:
                self._migrate_vec_table(conn)
            self._migrate_embedding_migration_table(conn)
            conn.commit()

    @staticmethod
    def _migrate_embedding_migration_table(conn) -> None:
        """Bookkeeping table for in-flight embedding-model swaps (v2.3).

        The table is always present so a resumed migration after an
        interrupted run finds a place to land. The per-attempt
        ``embeddings_migration`` shadow table is created on demand by
        :py:meth:`begin_embedding_migration` and dropped on swap / abort.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                kind          TEXT NOT NULL,
                started_at    DATETIME NOT NULL,
                target_model  TEXT,
                total         INTEGER NOT NULL DEFAULT 0,
                done          INTEGER NOT NULL DEFAULT 0,
                status        TEXT NOT NULL DEFAULT 'running'
                              CHECK(status IN ('running','complete','aborted'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_migrations_status ON migrations(status)"
        )

    def _migrate_vec_table(self, conn) -> None:
        """Create / detect the ``embeddings_vec`` virtual table.

        Three cases:

        * No ``embeddings`` rows yet → defer creation. We don't know the
          embedding dim until the first vector lands; the next
          :py:meth:`store_embedding` call backfills here on its own
          before inserting.
        * Table exists → record its dim on ``self._vec_dim`` so the
          insert path can refuse mismatches.
        * Table missing but rows exist → create at the dim of the first
          row's vector and backfill from ``embeddings``.

        A dim-mismatched legacy table (e.g. user swapped embedding model
        without migrating) is left alone and ``_vec_available`` flipped
        off so the connector uses numpy; the user gets one warning in
        the log and nothing silently corrupts.
        """
        existing_dim = self._read_vec_table_dim(conn)
        if existing_dim is not None:
            self._vec_dim = existing_dim
            return

        row = conn.execute(
            "SELECT vector FROM embeddings WHERE vector IS NOT NULL LIMIT 1"
        ).fetchone()
        if row is None:
            # Empty vault. The table will be created lazily on first insert.
            self._vec_dim = None
            return

        dim = len(row[0]) // 4
        if dim <= 0:
            self._vec_dim = None
            return
        self._create_vec_table(conn, dim)
        self._backfill_vec_table(conn)
        self._vec_dim = dim

    def _create_vec_table(self, conn, dim: int) -> None:
        """Create ``embeddings_vec`` at ``dim``. Cosine distance for parity
        with the numpy path (vectors are unit-normalized at embed time, so
        cosine == dot product)."""
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )

    @staticmethod
    def _read_vec_table_dim(conn) -> Optional[int]:
        """Return the vec table's embedding dim, or None if the table is absent.

        The dim is parsed out of the ``sqlite_master`` DDL because vec0
        doesn't expose it via PRAGMA. A future sqlite-vec might change
        this surface; falling through to ``None`` triggers a safe
        rebuild path so a schema change doesn't wedge the DB.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'embeddings_vec'"
        ).fetchone()
        if not row or not row[0]:
            return None
        match = re.search(r"float\[(\d+)\]", row[0])
        return int(match.group(1)) if match else None

    @staticmethod
    def _backfill_vec_table(conn) -> int:
        """Copy every existing ``embeddings`` row into ``embeddings_vec``.

        Used on the first upgrade after a user installs sqlite-vec on an
        existing vault. Reuses each row's primary key as the vec rowid so
        the join back to ``embeddings`` is a straight rowid lookup.
        """
        rows = conn.execute(
            "SELECT id, vector FROM embeddings WHERE vector IS NOT NULL"
        ).fetchall()
        for rowid, blob in rows:
            conn.execute(
                "INSERT INTO embeddings_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
        if rows:
            logger.info("vec_table_backfilled", rows=len(rows))
        return len(rows)

    @staticmethod
    def _migrate_category_column(conn) -> None:
        """Add ``notes.category`` if it's missing (idempotent upgrade path)."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        if "category" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN category TEXT")

    @staticmethod
    def _migrate_multiformat_columns(conn) -> None:
        """Add the multi-format columns on ``notes`` and ``embeddings``.

        Idempotent — every ALTER is gated on a ``PRAGMA table_info`` check
        so re-running the migration on an already-upgraded DB is a no-op.
        New rows default to Markdown semantics so v2.0 callers that
        haven't been ported keep working unchanged.

        See ``docs/MULTIFORMAT_BLUEPRINT.md`` §5 for the contract.
        """
        note_cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        if "format" not in note_cols:
            # SQLite forbids non-constant DEFAULTs in ALTER TABLE, so the
            # literal 'md' both backfills existing rows and stamps future
            # inserts that omit the column.
            conn.execute("ALTER TABLE notes ADD COLUMN format TEXT DEFAULT 'md'")
        if "file_hash" not in note_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN file_hash TEXT")
        if "sidecar_path" not in note_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN sidecar_path TEXT")
        if "size_bytes" not in note_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN size_bytes INTEGER")

        emb_cols = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
        if "page" not in emb_cols:
            conn.execute("ALTER TABLE embeddings ADD COLUMN page INTEGER")
        if "heading" not in emb_cols:
            conn.execute("ALTER TABLE embeddings ADD COLUMN heading TEXT")
        # Chunk-level incremental re-embedding (v2.3) keys each row by a
        # content+model hash so an unchanged chunk skips a network round-trip
        # on re-scan. Legacy rows have NULL here and are treated as stale on
        # first contact, which back-fills naturally without an explicit pass.
        if "chunk_hash" not in emb_cols:
            conn.execute("ALTER TABLE embeddings ADD COLUMN chunk_hash TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_hash "
            "ON embeddings(note_id, chunk_hash)"
        )

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
    ) -> int:
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

    def set_sidecar_path(self, note_id: int, sidecar_path: Optional[str]) -> None:
        """Record (or clear with ``None``) the sidecar ``.md`` for a note."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET sidecar_path = ? WHERE id = ?",
                (sidecar_path, note_id),
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
            self._mirror_vec_insert(conn, new_rowid, vector_blob)
            conn.commit()

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
        try:
            conn.execute(
                "INSERT INTO embeddings_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, vector_blob),
            )
        except sqlite3.OperationalError as e:  # pragma: no cover - defensive
            logger.warning("vec_insert_failed", rowid=rowid, error=str(e))

    def delete_note_embeddings(self, note_id: int):
        """Deletes all embeddings associated with a note (usually before re-indexing)."""
        with self._get_connection() as conn:
            if self._vec_available and self._vec_dim is not None:
                # Drop the mirrored vec rows first so we don't leave orphans.
                conn.execute(
                    "DELETE FROM embeddings_vec WHERE rowid IN "
                    "(SELECT id FROM embeddings WHERE note_id = ?)",
                    (note_id,),
                )
            conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
            conn.commit()

    def get_chunk_hashes(self, note_id: int) -> dict[int, Optional[str]]:
        """Return ``{chunk_index: chunk_hash}`` for a note's stored chunks.

        Used by the incremental re-embed path to detect which chunks
        actually changed since the last scan. Rows that pre-date the
        chunk_hash migration carry ``None`` here — the caller treats
        those as always-stale, which back-fills the column on first
        re-scan without an explicit migration pass.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT chunk_index, chunk_hash FROM embeddings WHERE note_id = ?",
                (note_id,),
            ).fetchall()
        return {int(r[0]): r[1] for r in rows}

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

    # ── sqlite-vec search ─────────────────────────────────────────────────

    @property
    def vec_available(self) -> bool:
        """True only when the extension loaded AND a vec table is ready.

        The connector reads this to decide whether to skip the numpy
        matrix build entirely; if it ever returns True transiently and
        flips to False (extension misbehaves mid-session), the connector
        falls back to numpy on the next call without raising.
        """
        return bool(self._vec_available and self._vec_dim is not None)

    @property
    def vec_dim(self) -> Optional[int]:
        """Embedding dim currently held by ``embeddings_vec`` (None if absent)."""
        return self._vec_dim

    def vec_search(
        self,
        query_vector: bytes | list[float],
        limit: int,
        exclude_note_id: Optional[int] = None,
    ) -> list[tuple[int, int, str, float]]:
        """k-NN search against ``embeddings_vec``.

        Returns ``[(embedding_id, note_id, text_content, similarity), …]``
        sorted by descending similarity. ``similarity = 1 - cosine_distance``
        so the score range matches the numpy dot-product path and the
        connector can fuse with BM25 the same way.

        Caller filters ``exclude_note_id`` post-hoc by widening ``limit``
        a touch — same pattern as the numpy fast path. Safe no-op (empty
        list) when the vec table isn't available.
        """
        if not self.vec_available or limit <= 0:
            return []
        if isinstance(query_vector, list):
            import struct
            qbytes = struct.pack(f"{len(query_vector)}f", *query_vector)
        else:
            qbytes = bytes(query_vector)
        with self._get_connection() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.note_id, e.text_content, v.distance
                    FROM embeddings_vec v
                    JOIN embeddings e ON e.id = v.rowid
                    WHERE v.embedding MATCH ?
                      AND k = ?
                    ORDER BY v.distance ASC
                    """,
                    (qbytes, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("vec_query_failed", error=str(e))
                return []
        out: list[tuple[int, int, str, float]] = []
        for eid, nid, text, dist in rows:
            if exclude_note_id is not None and nid == exclude_note_id:
                continue
            out.append((int(eid), int(nid), text, 1.0 - float(dist)))
        return out

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

    def rebuild_vec_table(self) -> int:
        """Drop and rebuild ``embeddings_vec`` from the current ``embeddings``.

        Used after a model swap when the dim changes — the existing
        virtual table can't be altered in place. Returns the row count
        that was reinserted. No-op when sqlite-vec isn't loaded.
        """
        if not self._vec_available:
            return 0
        self.drop_vec_table()
        with self._get_connection() as conn:
            self._migrate_vec_table(conn)
            conn.commit()
        # _migrate_vec_table already backfilled when it found pre-existing rows.
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM embeddings_vec").fetchone()
        return int(row[0]) if row else 0

    # ── Embedding-model migration ────────────────────────────────────────

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
        intact. The vec table is dropped here (it was sized for the old
        dim) and rebuilt lazily on the next insert / explicit
        ``rebuild_vec_table`` call.

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
            # Drop the vec table first — it's keyed on rowid + dim, both of
            # which become wrong after the swap. The rebuild happens after.
            if self._vec_available:
                conn.execute("DROP TABLE IF EXISTS embeddings_vec")
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
            self._vec_dim = None
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
