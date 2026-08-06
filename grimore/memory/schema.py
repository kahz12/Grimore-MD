"""
Connection handling and schema lifecycle for the persistence layer.

Everything here runs before (or beneath) the domain-level data access:
opening tuned SQLite connections, creating the base tables, and the
idempotent migrations that upgrade an existing vault DB in place.
"""
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

from grimore.memory._base import DbBase

from grimore.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaMixin(DbBase):
    """Connection factory + schema init/migrations for :class:`Database`."""

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

    def _new_connection(self) -> sqlite3.Connection:
        """Open and tune one connection. Called once per thread, not per query.

        ``check_same_thread=False`` is safe here only because the connection is
        handed out through :meth:`_thread_conn`, which keeps exactly one per
        thread in thread-local storage. The flag exists so :meth:`close` can
        reclaim a connection from a thread other than the one that opened it
        during teardown.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # WAL (Write-Ahead Logging) lets the daemon write while the CLI reads.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Wait for a writer instead of failing instantly. WAL allows N readers
        # plus one writer, so with the daemon indexing while the CLI queries,
        # the loser of a write race previously surfaced as "database is
        # locked". Five seconds is far longer than any single statement here.
        conn.execute("PRAGMA busy_timeout=5000")
        # 16 MB page cache and a 256 MB mmap window. Both are per-connection
        # and were pointless when every query opened its own connection --
        # a cache that is discarded microseconds later never gets a hit.
        conn.execute("PRAGMA cache_size=-16000")
        conn.execute("PRAGMA mmap_size=268435456")
        # Load sqlite-vec so its virtual tables and MATCH operator are visible.
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

    def _thread_conn(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use.

        The generation counter is what makes reuse-after-close correct. A
        connection lives in the owning thread's local storage, so :meth:`close`
        -- which may run on a different thread -- cannot clear another thread's
        slot. Bumping the generation instead invalidates every cached handle at
        once, and each thread notices on its next call and reopens.
        """
        tl = self._local
        conn = getattr(tl, "conn", None)
        if conn is not None and getattr(tl, "generation", None) == self._generation:
            return conn
        conn = self._new_connection()
        tl.conn = conn
        tl.generation = self._generation
        with self._conn_lock:
            # Reap before inserting. A dead thread's local storage is gone, but
            # the registry still holds a strong reference, so without this the
            # connection and its file descriptor would live until close().
            # Thread pools that retire idle workers -- anyio's, which is what
            # Starlette runs sync handlers on -- churn threads for the life of
            # the process, so the leak is unbounded and ends in EMFILE.
            # Reaping here and not on every call keeps it off the hot path:
            # this branch runs once per thread, not once per query.
            self._reap_dead_locked()
            self._connections[threading.get_ident()] = (
                threading.current_thread(), conn,
            )
        return conn

    def _reap_dead_locked(self) -> None:
        """Close connections whose owning thread has exited. Caller holds the lock.

        Liveness is checked against the stored Thread object rather than the
        ident key: idents are recycled by the OS, so a new thread can inherit a
        dead one's ident and would otherwise look alive.
        """
        dead = [
            ident for ident, (thread, _conn) in self._connections.items()
            if not thread.is_alive()
        ]
        for ident in dead:
            _thread, conn = self._connections.pop(ident)
            try:
                conn.close()
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.warning("db_reap_failed", error=str(e))

    @contextmanager
    def _get_connection(self):
        """Yields this thread's SQLite connection, wrapped in a transaction.

        Context manager: commit on clean exit, rollback on exception — the
        same transactional semantics as sqlite3's own ``with conn:``. What
        changed is that the connection is no longer opened and closed per
        call. It used to be: every one of the 73 data-access methods paid a
        ``sqlite3.connect`` plus ``PRAGMA journal_mode=WAL`` (which rewrites
        the DB header) plus a fresh ``sqlite_vec.load``. ``store_embedding``
        runs once per chunk, so indexing 2000 notes opened 66396 connections
        and paid a WAL fsync on each commit -- the dominant cost of a scan.

        Reusing the connection is only safe because ``_get_connection`` is
        never re-entered while already held: nesting would let the inner
        ``with conn:`` commit the outer block's transaction early. That
        invariant is enforced by tests/test_connection_reuse.py.

        Lifetime is now explicit rather than automatic: whoever builds the
        Database must call :meth:`close` (Session.close and daemon.stop do).
        """
        conn = self._thread_conn()
        with conn:
            yield conn

    def close(self) -> None:
        """Close every connection this Database opened, from any thread.

        Idempotent, and safe to call while other threads still hold handles:
        they see the bumped generation and reopen on next use. The daemon's
        observer thread and the main thread each own one, so teardown has to
        reach across threads -- hence the registry rather than just clearing
        the caller's own thread-local slot.
        """
        with self._conn_lock:
            connections = [conn for _thread, conn in self._connections.values()]
            self._connections = {}
            self._generation += 1
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.warning("db_close_failed", error=str(e))

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
