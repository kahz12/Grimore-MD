"""opt.1 regression: the Database reuses one SQLite connection per thread.

Before this change every data-access method opened its own connection, ran
PRAGMA journal_mode=WAL (which rewrites the DB header) and reloaded
sqlite-vec. store_embedding runs once per chunk, so indexing 2000 notes cost
66396 connections and a WAL fsync per commit.

Three properties have to hold for the reuse to be safe, and each has a test
here:

* one connection per thread, reused across calls, never shared between threads;
* the connection is genuinely released on close() -- the inverse of the v3.2.0
  fix, which added the close() this optimisation removes from the hot path;
* concurrent readers and a writer do not raise "database is locked", which is
  what busy_timeout buys and is the plan's non-negotiable gate for opt.1.
"""
import os
import sqlite3
import threading

import pytest

from grimore.memory.db import Database
from grimore.session import Session
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MemoryConfig,
    VaultConfig,
)


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "grimore.db"))
    yield database
    database.close()


def _fd_count() -> int:
    """Open file descriptors for this process (Linux only)."""
    return len(os.listdir("/proc/self/fd"))


class TestConnectionReuse:
    def test_repeated_calls_share_one_connection(self, db):
        for i in range(25):
            db.upsert_note(f"/vault/n{i}.md", f"N{i}", f"hash{i}")
        # One connection for _init_db and every upsert that followed.
        assert len(db._connections) == 1

    def test_the_same_connection_object_comes_back(self, db):
        with db._get_connection() as first:
            pass
        with db._get_connection() as second:
            pass
        assert first is second

    def test_pragmas_are_applied_once_at_open(self, db):
        conn = db._thread_conn()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        # busy_timeout is the setting the concurrency gate depends on; assert
        # it explicitly so a future edit to _new_connection cannot silently
        # drop it and leave the locking test passing by luck of timing.
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    def test_each_thread_gets_its_own_connection(self, db):
        # Hold the connection objects, not their id()s. Reaping frees a dead
        # thread's connection, and CPython happily reuses the address for the
        # next one -- comparing ids would report a false collision.
        seen: dict[int, sqlite3.Connection] = {}
        barrier = threading.Barrier(3)

        def worker():
            barrier.wait(timeout=30)
            seen[threading.get_ident()] = db._thread_conn()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(seen) == 3
        # Distinct connection objects: sharing one across threads would make
        # the transaction wrapping in _get_connection interleave between them.
        assert len({id(c) for c in seen.values()}) == 3


class TestExplicitClose:
    def test_close_releases_the_connection(self, tmp_path):
        database = Database(str(tmp_path / "grimore.db"))
        conn = database._thread_conn()
        database.close()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_close_does_not_leak_descriptors(self, tmp_path):
        # The inverse regression of the v3.2.0 fix: that change added a
        # guaranteed close() precisely because connections were leaking to the
        # GC. Holding one open per thread must not bring the leak back.
        before = _fd_count()
        for i in range(10):
            database = Database(str(tmp_path / f"db{i}.db"))
            database.upsert_note("/vault/a.md", "A", "h")
            database.close()
        assert _fd_count() <= before + 2

    def test_database_is_usable_again_after_close(self, tmp_path):
        database = Database(str(tmp_path / "grimore.db"))
        database.upsert_note("/vault/a.md", "A", "h1")
        database.close()
        # The generation bump invalidates the cached handle rather than
        # clearing it, so the next call has to transparently reopen.
        database.upsert_note("/vault/b.md", "B", "h2")
        assert len(database._connections) == 1
        database.close()

    def test_close_is_idempotent(self, tmp_path):
        database = Database(str(tmp_path / "grimore.db"))
        database.close()
        database.close()
        assert database._connections == {}

    def test_session_close_releases_the_db_connection(self, tmp_path):
        config = Config(
            vault=VaultConfig(path=str(tmp_path / "vault")),
            cognition=CognitionConfig(),
            memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        )
        session = Session(config)
        conn = session.db._thread_conn()
        session.close()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestDeadThreadReaping:
    """A connection outlives its thread unless something reclaims it.

    Thread-local storage dies with the thread, but the registry holds a strong
    reference, so the connection and its descriptor would survive until
    close(). Any pool that retires idle workers -- anyio's, which is what
    Starlette runs sync handlers on, and the ThreadPoolExecutor opt.7 would
    introduce -- churns threads for the life of the process, so the leak is
    unbounded and ends in EMFILE.
    """

    def test_short_lived_threads_do_not_accumulate_connections(self, db):
        db.upsert_note("/vault/a.md", "A", "h")

        for _ in range(4):
            threads = [
                threading.Thread(target=lambda: db.get_note_title(1), daemon=True)
                for _ in range(25)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # Reaping runs when a new connection is opened, so touch the DB from a
        # fresh thread to force one final sweep.
        final = threading.Thread(target=lambda: db.get_note_title(1), daemon=True)
        final.start()
        final.join(timeout=30)

        # 100 threads came and went; without reaping the registry would hold
        # over a hundred entries.
        assert len(db._connections) <= 5, (
            f"dead threads leaked connections: {len(db._connections)}"
        )

    def test_thread_pool_churn_does_not_accumulate_connections(self, db):
        from concurrent.futures import ThreadPoolExecutor

        db.upsert_note("/vault/a.md", "A", "h")

        # Five pools of eight workers: 40 worker threads created and joined.
        for _ in range(5):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: db.get_note_title(1), range(16)))

        # Every pool thread is joined and dead by now. Reaping happens when a
        # connection is opened, so touch the DB from one fresh thread to run
        # the sweep, then assert on the registry rather than on the process
        # descriptor count: how many descriptors a live pool holds mid-flight
        # varies with scheduling, which made the FD form of this assertion
        # flaky. The registry is the invariant that actually matters.
        sweeper = threading.Thread(target=lambda: db.get_note_title(1), daemon=True)
        sweeper.start()
        sweeper.join(timeout=30)

        # Left: this test's own thread and the sweeper's (reaped on the next
        # open). Without reaping there would be 40-plus.
        assert len(db._connections) <= 3, (
            f"pool threads leaked connections: {len(db._connections)}"
        )

    def test_reaping_leaves_live_threads_alone(self, db):
        # The sweep must not close a connection still in use by a running
        # thread; liveness is checked on the Thread object because the OS
        # recycles idents.
        errors: list[Exception] = []
        ready = threading.Event()
        release = threading.Event()

        def worker():
            try:
                db.get_note_title(1)
                ready.set()
                release.wait(timeout=30)
                db.get_note_title(1)
            except Exception as e:
                errors.append(e)

        holder = threading.Thread(target=worker, daemon=True)
        holder.start()
        assert ready.wait(timeout=30)

        for _ in range(20):
            t = threading.Thread(target=lambda: db.get_note_title(1), daemon=True)
            t.start()
            t.join(timeout=30)

        release.set()
        holder.join(timeout=30)
        assert errors == [], f"a live thread's connection was reaped: {errors!r}"


class TestConcurrency:
    """The plan's non-negotiable gate: daemon writing while the CLI reads."""

    def test_writer_and_readers_do_not_hit_database_is_locked(self, db):
        # Mirrors the real shape: the daemon indexes on the watchdog observer
        # thread while CLI/API queries read from another. WAL allows N readers
        # plus one writer; the loser of a race used to surface as an immediate
        # "database is locked" because no busy_timeout was set.
        errors: list[Exception] = []
        stop = threading.Event()
        start = threading.Barrier(4)

        def writer():
            start.wait(timeout=30)
            try:
                for i in range(150):
                    db.upsert_note(f"/vault/w{i}.md", f"W{i}", f"hash{i}")
            except Exception as e:
                errors.append(e)
            finally:
                stop.set()

        def reader():
            start.wait(timeout=30)
            try:
                while not stop.is_set():
                    db.get_dashboard_stats()
                    db.get_note_title(1)
                    db.get_content_hash_by_path("/vault/w1.md")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, daemon=True)]
        threads += [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not any(t.is_alive() for t in threads), "a worker hung"
        assert errors == [], f"concurrent access failed: {errors!r}"

    def test_concurrent_writes_are_not_lost_or_corrupted(self, db):
        # Correctness, not just absence of an exception: every row from every
        # thread must be present exactly once afterwards.
        errors: list[Exception] = []
        start = threading.Barrier(3)

        def writer(prefix: str):
            def run():
                start.wait(timeout=30)
                try:
                    for i in range(50):
                        db.upsert_note(f"/vault/{prefix}{i}.md", f"{prefix}{i}", f"h{i}")
                except Exception as e:
                    errors.append(e)
            return run

        threads = [threading.Thread(target=writer(p), daemon=True) for p in ("a", "b", "c")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"concurrent writes failed: {errors!r}"
        for prefix in ("a", "b", "c"):
            for i in range(50):
                assert db.get_content_hash_by_path(f"/vault/{prefix}{i}.md") == f"h{i}"
        assert db.get_dashboard_stats()["total_notes"] == 150

    def test_integrity_check_passes_after_concurrent_use(self, db):
        start = threading.Barrier(2)

        def worker(prefix: str):
            def run():
                start.wait(timeout=30)
                for i in range(60):
                    db.upsert_note(f"/vault/{prefix}{i}.md", f"{prefix}{i}", f"h{i}")
            return run

        threads = [threading.Thread(target=worker(p), daemon=True) for p in ("x", "y")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        conn = db._thread_conn()
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


class TestNoReentrancy:
    """Reuse is only transactionally safe while _get_connection is not nested.

    A nested block's inner ``with conn:`` would commit the outer block's
    transaction early, turning a rollback-on-error into a partial write. Static
    and runtime analysis both found zero nesting sites at the time of the
    change; this test keeps it that way.
    """

    def test_get_connection_is_never_re_entered_during_real_work(self, tmp_path):
        database = Database(str(tmp_path / "grimore.db"))
        depth = 0
        max_depth = 0
        original = type(database)._get_connection

        import contextlib

        @contextlib.contextmanager
        def tracking(self):
            nonlocal depth, max_depth
            depth += 1
            max_depth = max(max_depth, depth)
            try:
                with original(self) as conn:
                    yield conn
            finally:
                depth -= 1

        type(database)._get_connection = tracking
        try:
            note_id = database.upsert_note("/vault/a.md", "A", "h")
            database.upsert_tags(note_id, ["alpha", "beta"])
            database.set_note_category(note_id, "infra")
            database.get_dashboard_stats()
            database.get_note_title(note_id)
            database.prune_missing_notes([])
        finally:
            type(database)._get_connection = original
            database.close()

        assert max_depth == 1, f"_get_connection was re-entered (depth {max_depth})"
