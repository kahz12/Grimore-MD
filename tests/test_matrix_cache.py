"""The dense matrix is cached on disk between processes.

A one-shot ``grimore ask`` rebuilds the whole matrix inside its only query.
The cache stores it as a ``.npy`` next to the database and reloads it
memory-mapped.

The interesting part is invalidation. ``swap_embedding_migration`` re-inserts
the migrated rows with their original ids, so ``(count, max_id)`` -- the
obvious seal -- is *identical* after switching embedding
model, while every vector has changed. These tests pin both the stronger seal
and the explicit clear that covers what the seal cannot see.
"""
import pathlib
import struct
import tempfile

import pytest

from grimore.cognition.connector import Connector
from grimore.memory.db import Database
from grimore.utils import matrix_cache

_np = pytest.importorskip("numpy")


def _vec(*floats) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


class _StubEmbedder:
    model = "stub"

    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "grimore.db")


@pytest.fixture
def db(db_path):
    database = Database(db_path)
    yield database
    database.close()


def _seed(db, rows):
    for n, (title, text, floats) in enumerate(rows):
        note_id = db.upsert_note(
            path=f"/v/{title}.md", title=title, content_hash=f"{n:064d}",
        )
        db.store_embedding(note_id, 0, text, _vec(*floats))


def _connector(db, enabled=True):
    return Connector(db, _StubEmbedder(), vector_backend="numpy",
                     matrix_cache_enabled=enabled)


class TestCachePaths:
    def test_derived_from_the_db_path(self, tmp_path):
        matrix, seal = matrix_cache.cache_paths(str(tmp_path / "g.db"))
        assert matrix.parent == tmp_path and seal.parent == tmp_path
        assert matrix.name.startswith("g.db") and matrix.suffix == ".npy"

    @pytest.mark.parametrize("path", [":memory:", "", "file::memory:?cache=shared"])
    def test_no_cache_for_in_memory_databases(self, path):
        assert matrix_cache.cache_paths(path) is None

    def test_cannot_escape_the_database_directory(self, tmp_path):
        # Containment is structural: the paths only ever rename the DB file,
        # so there is no input that walks them elsewhere.
        weird = tmp_path / "sub" / ".." / "g.db"
        matrix, seal = matrix_cache.cache_paths(str(weird))
        assert matrix.resolve().parent == tmp_path.resolve()
        assert seal.resolve().parent == tmp_path.resolve()


class TestRoundTrip:
    def test_written_on_first_build_and_reused_on_the_next(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0)), ("b", "beta", (0.0, 1.0))])
        matrix_path, seal_path = matrix_cache.cache_paths(db_path)
        assert not matrix_path.exists()

        first = _connector(db)._load_dense()[1]
        assert matrix_path.exists() and seal_path.exists()

        # A fresh Connector stands in for a fresh process.
        second_conn = _connector(db)
        calls = []
        original = db.get_embedding_matrix_parts
        db.get_embedding_matrix_parts = lambda: (calls.append(1) or original())
        keys, second, blobs = second_conn._load_dense()

        assert calls == [], "a cache hit must not read the vectors from SQLite"
        assert _np.array_equal(first, second)
        assert len(keys) == 2 and blobs is None

    def test_hit_is_memory_mapped(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix = _connector(db)._load_dense()[1]
        assert isinstance(matrix, _np.memmap)

    def test_results_are_identical_with_and_without_the_cache(self, db):
        _seed(db, [(f"n{i}", f"text {i}", (1.0 - i / 10, i / 10))
                   for i in range(6)])
        uncached = _connector(db, enabled=False).find_similar_notes(
            [1.0, 0.0], top_k=4)
        _connector(db)._load_dense()                       # populate
        cached = _connector(db).find_similar_notes([1.0, 0.0], top_k=4)
        assert cached == uncached


class TestInvalidation:
    def test_new_rows_invalidate(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        seal_path = matrix_cache.cache_paths(db_path)[1]
        first_seal = seal_path.read_text()

        _seed(db, [("b", "beta", (0.0, 1.0))])
        keys, matrix, _b = _connector(db)._load_dense()
        assert len(keys) == 2 and matrix.shape[0] == 2
        assert seal_path.read_text() != first_seal

    def test_deleted_rows_invalidate(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0)), ("b", "beta", (0.0, 1.0))])
        _connector(db)._load_dense()
        with db._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE text_content = 'beta'")
        keys, matrix, _b = _connector(db)._load_dense()
        assert len(keys) == 1 and matrix.shape[0] == 1

    def test_dimension_change_invalidates(self, db):
        """Same row count and same max id, wider vectors. Only the byte total
        in the seal can catch this."""
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        with db._get_connection() as conn:
            conn.execute("UPDATE embeddings SET vector = ?",
                         (_vec(1.0, 0.0, 0.0),))
        matrix = _connector(db)._load_dense()[1]
        assert matrix.shape == (1, 3)

    def test_same_shape_rewrite_is_caught_by_the_explicit_clear(self, db, db_path):
        """The migration swap re-inserts with the original ids, so count,
        max_id and total bytes are all unchanged while every vector differs.
        No seal can see this, which is why clear() exists.
        """
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        before_seal = db.matrix_cache_signature()
        _connector(db)._load_dense()

        with db._get_connection() as conn:
            conn.execute("UPDATE embeddings SET vector = ?", (_vec(0.0, 1.0),))
        assert db.matrix_cache_signature() == before_seal, \
            "precondition: the seal cannot distinguish this rewrite"

        # Without the clear the stale matrix is served...
        stale = _connector(db)._load_dense()[1]
        assert _np.array_equal(stale, _np.array([[1.0, 0.0]], dtype=_np.float32))

        # ...and with it, the truth comes back.
        matrix_cache.clear(db_path)
        fresh = _connector(db)._load_dense()[1]
        assert _np.array_equal(fresh, _np.array([[0.0, 1.0]], dtype=_np.float32))

    def test_migration_swap_clears_the_cache(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix_path = matrix_cache.cache_paths(db_path)[0]
        assert matrix_path.exists()

        db.begin_embedding_migration("other-model")
        for source_id, note_id, chunk_index, text, page, heading in \
                db.iter_pending_migration_rows():
            db.append_migration_row(
                source_id, note_id, chunk_index, text,
                _vec(0.0, 1.0), page, heading, None,
            )
        db.swap_embedding_migration()

        assert not matrix_path.exists(), \
            "the swap preserves ids, so the cache must be dropped outright"


class TestResilience:
    def test_corrupt_matrix_file_falls_back_to_sqlite(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix_path = matrix_cache.cache_paths(db_path)[0]
        matrix_path.write_bytes(b"not a numpy file")

        keys, matrix, _b = _connector(db)._load_dense()
        assert len(keys) == 1 and matrix is not None

    def test_truncated_matrix_with_a_valid_seal_is_a_miss(self, db, db_path):
        """A crash between the two writes can leave a short matrix under a
        seal that still validates, so the shape is re-checked on load."""
        _seed(db, [("a", "alpha", (1.0, 0.0)), ("b", "beta", (0.0, 1.0))])
        _connector(db)._load_dense()
        matrix_path = matrix_cache.cache_paths(db_path)[0]
        _np.save(matrix_path, _np.zeros((1, 2), dtype=_np.float32))

        keys, matrix, _b = _connector(db)._load_dense()
        assert matrix.shape == (2, 2), "wrong shape must rebuild, not be served"

    def test_missing_seal_is_a_miss(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix_cache.cache_paths(db_path)[1].unlink()
        assert _connector(db)._load_dense()[1] is not None

    def test_unwritable_directory_does_not_break_retrieval(self, db, db_path,
                                                           monkeypatch):
        _seed(db, [("a", "alpha", (1.0, 0.0))])

        def _boom(*_a, **_kw):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(matrix_cache._np, "save", _boom)
        keys, matrix, _b = _connector(db)._load_dense()
        assert len(keys) == 1 and matrix is not None

    def test_clear_is_safe_when_nothing_is_cached(self, db_path):
        matrix_cache.clear(db_path)
        matrix_cache.clear(":memory:")


class TestDisabled:
    def test_discard_reclaims_the_file(self, db, db_path):
        """The file is the size of every vector in the vault, so switching the
        feature off has to reclaim it rather than orphan it -- but through an
        explicit call, not as a side effect of building a Connector."""
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        conn = _connector(db)
        conn._load_dense()
        matrix_path, seal_path = matrix_cache.cache_paths(db_path)
        assert matrix_path.exists()

        conn.discard_matrix_cache()
        assert not matrix_path.exists() and not seal_path.exists()
        assert conn._cache_sig is None, "the in-memory copy must go too"

    def test_constructing_a_connector_deletes_nothing(self, db, db_path):
        """Guards against reintroducing the destructive side effect: a second
        Connector must not wipe a cache the first one is relying on."""
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix_path = matrix_cache.cache_paths(db_path)[0]

        _connector(db, enabled=False)
        _connector(db, enabled=True)
        assert matrix_path.exists()

    def test_flag_off_writes_nothing(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db, enabled=False)._load_dense()
        matrix_path, seal_path = matrix_cache.cache_paths(db_path)
        assert not matrix_path.exists() and not seal_path.exists()

    def test_flag_off_ignores_an_existing_cache(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()                       # populate
        calls = []
        original = db.get_embedding_matrix_parts
        db.get_embedding_matrix_parts = lambda: (calls.append(1) or original())
        _connector(db, enabled=False)._load_dense()
        assert calls == [1], "with the flag off the vectors come from SQLite"

    def test_ragged_vectors_never_populate_the_cache(self, db, db_path):
        note_id = db.upsert_note(path="/v/r.md", title="R", content_hash="c" * 64)
        db.store_embedding(note_id, 0, "two", _vec(1.0, 0.0))
        db.store_embedding(note_id, 1, "three", _vec(1.0, 0.0, 0.0))

        keys, matrix, blobs = _connector(db)._load_dense()
        assert matrix is None and blobs is not None
        assert not matrix_cache.cache_paths(db_path)[0].exists()


class TestConcurrentWrite:
    def test_a_write_during_the_build_skips_the_cache_write(self, db, db_path):
        """The daemon can index while a query builds the matrix. The resulting
        file would be sealed with a signature no later reader can match, so it
        would cost a write and then miss forever. Not a safety issue -- the
        shape check and clear() cover that -- but a pointless one.
        """
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        conn = _connector(db)

        original = db.get_embedding_matrix_parts

        def racing_build():
            result = original()
            _seed(db, [("b", "beta", (0.0, 1.0))])   # a writer lands here
            return result

        db.get_embedding_matrix_parts = racing_build
        conn._load_dense()

        matrix_path, seal_path = matrix_cache.cache_paths(db_path)
        assert not matrix_path.exists() and not seal_path.exists()

    def test_a_quiet_build_does_write_the_cache(self, db, db_path):
        # The control for the test above: without a racing writer the same
        # path must produce a cache, or the assertion above proves nothing.
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        assert matrix_cache.cache_paths(db_path)[0].exists()


class TestWriteSafety:
    def test_seal_is_removed_first(self, db, db_path, monkeypatch):
        """On Windows a memory-mapped matrix cannot be unlinked. Removing the
        seal first means the cache is invalidated even when the matrix file
        survives, which is the difference between wasted disk and serving a
        previous model's vectors.
        """
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        matrix_path, seal_path = matrix_cache.cache_paths(db_path)

        real_unlink = pathlib.Path.unlink

        def refuse_matrix(self, *a, **kw):
            if self.name.endswith(".npy"):
                raise OSError("mapped by another process")
            return real_unlink(self, *a, **kw)

        monkeypatch.setattr(pathlib.Path, "unlink", refuse_matrix)
        matrix_cache.clear(db_path)

        assert matrix_path.exists(), "precondition: the matrix could not be deleted"
        assert not seal_path.exists()
        # No seal means a miss, so the stale matrix is never served.
        keys, matrix, _b = _connector(db)._load_dense()
        assert not isinstance(matrix, _np.memmap)

    def test_temp_files_are_unique_per_writer(self, db, db_path, monkeypatch):
        """A fixed '<final>.tmp' name lets two processes sharing one database
        truncate each other's in-flight file."""
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        seen = []
        real = tempfile.mkstemp

        def spy(*a, **kw):
            fd, name = real(*a, **kw)
            seen.append(name)
            return fd, name

        monkeypatch.setattr(tempfile, "mkstemp", spy)
        matrix_cache.clear(db_path)
        _connector(db)._load_dense()
        matrix_cache.clear(db_path)
        _connector(db)._load_dense()

        assert len(seen) == len(set(seen)), f"temp names collided: {seen}"
        assert len(seen) >= 2

    def test_no_temp_files_survive_a_successful_write(self, db, db_path):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        leftovers = list(pathlib.Path(db_path).parent.glob("*.tmp"))
        assert leftovers == []

    def test_a_failed_write_leaves_no_temp_behind(self, db, db_path, monkeypatch):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        matrix_cache.clear(db_path)

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(matrix_cache._np, "save", boom)
        _connector(db)._load_dense()
        assert list(pathlib.Path(db_path).parent.glob("*.tmp")) == []


class TestInProcessInvalidation:
    def test_model_swap_invalidates_a_live_connector(self, db, db_path):
        """The disk cache is cleared by the swap, but a long-lived Connector
        keys on (count, max_id) — exactly the pair the swap preserves. Without
        the generation bump it would keep scoring against the old vectors for
        the rest of the process.
        """
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        conn = _connector(db)
        first = conn._load_dense()[1]
        assert _np.array_equal(first, _np.array([[1.0, 0.0]], dtype=_np.float32))

        db.begin_embedding_migration("other-model")
        for source_id, note_id, chunk_index, text, page, heading in \
                db.iter_pending_migration_rows():
            db.append_migration_row(source_id, note_id, chunk_index, text,
                                    _vec(0.0, 1.0), page, heading, None)
        db.swap_embedding_migration()

        assert db.embeddings_signature() == (1, 1), \
            "precondition: the cheap signature is unchanged by the swap"
        second = conn._load_dense()[1]
        assert _np.array_equal(second, _np.array([[0.0, 1.0]], dtype=_np.float32))

    def test_generation_starts_at_zero_and_only_the_swap_moves_it(self, db):
        assert db.data_generation == 0
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        _connector(db)._load_dense()
        assert db.data_generation == 0
