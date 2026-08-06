"""Regression: a note's chunks are written in one transaction.

The old re-embed path called store_embedding once per chunk, and each call
committed. In WAL mode a commit is an fsync, measured at ~7 ms across a
2000-note scan, which made the commit -- not the SQL -- the dominant cost of
indexing.

The risk the tests below guard is not speed but the ``embeddings_vec`` mirror.
It has no trigger; it is kept in sync by explicit dual-write, so the bulk path
has to resolve each new row's AUTOINCREMENT id correctly. Getting that wrong
files a vector under another chunk's id, and retrieval then answers with a
citation pointing at the wrong note -- silent, and invisible without a parity
check.
"""
from __future__ import annotations

import struct

import pytest

from grimore.cognition.reembed import Chunk, reembed_note
from grimore.memory.db import Database

VEC_AVAILABLE = Database._probe_vec_extension()
requires_vec = pytest.mark.skipif(
    not VEC_AVAILABLE, reason="sqlite-vec extension not loadable on this Python build"
)

DIM = 4


def _blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


class _Embedder:
    """Deterministic stand-in: the vector encodes the chunk text's length."""

    model = "stub-embed"

    @staticmethod
    def chunk_hash(text: str, model: str) -> str:
        return f"h:{model}:{text}"

    @staticmethod
    def serialize_vector(vector):
        return _blob(list(vector))

    def embed_batch(self, texts):
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]

    def embed(self, text):
        return [float(len(text)), 1.0, 0.0, 0.0]


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "bulk.db"))
    yield database
    database.close()


def _chunks(*texts, page=None, heading=None):
    return [Chunk(text=t, page=page, heading=heading) for t in texts]


def _rows(db, note_id):
    with db._get_connection() as conn:
        return conn.execute(
            "SELECT chunk_index, id, text_content, page, heading, chunk_hash, vector "
            "FROM embeddings WHERE note_id = ? ORDER BY chunk_index",
            (note_id,),
        ).fetchall()


class TestBulkWrite:
    def test_all_chunks_are_persisted(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        result = reembed_note(db, _Embedder(), note_id, _chunks("one", "two", "three"))
        assert result.embedded == 3
        assert result.stored == 3
        assert [r[0] for r in _rows(db, note_id)] == [0, 1, 2]

    def test_anchors_and_hashes_survive_the_bulk_path(self, db):
        note_id = db.upsert_note("/vault/p.pdf", "P", "hash-p")
        chunks = [
            Chunk(text="page one", page=1, heading="Intro"),
            Chunk(text="page two", page=2, heading="Body"),
        ]
        reembed_note(db, _Embedder(), note_id, chunks)
        rows = _rows(db, note_id)
        assert [(r[3], r[4]) for r in rows] == [(1, "Intro"), (2, "Body")]
        # chunk_hash is what the incremental path diffs against; a bulk write
        # that dropped it would make every later scan re-embed the whole note.
        assert all(r[5] for r in rows)

    def test_text_is_truncated_at_the_configured_width(self, db):
        note_id = db.upsert_note("/vault/long.md", "L", "hash-l")
        reembed_note(db, _Embedder(), note_id, _chunks("x" * 900), text_truncation=100)
        assert len(_rows(db, note_id)[0][2]) == 100

    def test_empty_chunk_list_writes_nothing(self, db):
        note_id = db.upsert_note("/vault/e.md", "E", "hash-e")
        assert db.store_embeddings_bulk(note_id, []) == 0
        assert _rows(db, note_id) == []

    def test_one_transaction_for_the_whole_note(self, db):
        note_id = db.upsert_note("/vault/t.md", "T", "hash-t")
        conn = db._thread_conn()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            reembed_note(db, _Embedder(), note_id, _chunks(*[f"c{i}" for i in range(10)]))
        finally:
            conn.set_trace_callback(None)
        # Ten chunks, one COMMIT. The per-chunk path emitted one per chunk,
        # and each is an fsync under WAL.
        commits = [s for s in statements if s.strip().upper().startswith("COMMIT")]
        assert len(commits) == 1, f"expected a single commit, got {len(commits)}"


class TestIncrementalStillWorks:
    """The bulk write must not disturb the kept/removed/re-anchored accounting."""

    def test_unchanged_chunks_are_kept_not_rewritten(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        embedder = _Embedder()
        reembed_note(db, embedder, note_id, _chunks("p1", "p2", "p3"))
        ids_before = {r[0]: r[1] for r in _rows(db, note_id)}

        result = reembed_note(db, embedder, note_id, _chunks("p1", "p2-edited", "p3"))
        assert result.kept == 2
        assert result.embedded == 1

        ids_after = {r[0]: r[1] for r in _rows(db, note_id)}
        # Untouched chunks keep their original rows; only the edited one is new.
        assert ids_after[0] == ids_before[0]
        assert ids_after[2] == ids_before[2]
        assert ids_after[1] != ids_before[1]

    def test_shrinking_a_note_removes_the_tail(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        embedder = _Embedder()
        reembed_note(db, embedder, note_id, _chunks("p1", "p2", "p3", "p4"))
        result = reembed_note(db, embedder, note_id, _chunks("p1", "p2"))
        assert result.removed == 2
        assert [r[0] for r in _rows(db, note_id)] == [0, 1]

    def test_a_failed_embed_skips_only_its_own_chunk(self, db):
        class _Flaky(_Embedder):
            def embed_batch(self, texts):
                return [
                    None if t == "broken" else [float(len(t)), 1.0, 0.0, 0.0]
                    for t in texts
                ]

        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        result = reembed_note(db, _Flaky(), note_id, _chunks("ok1", "broken", "ok2"))
        assert result.embedded == 2
        assert [r[0] for r in _rows(db, note_id)] == [0, 2]


@requires_vec
class TestVecMirrorParity:
    """The gate for the bulk path: the vec mirror stays row-for-row consistent."""

    def _vec_rows(self, db):
        with db._get_connection() as conn:
            return conn.execute("SELECT rowid FROM embeddings_vec ORDER BY rowid").fetchall()

    def test_row_counts_match_after_a_bulk_write(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        reembed_note(db, _Embedder(), note_id, _chunks(*[f"c{i}" for i in range(12)]))
        assert len(self._vec_rows(db)) == len(_rows(db, note_id)) == 12

    def test_every_vector_is_filed_under_its_own_row_id(self, db):
        # The failure this catches: ids derived arithmetically from lastrowid
        # instead of read back. A shifted mapping keeps the row counts equal,
        # so only comparing the stored vector to the source row detects it.
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        reembed_note(db, _Embedder(), note_id, _chunks(*texts))

        with db._get_connection() as conn:
            for chunk_index, rowid, _text, _p, _h, _hash, vector in _rows(db, note_id):
                stored = conn.execute(
                    "SELECT vec_to_json(embedding) FROM embeddings_vec WHERE rowid = ?",
                    (rowid,),
                ).fetchone()
                assert stored is not None, f"chunk {chunk_index} missing from the mirror"
                first = struct.unpack(f"{DIM}f", vector)[0]
                assert abs(float(stored[0].strip("[]").split(",")[0]) - first) < 1e-5

    def test_mirror_survives_an_incremental_re_embed(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        embedder = _Embedder()
        reembed_note(db, embedder, note_id, _chunks("p1", "p2", "p3"))
        reembed_note(db, embedder, note_id, _chunks("p1", "p2-edited", "p3"))
        assert len(self._vec_rows(db)) == len(_rows(db, note_id)) == 3

    def test_mirror_shrinks_with_the_note(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        embedder = _Embedder()
        reembed_note(db, embedder, note_id, _chunks("p1", "p2", "p3", "p4"))
        reembed_note(db, embedder, note_id, _chunks("p1", "p2"))
        assert len(self._vec_rows(db)) == len(_rows(db, note_id)) == 2

    def test_two_notes_do_not_cross_contaminate(self, db):
        # The read-back selects by note_id, so a second note's rows must not
        # leak into the first note's mapping.
        embedder = _Embedder()
        first = db.upsert_note("/vault/a.md", "A", "hash-a")
        second = db.upsert_note("/vault/b.md", "B", "hash-b")
        reembed_note(db, embedder, first, _chunks("a1", "a2"))
        reembed_note(db, embedder, second, _chunks("b1", "b2", "b3"))
        assert len(self._vec_rows(db)) == 5
        assert len(_rows(db, first)) == 2
        assert len(_rows(db, second)) == 3

    def test_dim_mismatch_still_lands_the_source_row(self, db):
        class _WideEmbedder(_Embedder):
            def embed_batch(self, texts):
                return [[1.0] * 8 for _ in texts]

        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        reembed_note(db, _Embedder(), note_id, _chunks("seed"))
        before = len(self._vec_rows(db))

        other = db.upsert_note("/vault/w.md", "W", "hash-w")
        reembed_note(db, _WideEmbedder(), other, _chunks("wide"))
        # embeddings is the source of truth and must accept the row; the
        # mirror skips it until migrate-embeddings rebuilds at the new dim.
        assert len(_rows(db, other)) == 1
        assert len(self._vec_rows(db)) == before


@requires_vec
class TestVecTableSelfHealing:
    """A rolled-back table creation must not disable the mirror for good.

    ``_vec_dim`` is set inside the transaction that creates ``embeddings_vec``.
    If that transaction rolls back, the DDL is undone but the attribute
    survives, so later writes skip creation and insert into a table that is no
    longer there. Before the fix that was a warning and nothing else: the
    mirror stayed empty for the rest of the process while ``embeddings`` kept
    filling up, leaving a vec index that answers with a fraction of the vault.

    Pre-existing behaviour -- the single-row path had it too -- so both are
    covered here.
    """

    def _rollback_via(self, db, attr, write):
        original = getattr(type(db), attr)

        def sabotaged(self, conn, *args, **kwargs):
            original(self, conn, *args, **kwargs)
            raise RuntimeError("failure later in the same transaction")

        setattr(type(db), attr, sabotaged)
        try:
            with pytest.raises(RuntimeError):
                write()
        finally:
            setattr(type(db), attr, original)

    def _vec_count(self, db):
        with db._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM embeddings_vec").fetchone()[0]

    def test_bulk_path_recovers_after_a_rolled_back_creation(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        rows = [
            {"chunk_index": i, "text_content": f"c{i}", "vector": _blob([float(i), 1, 0, 0]),
             "page": None, "heading": None, "chunk_hash": f"h{i}"}
            for i in range(3)
        ]
        self._rollback_via(
            db, "_mirror_vec_insert_many", lambda: db.store_embeddings_bulk(note_id, rows)
        )

        # The very next write must mirror completely, not just stop erroring:
        # its embeddings rows commit either way, so anything skipped here would
        # never be mirrored by anyone.
        reembed_note(db, _Embedder(), note_id, _chunks("p1", "p2"))
        assert self._vec_count(db) == len(_rows(db, note_id)) == 2

    def test_single_row_path_recovers_after_a_rolled_back_creation(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        self._rollback_via(
            db,
            "_mirror_vec_insert",
            lambda: db.store_embedding(
                note_id, 0, "c0", _blob([1.0, 1, 0, 0]), chunk_hash="h0"
            ),
        )

        db.store_embedding(note_id, 1, "c1", _blob([2.0, 1, 0, 0]), chunk_hash="h1")
        assert self._vec_count(db) == len(_rows(db, note_id)) == 1

    def test_recovery_clears_the_stale_dim(self, db):
        note_id = db.upsert_note("/vault/a.md", "A", "hash-a")
        rows = [{"chunk_index": 0, "text_content": "c", "vector": _blob([1.0, 1, 0, 0]),
                 "page": None, "heading": None, "chunk_hash": "h"}]
        self._rollback_via(
            db, "_mirror_vec_insert_many", lambda: db.store_embeddings_bulk(note_id, rows)
        )
        # Stale: the attribute still claims a table that the rollback removed.
        assert db._vec_dim == DIM

        db.store_embeddings_bulk(note_id, rows)
        assert db._vec_dim == DIM
        assert self._vec_count(db) == 1
