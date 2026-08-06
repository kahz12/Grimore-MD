"""`connect` scores every note against every chunk in one blocked matmul.

The old pass called ``find_similar_notes`` once per note, and each of those
multiplied the query against every chunk in the vault. The batched form issues
the same arithmetic as a handful of large BLAS calls instead of thousands of
small ones.

The gate is parity: for the suggested links to be unchanged, the batch has to
reproduce the single-query path exactly -- including the parts that look like
implementation accidents, such as the self-note filter and the per-note dedup
running *after* the oversampled top-k cut.
"""
import random
import struct

import pytest

from grimore.cognition.connector import Connector
from grimore.memory.db import Database

_np = pytest.importorskip("numpy")


def _blob(floats) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def _unit(vec):
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class _StubEmbedder:
    model = "stub"

    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "grimore.db"))
    yield database
    database.close()


def _vault(db, *, notes: int, chunks_per_note: int, dim: int = 8, seed: int = 7):
    """A deterministic multi-chunk vault. Vectors are unit-normalized, as the
    embedder guarantees, so a dot product is a cosine."""
    rng = random.Random(seed)
    for n in range(notes):
        note_id = db.upsert_note(
            path=f"/v/n{n}.md", title=f"Note {n}", content_hash=f"{n:064d}",
        )
        for c in range(chunks_per_note):
            vec = _unit([rng.uniform(-1, 1) for _ in range(dim)])
            db.store_embedding(note_id, c, f"n{n} chunk {c}", _blob(vec))


def _connector(db):
    return Connector(db, _StubEmbedder(), vector_backend="numpy",
                     matrix_cache_enabled=False)


def _assert_same_hits(got, expected, tol=1e-5):
    """Same notes, same order, scores equal to within float32 accumulation noise.

    Bit-exact scores are not achievable and not the contract: the per-query
    path multiplies a matrix by a vector (gemv) while the batch multiplies two
    matrices (gemm), and BLAS accumulates them in a different order. Measured
    over 200 queries against 1000 chunks at 768 dims, the largest disagreement
    was 4.8e-07 -- four float32 eps -- with a median of 7.5e-09, and no query
    had a different top-20. The tolerance here is 20x that worst case, which
    still leaves any real ranking bug orders of magnitude outside it.
    """
    assert [[h["note_id"] for h in row] for row in got] == \
           [[h["note_id"] for h in row] for row in expected]
    assert [[h["text"] for h in row] for row in got] == \
           [[h["text"] for h in row] for row in expected]
    for got_row, exp_row in zip(got, expected, strict=True):
        for g, e in zip(got_row, exp_row, strict=True):
            assert g["score"] == pytest.approx(e["score"], abs=tol)


class TestBatchParity:
    """Every batched result must equal the single-query result for its query."""

    @pytest.mark.parametrize("dedupe", [True, False])
    @pytest.mark.parametrize("top_k", [1, 3, 12])
    def test_matches_the_per_query_path(self, db, dedupe, top_k):
        _vault(db, notes=12, chunks_per_note=4)
        conn = _connector(db)
        pairs = db.get_first_chunk_vectors()
        from grimore.cognition.embedder import Embedder
        queries = [Embedder.deserialize_vector(b) for _n, b in pairs]
        note_ids = [n for n, _b in pairs]

        expected = [
            conn.find_similar_notes(q, top_k=top_k, exclude_note_id=nid,
                                    dedupe_by_note=dedupe, with_text=False)
            for q, nid in zip(queries, note_ids, strict=True)
        ]
        got = conn.find_similar_notes_batch(
            queries, top_k=top_k, exclude_note_ids=note_ids,
            dedupe_by_note=dedupe, with_text=False,
        )
        _assert_same_hits(got, expected)

    def test_parity_holds_across_several_blocks(self, db):
        """Blocking must not change which notes come back, or their order.

        It does move the last bits of the scores: gemm accumulates differently
        depending on the block shape, so ``block_rows`` is not a pure
        performance knob. Same tolerance as the gemv/gemm comparison, and the
        ranking assertion is the part that has to hold exactly.
        """
        _vault(db, notes=10, chunks_per_note=3)
        conn = _connector(db)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        queries = [Embedder.deserialize_vector(b) for _n, b in pairs]
        note_ids = [n for n, _b in pairs]

        one_block = conn.find_similar_notes_batch(
            queries, top_k=5, exclude_note_ids=note_ids,
            dedupe_by_note=True, with_text=False, block_rows=1000)
        many_blocks = conn.find_similar_notes_batch(
            queries, top_k=5, exclude_note_ids=note_ids,
            dedupe_by_note=True, with_text=False, block_rows=3)
        _assert_same_hits(one_block, many_blocks)

    def test_mixed_excluded_and_plain_queries(self, db):
        """The oversample window widens by one only for queries that exclude a
        note, so it has to be computed per query, not once for the batch."""
        _vault(db, notes=8, chunks_per_note=2)
        conn = _connector(db)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        queries = [Embedder.deserialize_vector(b) for _n, b in pairs]
        excludes = [n if i % 2 == 0 else None
                    for i, (n, _b) in enumerate(pairs)]

        expected = [
            conn.find_similar_notes(q, top_k=4, exclude_note_id=ex,
                                    dedupe_by_note=True, with_text=False)
            for q, ex in zip(queries, excludes, strict=True)
        ]
        got = conn.find_similar_notes_batch(
            queries, top_k=4, exclude_note_ids=excludes,
            dedupe_by_note=True, with_text=False)
        _assert_same_hits(got, expected)

    def test_with_text_matches_too(self, db):
        _vault(db, notes=6, chunks_per_note=2)
        conn = _connector(db)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        queries = [Embedder.deserialize_vector(b) for _n, b in pairs]
        note_ids = [n for n, _b in pairs]

        expected = [conn.find_similar_notes(q, top_k=3, exclude_note_id=nid,
                                            dedupe_by_note=True)
                    for q, nid in zip(queries, note_ids, strict=True)]
        got = conn.find_similar_notes_batch(
            queries, top_k=3, exclude_note_ids=note_ids, dedupe_by_note=True)
        _assert_same_hits(got, expected)
        assert any(h["text"] for hits in got for h in hits)

    def test_empty_batch(self, db):
        _vault(db, notes=2, chunks_per_note=1)
        assert _connector(db).find_similar_notes_batch([]) == []

    def test_misaligned_excludes_are_rejected(self, db):
        _vault(db, notes=2, chunks_per_note=1)
        with pytest.raises(ValueError):
            _connector(db).find_similar_notes_batch(
                [[1.0] + [0.0] * 7], exclude_note_ids=[1, 2])


class TestFallbacks:
    def test_ragged_vectors_use_the_per_query_path(self, db):
        note_id = db.upsert_note(path="/v/r.md", title="R", content_hash="c" * 64)
        db.store_embedding(note_id, 0, "two", _blob([1.0, 0.0]))
        other = db.upsert_note(path="/v/s.md", title="S", content_hash="d" * 64)
        db.store_embedding(other, 0, "three", _blob([1.0, 0.0, 0.0]))

        conn = _connector(db)
        keys, matrix, blobs = conn._load_dense()
        assert matrix is None, "precondition: no matrix can be built"
        got = conn.find_similar_notes_batch([[1.0, 0.0]], top_k=2,
                                            with_text=False)
        assert len(got) == 1

    def test_query_of_the_wrong_width_falls_back(self, db):
        _vault(db, notes=4, chunks_per_note=1, dim=8)
        conn = _connector(db)
        expected = conn.find_similar_notes([1.0, 0.0], top_k=2, with_text=False)
        got = conn.find_similar_notes_batch([[1.0, 0.0]], top_k=2,
                                            with_text=False)
        _assert_same_hits(got, [expected])

    def test_vec_backend_delegates_per_query(self, db, monkeypatch):
        _vault(db, notes=4, chunks_per_note=1)
        conn = Connector(db, _StubEmbedder(), vector_backend="numpy")
        monkeypatch.setattr(conn, "_use_vec_backend", lambda: True)
        calls = []
        monkeypatch.setattr(
            conn, "find_similar_notes",
            lambda *a, **kw: calls.append(1) or [],
        )
        conn.find_similar_notes_batch([[1.0] + [0.0] * 7] * 3,
                                      exclude_note_ids=[None] * 3)
        assert len(calls) == 3


class TestFirstChunkVectors:
    def test_one_row_per_note_at_the_lowest_id(self, db):
        _vault(db, notes=3, chunks_per_note=4)
        pairs = db.get_first_chunk_vectors()
        assert len(pairs) == 3

        with db._get_connection() as conn:
            expected = conn.execute(
                "SELECT note_id, vector FROM embeddings WHERE id IN "
                "(SELECT MIN(id) FROM embeddings GROUP BY note_id) ORDER BY id"
            ).fetchall()
        assert pairs == [(int(n), v) for n, v in expected]

    def test_matches_what_the_old_scan_order_picked(self, db):
        """The old loop kept the first chunk it met per note while walking the
        table. Under a full scan that is the lowest id, which is what this
        selects -- now explicitly rather than by luck of the scan order.
        """
        _vault(db, notes=4, chunks_per_note=3)
        seen: dict[int, bytes] = {}
        for note_id, _text, blob in db.get_all_embeddings():
            seen.setdefault(note_id, blob)

        assert db.get_first_chunk_vectors() == list(seen.items())

    def test_notes_without_embeddings_are_absent(self, db):
        _vault(db, notes=2, chunks_per_note=1)
        db.upsert_note(path="/v/empty.md", title="Empty", content_hash="e" * 64)
        assert len(db.get_first_chunk_vectors()) == 2

    def test_empty_table(self, db):
        assert db.get_first_chunk_vectors() == []


class TestSuggestedLinkParity:
    """The end-to-end gate: the links `connect` would inject are unchanged."""

    def _links_via_loop(self, db, threshold):
        from grimore.cognition.embedder import Embedder
        conn = _connector(db)
        out = {}
        for note_id, blob in db.get_first_chunk_vectors():
            hits = conn.find_similar_notes(
                Embedder.deserialize_vector(blob), top_k=12,
                exclude_note_id=note_id, dedupe_by_note=True, with_text=False)
            picks = [h["note_id"] for h in hits if h["score"] > threshold][:3]
            if picks:
                out[note_id] = picks
        return out

    def _links_via_sweep(self, db, threshold):
        from grimore.cognition.embedder import Embedder
        conn = _connector(db)
        pairs = db.get_first_chunk_vectors()
        note_ids = [n for n, _b in pairs]
        batched = conn.find_similar_notes_batch(
            [Embedder.deserialize_vector(b) for _n, b in pairs],
            top_k=12, exclude_note_ids=note_ids, dedupe_by_note=True,
            with_text=False)
        out = {}
        for note_id, hits in zip(note_ids, batched, strict=True):
            picks = [h["note_id"] for h in hits if h["score"] > threshold][:3]
            if picks:
                out[note_id] = picks
        return out

    @pytest.mark.parametrize("threshold", [-1.0, 0.0, 0.3, 0.7])
    def test_identical_links_at_every_threshold(self, db, threshold):
        _vault(db, notes=25, chunks_per_note=3, seed=11)
        assert self._links_via_sweep(db, threshold) == \
            self._links_via_loop(db, threshold)

    def test_identical_with_a_single_chunk_per_note(self, db):
        _vault(db, notes=15, chunks_per_note=1, seed=3)
        assert self._links_via_sweep(db, 0.0) == self._links_via_loop(db, 0.0)

    def test_identical_when_notes_share_an_identical_chunk(self, db):
        """Tied scores are where a different top-k implementation shows up."""
        shared = _unit([1.0, 0.0, 0.0, 0.0])
        for n in range(6):
            note_id = db.upsert_note(
                path=f"/v/t{n}.md", title=f"T{n}", content_hash=f"{n:064d}")
            db.store_embedding(note_id, 0, "shared", _blob(shared))
            db.store_embedding(note_id, 1, f"own {n}",
                               _blob(_unit([0.0, 1.0, float(n), 0.0])))
        assert self._links_via_sweep(db, 0.0) == self._links_via_loop(db, 0.0)


class TestAbsoluteBehaviour:
    """Parity tests compare two paths that share the assembly helper, so a bug
    in that shared code passes them both. These pin the behaviour outright.
    """

    def test_a_note_never_suggests_itself(self, db):
        _vault(db, notes=10, chunks_per_note=3, seed=5)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        note_ids = [n for n, _b in pairs]
        batched = _connector(db).find_similar_notes_batch(
            [Embedder.deserialize_vector(b) for _n, b in pairs],
            top_k=12, exclude_note_ids=note_ids, dedupe_by_note=True,
            with_text=False)
        for note_id, hits in zip(note_ids, batched, strict=True):
            assert note_id not in [h["note_id"] for h in hits]

    def test_dedupe_returns_each_note_once(self, db):
        _vault(db, notes=8, chunks_per_note=5, seed=9)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        batched = _connector(db).find_similar_notes_batch(
            [Embedder.deserialize_vector(b) for _n, b in pairs],
            top_k=6, exclude_note_ids=[n for n, _b in pairs],
            dedupe_by_note=True, with_text=False)
        for hits in batched:
            ids = [h["note_id"] for h in hits]
            assert len(ids) == len(set(ids))

    def test_scores_are_sorted_descending(self, db):
        _vault(db, notes=8, chunks_per_note=3, seed=13)
        from grimore.cognition.embedder import Embedder
        pairs = db.get_first_chunk_vectors()
        batched = _connector(db).find_similar_notes_batch(
            [Embedder.deserialize_vector(b) for _n, b in pairs],
            top_k=5, exclude_note_ids=[n for n, _b in pairs],
            dedupe_by_note=True, with_text=False)
        for hits in batched:
            scores = [h["score"] for h in hits]
            assert scores == sorted(scores, reverse=True)


class TestOversampleWindow:
    """The dedupe oversample has to be computed per query. With a fixture
    small enough that the window covers the whole table, its size is
    unobservable -- so this builds one where it genuinely cuts.
    """

    def _crowded_vault(self, db):
        # One note owns 14 near-identical chunks that dominate any ranking;
        # a second note has a single, less similar chunk. With top_k=2 and
        # dedupe, a 12-wide window sees only the first note, a 15-wide one
        # reaches the second.
        near = db.upsert_note(path="/v/near.md", title="Near",
                              content_hash="a" * 64)
        for c in range(14):
            db.store_embedding(near, c, f"near {c}",
                               _blob(_unit([1.0, 0.001 * c])))
        far = db.upsert_note(path="/v/far.md", title="Far",
                             content_hash="b" * 64)
        db.store_embedding(far, 0, "far", _blob(_unit([0.6, 0.8])))
        return [1.0, 0.0]

    def test_window_width_is_observable_in_this_fixture(self, db):
        """Guards the guard: if the fixture stopped exercising the window, the
        test below would pass for the wrong reason."""
        query = self._crowded_vault(db)
        conn = _connector(db)
        narrow = conn.find_similar_notes_batch(
            [query], top_k=2, exclude_note_ids=[None],
            dedupe_by_note=True, with_text=False)[0]
        wide = conn.find_similar_notes_batch(
            [query], top_k=2, exclude_note_ids=[999],
            dedupe_by_note=True, with_text=False)[0]
        assert [h["note_id"] for h in narrow] != [h["note_id"] for h in wide], \
            "fixture no longer distinguishes the two window widths"

    def test_batch_uses_each_querys_own_window(self, db):
        query = self._crowded_vault(db)
        conn = _connector(db)
        # Same two queries, one excluding a note and one not, in one batch.
        expected = [
            conn.find_similar_notes(query, top_k=2, exclude_note_id=None,
                                    dedupe_by_note=True, with_text=False),
            conn.find_similar_notes(query, top_k=2, exclude_note_id=999,
                                    dedupe_by_note=True, with_text=False),
        ]
        got = conn.find_similar_notes_batch(
            [query, query], top_k=2, exclude_note_ids=[None, 999],
            dedupe_by_note=True, with_text=False)
        _assert_same_hits(got, expected)
