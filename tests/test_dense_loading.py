"""The dense scoring matrix is built without dragging text_content along.

``_load_dense`` used to select ``id, note_id, text_content, vector`` for every
row and keep all of it cached, so the 500 chars stored per chunk were carried
through a pass that only ever reads the vectors. These tests pin the split:
the matrix comes from a vector-only load, the text comes back for the winners
alone, and the results are identical to what the combined load produced.
"""
import struct

import pytest

from grimore.cognition.connector import Connector
from grimore.cognition.embedder import Embedder
from grimore.memory.db import Database

_np = pytest.importorskip("numpy")


def _vec(*floats) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


class _StubEmbedder:
    model = "stub"

    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "grimore.db"))
    yield database
    database.close()


def _seed(db, rows):
    """rows: list of (note_title, text, vector floats)."""
    ids = []
    for n, (title, text, floats) in enumerate(rows):
        note_id = db.upsert_note(
            path=f"/v/{title}.md", title=title, content_hash=f"{n:064d}",
        )
        db.store_embedding(note_id, 0, text, _vec(*floats))
        ids.append(note_id)
    return ids


class TestMatrixParts:
    def test_keys_and_blob_align_and_exclude_text(self, db):
        _seed(db, [("a", "alpha text", (1.0, 0.0)),
                   ("b", "beta text", (0.0, 1.0))])
        keys, blob, dim = db.get_embedding_matrix_parts()

        assert dim == 2
        assert len(keys) == 2
        assert len(blob) == 2 * 2 * 4
        # The text is genuinely absent from what came back, which is the
        # whole point of the split.
        assert b"alpha text" not in blob and b"beta text" not in blob

    def test_row_order_is_by_id_and_stable(self, db):
        _seed(db, [(c, f"{c} text", (1.0, 0.0)) for c in "abcde"])
        first = db.get_embedding_matrix_parts()[0]
        second = db.get_embedding_matrix_parts()[0]
        assert first == second
        assert [k[0] for k in first] == sorted(k[0] for k in first)

    def test_empty_table(self, db):
        assert db.get_embedding_matrix_parts() == ([], b"", 0)

    def test_ragged_vectors_report_no_usable_buffer(self, db):
        """A model swapped without a re-scan leaves mixed widths. Concatenating
        those destroys the row boundaries, so the buffer must be refused.
        """
        note_id = db.upsert_note(path="/v/r.md", title="R", content_hash="c" * 64)
        db.store_embedding(note_id, 0, "two floats", _vec(1.0, 0.0))
        db.store_embedding(note_id, 1, "three floats", _vec(1.0, 0.0, 0.0))

        keys, blob, dim = db.get_embedding_matrix_parts()
        assert len(keys) == 2          # keys still describe every row
        assert (blob, dim) == (b"", 0)  # but the buffer is refused

    def test_ragged_lengths_that_sum_to_a_uniform_total_are_still_refused(self, db):
        """Widths 8, 4 and 12 sum to exactly 3 x 8, so a length-only check
        would accept them. The uniformity test has to be explicit.
        """
        note_id = db.upsert_note(path="/v/r2.md", title="R2", content_hash="d" * 64)
        db.store_embedding(note_id, 0, "two", _vec(1.0, 0.0))          # 8 bytes
        db.store_embedding(note_id, 1, "one", _vec(1.0))               # 4 bytes
        db.store_embedding(note_id, 2, "three", _vec(1.0, 0.0, 0.0))   # 12 bytes

        keys, blob, dim = db.get_embedding_matrix_parts()
        assert len(blob) == 0, "ragged rows summing to n*first must be refused"
        assert dim == 0 and len(keys) == 3

    def test_vectors_fallback_matches_matrix_order(self, db):
        _seed(db, [(c, f"{c} text", (float(i), 0.0)) for i, c in enumerate("abc")])
        keys, blob, dim = db.get_embedding_matrix_parts()
        vectors = db.get_embedding_vectors()
        assert len(vectors) == len(keys)
        assert b"".join(vectors) == blob


class TestChunkTexts:
    def test_returns_text_for_requested_ids_only(self, db):
        _seed(db, [("a", "alpha", (1.0, 0.0)), ("b", "beta", (0.0, 1.0))])
        keys, _blob, _dim = db.get_embedding_matrix_parts()
        first = keys[0][0]
        assert db.get_chunk_texts([first]) == {first: "alpha"}

    def test_empty_input(self, db):
        assert db.get_chunk_texts([]) == {}

    def test_unknown_ids_are_absent(self, db):
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        assert db.get_chunk_texts([99999]) == {}

    def test_chunks_past_the_host_parameter_limit(self, db, monkeypatch):
        from grimore.memory import chunks as chunks_mod
        monkeypatch.setattr(chunks_mod, "_MAX_SQL_VARS", 3)
        _seed(db, [(f"n{i}", f"text {i}", (1.0, 0.0)) for i in range(10)])
        keys, _b, _d = db.get_embedding_matrix_parts()
        ids = [k[0] for k in keys]
        got = db.get_chunk_texts(ids)
        assert got == {eid: f"text {n}" for n, eid in enumerate(ids)}


class TestBufferToMatrix:
    def test_matches_the_per_row_builder(self, db):
        blobs = [_vec(1.0, 0.0), _vec(0.0, 1.0), _vec(0.5, 0.5)]
        from_list = Embedder.vectors_to_matrix(blobs)
        from_buffer = Embedder.buffer_to_matrix(b"".join(blobs), 3, 2)
        assert _np.array_equal(from_list, from_buffer)

    def test_refuses_a_buffer_that_disagrees_with_the_shape(self):
        assert Embedder.buffer_to_matrix(_vec(1.0, 0.0), 2, 2) is None

    def test_empty_inputs(self):
        assert Embedder.buffer_to_matrix(b"", 0, 0) is None


class TestConnectorParity:
    """The observable results must be what the combined load produced."""

    def _connector(self, db):
        return Connector(db, _StubEmbedder(), vector_backend="numpy")

    def test_find_similar_notes_returns_the_winning_text(self, db):
        _seed(db, [("near", "the near chunk", (1.0, 0.0)),
                   ("far", "the far chunk", (0.0, 1.0))])
        hits = self._connector(db).find_similar_notes([1.0, 0.0], top_k=1)
        assert len(hits) == 1
        assert hits[0]["text"] == "the near chunk"
        assert hits[0]["score"] == pytest.approx(1.0)

    def test_vector_candidates_carry_text_and_embedding_id(self, db):
        _seed(db, [("near", "the near chunk", (1.0, 0.0)),
                   ("far", "the far chunk", (0.0, 1.0))])
        keys, _b, _d = db.get_embedding_matrix_parts()
        hits = self._connector(db)._vector_candidates([1.0, 0.0], limit=2)
        assert hits[0]["embedding_id"] == keys[0][0]
        assert hits[0]["text"] == "the near chunk"

    def test_ranking_matches_the_old_combined_load(self, db):
        """Reference implementation: score straight off the legacy loader and
        compare the resulting order. Guards against the split silently
        reordering or misaligning keys against matrix rows.
        """
        _seed(db, [(f"n{i}", f"text {i}", (1.0 - i / 10, i / 10))
                   for i in range(8)])
        query = [1.0, 0.0]

        legacy = db.get_all_embeddings_with_id()
        expected = sorted(
            (
                Embedder.dot_product(query, Embedder.deserialize_vector(r[3])),
                r[2],
            )
            for r in legacy
        )[::-1][:3]

        hits = self._connector(db).find_similar_notes(query, top_k=3)
        assert [h["text"] for h in hits] == [t for _s, t in expected]

    def test_scoring_still_works_when_vectors_are_ragged(self, db):
        """No matrix can be built, so the connector must fall back to the
        per-row path rather than returning nothing.
        """
        note_id = db.upsert_note(path="/v/r.md", title="R", content_hash="e" * 64)
        db.store_embedding(note_id, 0, "two floats", _vec(1.0, 0.0))
        db.store_embedding(note_id, 1, "three floats", _vec(0.0, 1.0, 0.0))

        keys, blob, dim = db.get_embedding_matrix_parts()
        assert (blob, dim) == (b"", 0), "precondition: no matrix is possible"

        hits = self._connector(db).find_similar_notes([1.0, 0.0], top_k=1)
        assert hits and hits[0]["text"] == "two floats"

    def test_cache_rebuilds_when_the_vault_changes(self, db):
        conn = self._connector(db)
        _seed(db, [("a", "first", (1.0, 0.0))])
        assert len(conn.find_similar_notes([1.0, 0.0], top_k=5)) == 1

        _seed(db, [("b", "second", (1.0, 0.0))])
        assert len(conn.find_similar_notes([1.0, 0.0], top_k=5)) == 2

    def test_blobs_are_not_retained_when_a_matrix_exists(self, db):
        """The point of the change: with a matrix in hand the raw bytes must
        not be cached alongside it.
        """
        _seed(db, [("a", "alpha", (1.0, 0.0))])
        conn = self._connector(db)
        keys, matrix, blobs = conn._load_dense()
        assert matrix is not None
        assert blobs is None
        assert keys and len(keys[0]) == 2


class TestWithTextFlag:
    """connect and the graph's suggested edges read only note_id and score.
    Since chunk text no longer rides along with the scoring matrix, fetching it
    for them would cost one query per note across the whole vault.
    """

    def _connector(self, db):
        return Connector(db, _StubEmbedder(), vector_backend="numpy")

    def test_disabled_skips_the_text_query(self, db):
        _seed(db, [("a", "alpha", (1.0, 0.0)), ("b", "beta", (0.0, 1.0))])
        conn = self._connector(db)
        calls = []
        original = db.get_chunk_texts
        db.get_chunk_texts = lambda ids: (calls.append(list(ids)) or original(ids))

        conn.find_similar_notes([1.0, 0.0], top_k=2, with_text=False)
        assert calls == [], "no text lookup should happen"

        conn.find_similar_notes([1.0, 0.0], top_k=2)
        assert len(calls) == 1, "the default still fetches text"

    def test_disabled_keeps_ranking_and_scores_intact(self, db):
        _seed(db, [(f"n{i}", f"text {i}", (1.0 - i / 10, i / 10))
                   for i in range(6)])
        conn = self._connector(db)
        with_text = conn.find_similar_notes([1.0, 0.0], top_k=4)
        conn._cache_sig = None
        without = conn.find_similar_notes([1.0, 0.0], top_k=4, with_text=False)

        assert [h["note_id"] for h in without] == [h["note_id"] for h in with_text]
        assert [h["score"] for h in without] == [h["score"] for h in with_text]
        assert all(h["text"] == "" for h in without)

    def test_default_is_on_so_existing_callers_are_unaffected(self, db):
        _seed(db, [("a", "alpha text", (1.0, 0.0))])
        hits = self._connector(db).find_similar_notes([1.0, 0.0], top_k=1)
        assert hits[0]["text"] == "alpha text"
