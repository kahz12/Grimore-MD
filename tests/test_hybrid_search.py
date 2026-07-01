"""Tests for FTS5-backed BM25 search and hybrid RRF retrieval."""
import struct

import pytest

from grimore.cognition.connector import Connector
from grimore.memory.db import Database


def _vec(values):
    """Serialize a list of floats into the same binary layout Embedder uses."""
    return struct.pack(f"{len(values)}f", *values)


class _StubEmbedder:
    """Minimal Embedder stand-in — we only need deserialization + dot product."""

    @staticmethod
    def deserialize_vector(data: bytes) -> list[float]:
        n = len(data) // 4
        return list(struct.unpack(f"{n}f", data))


def _seed(db: Database, rows):
    """
    Insert embeddings directly (bypassing the embedder). ``rows`` is a list of
    ``(note_title, chunk_text, vector)`` tuples; each note gets one chunk.
    """
    with db._get_connection() as conn:
        for i, (title, text, vec) in enumerate(rows):
            conn.execute(
                "INSERT INTO notes (path, title, content_hash) VALUES (?, ?, ?)",
                (f"{title}.md", title, f"hash-{i}"),
            )
            note_id = conn.execute(
                "SELECT id FROM notes WHERE path = ?", (f"{title}.md",)
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO embeddings (note_id, chunk_index, text_content, vector)
                   VALUES (?, ?, ?, ?)""",
                (note_id, 0, text, _vec(vec)),
            )
        conn.commit()


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "grimore.db"))


class TestFTSMigration:
    def test_fts_available_flag_set(self, db):
        # SQLite on CPython includes FTS5 by default; should be True here.
        assert db.fts_available is True

    def test_fts_table_created(self, db):
        with db._get_connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE name='embeddings_fts'"
            ).fetchone()
        assert row is not None


class TestFTSSearch:
    def test_inserted_text_is_indexed(self, db):
        _seed(db, [
            ("A", "Heidegger on the question of Being", [1.0, 0.0, 0.0]),
            ("B", "Dinner recipes with kale", [0.0, 1.0, 0.0]),
        ])
        hits = db.fts_search("Heidegger")
        assert len(hits) == 1
        _eid, _nid, text, score = hits[0]
        assert "Heidegger" in text
        assert score < 0  # SQLite bm25() returns negative values

    def test_deletion_removes_from_index(self, db):
        _seed(db, [("A", "unique-keyword alpha", [1.0, 0.0])])
        with db._get_connection() as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
        assert db.fts_search("unique-keyword") == []

    def test_empty_query_returns_empty(self, db):
        _seed(db, [("A", "hello world", [1.0, 0.0])])
        assert db.fts_search("") == []
        assert db.fts_search("   ") == []

    def test_handles_quotes_in_query(self, db):
        # Query with embedded double-quote shouldn't crash the MATCH parser.
        _seed(db, [("A", 'the word foo is present', [1.0, 0.0])])
        hits = db.fts_search('foo"bar')
        assert isinstance(hits, list)

    def test_multiword_query_matches_words_out_of_order(self, db):
        # Bag-of-words recall: the query words are all in the document but in a
        # different order and not adjacent. The old single-quoted-phrase MATCH
        # demanded the exact phrase and missed this; OR-ing the tokens restores
        # normal BM25 recall (regression guard for the phrase-only bug).
        _seed(db, [
            ("A", "the ribbed vault and the pointed arch of the cathedral", [1.0, 0.0]),
            ("B", "an unrelated note about kale", [0.0, 1.0]),
        ])
        hits = db.fts_search("pointed arch ribbed vault")
        assert len(hits) == 1
        _eid, _nid, text, _score = hits[0]
        assert "ribbed vault" in text

    def test_operators_in_query_are_neutralised(self, db):
        # FTS5 operators / special chars in user input are quoted per-token, so
        # they're treated as literal words and can't be parsed as MATCH syntax
        # (no crash, no injection).
        _seed(db, [("A", "alpha beta gamma", [1.0, 0.0])])
        for q in ("alpha OR beta", "alpha NEAR beta", "alpha*", "(alpha", "alpha AND"):
            assert isinstance(db.fts_search(q), list)
        # The real words still match through the operator noise.
        assert len(db.fts_search("alpha OR beta")) == 1


class TestHybridFusion:
    def test_vector_only_when_bm25_empty(self, db):
        _seed(db, [
            ("A", "alpha", [1.0, 0.0, 0.0]),
            ("B", "beta",  [0.0, 1.0, 0.0]),
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid(
            query_text="zzz-no-match-in-corpus",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
        )
        assert hits[0]["text"] == "alpha"

    def test_bm25_only_when_no_vector(self, db):
        _seed(db, [
            ("A", "keyword needle appears here", [1.0, 0.0, 0.0]),
            ("B", "a completely different subject", [0.0, 1.0, 0.0]),
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid(
            query_text="needle",
            query_vector=None,
            top_k=2,
        )
        assert hits[0]["text"].startswith("keyword needle")

    def test_fusion_boosts_doc_present_in_both(self, db):
        # A is top of BM25 (contains "needle") AND top of vector (matches query).
        # B is top of vector only. C is top of BM25 only.
        # A should win the fused ranking.
        _seed(db, [
            ("A", "needle matches query",           [1.0, 0.0, 0.0]),
            ("B", "unrelated corpus item",          [0.9, 0.1, 0.0]),
            ("C", "needle needle needle keyword",   [0.0, 0.0, 1.0]),
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid(
            query_text="needle",
            query_vector=[1.0, 0.0, 0.0],
            top_k=3,
        )
        assert hits[0]["text"] == "needle matches query"

    def test_respects_top_k(self, db):
        _seed(db, [
            (f"N{i}", f"term-{i} word", [float(i == 0), 0.0, 0.0])
            for i in range(5)
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid(
            query_text="word",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
        )
        assert len(hits) == 2

    def test_excludes_note_id(self, db):
        _seed(db, [
            ("A", "needle keyword",   [1.0, 0.0, 0.0]),
            ("B", "needle elsewhere", [0.9, 0.1, 0.0]),
        ])
        # Note A has id 1 (first insert); exclude it.
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid(
            query_text="needle",
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            exclude_note_id=1,
        )
        assert all(h["note_id"] != 1 for h in hits)

    def test_empty_corpus_returns_empty(self, db):
        conn = Connector(db, _StubEmbedder())
        assert conn.find_hybrid("anything", [1.0, 0.0, 0.0], top_k=5) == []

    def test_returned_dicts_do_not_leak_embedding_id(self, db):
        # embedding_id is retained internally for rank-input logging but must
        # be stripped before the result reaches callers.
        _seed(db, [
            ("A", "needle keyword one", [1.0, 0.0, 0.0]),
            ("B", "needle keyword two", [0.9, 0.1, 0.0]),
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_hybrid("needle", [1.0, 0.0, 0.0], top_k=2)
        assert hits and all("embedding_id" not in h for h in hits)
        assert all(set(h) == {"note_id", "text", "score"} for h in hits)


class TestRRFRankInputLogging:
    def test_reports_dense_and_bm25_ranks_per_survivor(self, monkeypatch):
        from grimore.cognition import connector as conn_mod

        calls: list[tuple[str, dict]] = []

        class _CaptureLogger:
            def debug(self, event, **kw):
                calls.append((event, kw))

            def __getattr__(self, _name):  # info/warning/… are no-ops
                return lambda *a, **k: None

        monkeypatch.setattr(conn_mod, "logger", _CaptureLogger())

        dense = [{"embedding_id": 10}, {"embedding_id": 20}, {"embedding_id": 30}]
        sparse = [{"embedding_id": 30}, {"embedding_id": 40}]
        survivors = [
            {"embedding_id": 30, "note_id": 3},   # both lists: dense#3, bm25#1
            {"embedding_id": 10, "note_id": 1},   # dense-only: dense#1, bm25 None
            {"embedding_id": 40, "note_id": 4},   # bm25-only: dense None, bm25#2
        ]
        Connector._log_rrf_inputs(dense, sparse, survivors)

        event, kw = calls[-1]
        assert event == "rrf_rank_inputs"
        assert kw["dense_pool"] == 3 and kw["bm25_pool"] == 2
        assert kw["inputs"] == [
            {"note_id": 3, "dense_rank": 3, "bm25_rank": 1},
            {"note_id": 1, "dense_rank": 1, "bm25_rank": None},
            {"note_id": 4, "dense_rank": None, "bm25_rank": 2},
        ]

    def test_no_survivors_logs_nothing(self, monkeypatch):
        from grimore.cognition import connector as conn_mod

        calls: list = []

        class _Cap:
            def debug(self, *a, **k):
                calls.append((a, k))

            def __getattr__(self, _name):
                return lambda *a, **k: None

        monkeypatch.setattr(conn_mod, "logger", _Cap())
        Connector._log_rrf_inputs([{"embedding_id": 1}], [], [])
        assert calls == []


class TestFindSimilarStillWorks:
    """The dense-only path is still the fallback when FTS is off — keep it wired."""

    def test_returns_sorted_by_cosine(self, db):
        _seed(db, [
            ("A", "a", [1.0, 0.0, 0.0]),
            ("B", "b", [0.0, 1.0, 0.0]),
            ("C", "c", [0.7, 0.7, 0.0]),
        ])
        conn = Connector(db, _StubEmbedder())
        hits = conn.find_similar_notes([1.0, 0.0, 0.0], top_k=3)
        assert hits[0]["text"] == "a"
        assert hits[-1]["text"] == "b"
