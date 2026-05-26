"""Tests for the v2.1 retrieval-quality upgrades:

* NumPy-vectorized dense scoring (with parity against the pure-Python loop
  and graceful fallback on ragged vectors / missing numpy),
* the embeddings change-detection signature,
* conversation-aware Oracle (query rewrite + history block),
* citation grounding,
* optional LLM re-rank, and
* the single-source version string.
"""
import struct
from unittest.mock import MagicMock

import pytest

from grimore.cognition.connector import Connector
from grimore.cognition.embedder import Embedder
from grimore.cognition.oracle import Oracle
from grimore.memory.db import Database
from grimore.session import Session


def _vec(values):
    return struct.pack(f"{len(values)}f", *values)


def _seed(db: Database, rows):
    """Insert ``(title, text, vector)`` rows directly (one chunk per note)."""
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
                "INSERT INTO embeddings (note_id, chunk_index, text_content, vector)"
                " VALUES (?, ?, ?, ?)",
                (note_id, 0, text, _vec(vec)),
            )
        conn.commit()


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "grimore.db"))


# ── NumPy vector search ──────────────────────────────────────────────────


class TestVectorSearch:
    def test_numpy_matches_python_loop(self, db, monkeypatch):
        """The matmul fast path must rank identically to the per-row loop and
        agree on scores to float32 tolerance."""
        rows = [
            ("A", "alpha", Embedder.normalize([1.0, 0.2, 0.1])),
            ("B", "beta", Embedder.normalize([0.1, 1.0, 0.2])),
            ("C", "gamma", Embedder.normalize([0.2, 0.1, 1.0])),
            ("D", "delta", Embedder.normalize([0.9, 0.9, 0.1])),
        ]
        _seed(db, rows)
        query = Embedder.normalize([0.8, 0.3, 0.1])

        fast = Connector(db, Embedder.__new__(Embedder)).find_similar_notes(query, top_k=4)

        # Force the fallback: numpy invisible to both modules → per-row loop.
        monkeypatch.setattr("grimore.cognition.connector._np", None)
        monkeypatch.setattr("grimore.cognition.embedder._np", None)
        slow = Connector(db, Embedder.__new__(Embedder)).find_similar_notes(query, top_k=4)

        assert [r["note_id"] for r in fast] == [r["note_id"] for r in slow]
        for f, s in zip(fast, slow):
            assert f["score"] == pytest.approx(s["score"], abs=1e-4)

    def test_ragged_vectors_fall_back_without_crashing(self, db):
        """Mixed dims (model swapped without a re-scan) must not raise — the
        matrix builder returns None and the loop handles each vector."""
        _seed(db, [
            ("A", "three dims", [1.0, 0.0, 0.0]),
            ("B", "two dims", [0.0, 1.0]),
        ])
        assert Embedder.vectors_to_matrix([_vec([1.0, 0.0, 0.0]), _vec([0.0, 1.0])]) is None
        out = Connector(db, Embedder.__new__(Embedder)).find_similar_notes(
            [1.0, 0.0, 0.0], top_k=2
        )
        assert {r["note_id"] for r in out} == {1, 2}

    def test_empty_db_returns_empty(self, db):
        assert Connector(db, Embedder.__new__(Embedder)).find_similar_notes(
            [1.0, 0.0], top_k=5
        ) == []


class TestEmbeddingsSignature:
    def test_signature_tracks_inserts(self, db):
        assert db.embeddings_signature() == (0, 0)
        _seed(db, [("A", "a", [1.0, 0.0])])
        count, max_id = db.embeddings_signature()
        assert count == 1 and max_id >= 1


# ── citation grounding ─────────────────────────────────────────────────────


class TestCitationGrounding:
    def test_hallucinated_citation_is_unlinked(self):
        sources = ["Quantum Notes#p.4", "Stoicism"]
        text = "Per [[Quantum Notes#p.4]] and [[Quantum Notes]], not [[Bogus]]. See [[Stoicism]]."
        cleaned, dropped = Oracle.verify_citations(text, sources)
        assert dropped == 1
        assert "[[Bogus]]" not in cleaned and "Bogus" in cleaned
        assert "[[Quantum Notes#p.4]]" in cleaned  # exact match kept
        assert "[[Quantum Notes]]" in cleaned       # anchor-tolerant match kept
        assert "[[Stoicism]]" in cleaned

    def test_no_sources_drops_everything(self):
        cleaned, dropped = Oracle.verify_citations("see [[X]] and [[Y]]", [])
        assert dropped == 2 and "[[" not in cleaned

    def test_empty_text_is_safe(self):
        assert Oracle.verify_citations("", ["A"]) == ("", 0)


# ── conversation memory ────────────────────────────────────────────────────


class TestConversationMemory:
    def _oracle(self):
        o = Oracle.__new__(Oracle)
        o.router = MagicMock()
        return o

    def test_rewrite_uses_router_with_history(self):
        o = self._oracle()
        o.router.complete.return_value = {"query": "stoic view on death"}
        assert o._rewrite_query("what about death?", [{"q": "stoicism", "a": "..."}]) == \
            "stoic view on death"

    def test_rewrite_identity_without_history(self):
        o = self._oracle()
        assert o._rewrite_query("plain question", None) == "plain question"
        o.router.complete.assert_not_called()

    def test_rewrite_falls_back_on_failure(self):
        o = self._oracle()
        o.router.complete.return_value = None  # circuit open / Ollama down
        assert o._rewrite_query("orig", [{"q": "a", "a": "b"}]) == "orig"

    def test_history_block_is_bounded(self):
        o = self._oracle()
        block = o._format_history([{"q": "Q" * 5000, "a": "A" * 5000}])
        assert "User:" in block and len(block) <= o._HISTORY_MAX_CHARS

    def test_session_turns_capped_and_cleared(self, tmp_path):
        s = Session.__new__(Session)
        s.turns = []
        s.last_question = s.last_answer = None
        s.question_log = []
        for i in range(6):
            s.record_turn(f"q{i}", f"a{i}", [])
        assert len(s.turns) == Session.MAX_TURNS
        assert s.turns[-1]["q"] == "q5"
        s.forget()
        assert s.turns == [] and s.question_log == [] and s.last_answer is None


# ── LLM re-rank ─────────────────────────────────────────────────────────────


class TestRerank:
    def _connector(self, return_value):
        c = Connector.__new__(Connector)
        c.router = MagicMock()
        c.router.complete.return_value = return_value
        return c

    def test_reorders_by_llm_score(self):
        cands = [{"note_id": i, "text": f"p{i}", "score": 1.0 / (i + 1)} for i in range(5)]
        c = self._connector({"scores": [
            {"index": 3, "score": 9}, {"index": 0, "score": 7}, {"index": 1, "score": 2},
        ]})
        out = c._llm_rerank("q", cands, pool=5)
        assert [d["note_id"] for d in out][:2] == [3, 0]

    def test_failure_preserves_order(self):
        cands = [{"note_id": i, "text": f"p{i}", "score": 1.0} for i in range(4)]
        for bad in (None, {"nope": 1}, {"scores": "garbage"}):
            c = self._connector(bad)
            assert [d["note_id"] for d in c._llm_rerank("q", cands, 4)] == [0, 1, 2, 3]


# ── version ─────────────────────────────────────────────────────────────────


def test_version_is_exposed():
    import grimore
    assert isinstance(grimore.__version__, str) and grimore.__version__


def test_cli_version_flag():
    from typer.testing import CliRunner
    from grimore.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "grimore" in result.stdout.lower()
