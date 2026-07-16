"""
Pluggable vector backend.

Two surfaces are covered:

* **Routing.** The Connector inspects ``db.vec_available`` + the
  ``vector_backend`` mode and either delegates to ``db.vec_search`` or
  builds the in-memory matmul matrix. These tests stub the DB so they
  don't depend on sqlite-vec being installed.
* **End-to-end against the real extension.** Marked with
  ``pytest.mark.vec``; auto-skipped when the extension probe fails so
  Termux / minimal CI environments stay green.
"""
from __future__ import annotations

import sqlite3
import struct

import pytest

from grimore.cognition.connector import Connector
from grimore.cognition.embedder import Embedder
from grimore.memory.db import Database


# ── Helpers ──────────────────────────────────────────────────────────────


class _StubEmbedder:
    """Minimal stand-in so Connector can be constructed."""
    model = "stub"


def _vec_blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _unit(values: list[float]) -> list[float]:
    return Embedder.normalize(values)


@pytest.fixture
def vec_db(tmp_path) -> Database:
    """Real Database; sqlite-vec may or may not be loaded under it.

    The downstream routing tests don't care — they monkey-patch
    ``vec_available`` to True so the code path is exercised either way.
    """
    return Database(str(tmp_path / "vec.db"))


# ── Routing tests (no extension required) ───────────────────────────────


class TestRouting:
    def test_use_vec_backend_off_when_numpy_pinned(self, vec_db, monkeypatch):
        monkeypatch.setattr(type(vec_db), "vec_available",
                            property(lambda self: True))
        conn = Connector(vec_db, _StubEmbedder(), vector_backend="numpy")
        assert conn._use_vec_backend() is False

    def test_use_vec_backend_on_when_auto_and_available(self, vec_db, monkeypatch):
        monkeypatch.setattr(type(vec_db), "vec_available",
                            property(lambda self: True))
        conn = Connector(vec_db, _StubEmbedder(), vector_backend="auto")
        assert conn._use_vec_backend() is True

    def test_use_vec_backend_off_when_not_available(self, vec_db, monkeypatch):
        monkeypatch.setattr(type(vec_db), "vec_available",
                            property(lambda self: False))
        conn = Connector(vec_db, _StubEmbedder(), vector_backend="sqlite-vec")
        assert conn._use_vec_backend() is False

    def test_find_similar_notes_delegates_to_vec_search(self, vec_db, monkeypatch):
        """When the vec backend is on, the Connector must not build the
        in-memory matrix; it must hand the query to ``db.vec_search``."""
        monkeypatch.setattr(type(vec_db), "vec_available",
                            property(lambda self: True))
        calls = []

        def fake_vec_search(query, limit, exclude_note_id=None):
            calls.append((list(query), limit, exclude_note_id))
            return [(7, 42, "hit-text", 0.91)]

        monkeypatch.setattr(vec_db, "vec_search", fake_vec_search)
        # Ensure the numpy path would explode if called — proves we routed away.
        def _boom(*_a, **_kw):
            raise AssertionError("matmul path should not run when vec backend is on")
        monkeypatch.setattr(vec_db, "get_all_embeddings_with_id", _boom)

        conn = Connector(vec_db, _StubEmbedder(), vector_backend="auto")
        out = conn.find_similar_notes([0.1, 0.2, 0.3], top_k=3)
        assert out == [{"note_id": 42, "text": "hit-text", "score": 0.91}]
        assert len(calls) == 1
        _, limit, exclude = calls[0]
        assert limit == 3
        assert exclude is None

    def test_vector_candidates_delegates_to_vec_search(self, vec_db, monkeypatch):
        monkeypatch.setattr(type(vec_db), "vec_available",
                            property(lambda self: True))
        monkeypatch.setattr(vec_db, "vec_search",
                            lambda q, k, exclude_note_id=None: [(1, 10, "a", 0.8),
                                                                (2, 11, "b", 0.7)])
        conn = Connector(vec_db, _StubEmbedder(), vector_backend="auto")
        hits = conn._vector_candidates([0.1, 0.2], limit=2)
        assert [h["note_id"] for h in hits] == [10, 11]
        assert hits[0]["embedding_id"] == 1


# ── DB-level probing & schema (no extension required) ──────────────────


class TestProbeAndSchema:
    def test_probe_returns_false_without_extension(self, monkeypatch):
        # Force the import to fail even if sqlite_vec is somehow on path.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sqlite_vec":
                raise ImportError("sqlite_vec not installed in this run")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        assert Database._probe_vec_extension() is False

    def test_vec_available_false_when_dim_unknown(self, vec_db, monkeypatch):
        # Even if the extension probe succeeds, vec_available stays False
        # until a vec table at a known dim exists.
        monkeypatch.setattr(vec_db, "_vec_available", True)
        monkeypatch.setattr(vec_db, "_vec_dim", None)
        assert vec_db.vec_available is False

    def test_vec_search_noop_when_unavailable(self, vec_db):
        # Default install path has no extension → vec_search returns [].
        out = vec_db.vec_search([0.1, 0.2], 5)
        assert out == []


# ── Real extension end-to-end (skipped without sqlite-vec) ─────────────


_VEC_AVAILABLE = Database._probe_vec_extension()
needs_vec = pytest.mark.skipif(
    not _VEC_AVAILABLE,
    reason="sqlite-vec extension not loadable on this Python build",
)


@pytest.mark.vec
class TestVecEndToEnd:
    """Marked tests that require the real extension. Skipped on environments
    where ``sqlite-vec`` isn't installable (Termux ARM64 PyPI gap, etc.)."""

    @needs_vec
    def test_insert_then_search_returns_nearest(self, tmp_path):
        db = Database(str(tmp_path / "vec_e2e.db"))
        note_id = db.upsert_note(path="/a.md", title="A", content_hash="x")
        v1 = _unit([1.0, 0.0, 0.0, 0.0])
        v2 = _unit([0.0, 1.0, 0.0, 0.0])
        v3 = _unit([0.0, 0.0, 1.0, 0.0])
        db.store_embedding(note_id, 0, "near-v1", _vec_blob(v1), chunk_hash="h1")
        db.store_embedding(note_id, 1, "near-v2", _vec_blob(v2), chunk_hash="h2")
        db.store_embedding(note_id, 2, "near-v3", _vec_blob(v3), chunk_hash="h3")

        results = db.vec_search(v1, 2)
        # Top hit must be the chunk whose vector matches the query exactly.
        assert results[0][2] == "near-v1"
        # Score is 1 - cosine_distance, so a perfect hit ≈ 1.0
        assert results[0][3] == pytest.approx(1.0, abs=1e-3)
        assert len(results) == 2

    @needs_vec
    def test_parity_with_numpy_backend(self, tmp_path):
        """Same vectors, same query → numpy path and vec path return the
        same top-k by note_id (cosine vs cosine, ties allowed)."""
        db = Database(str(tmp_path / "parity.db"))
        nid = db.upsert_note(path="/p.md", title="P", content_hash="x")
        rng = __import__("random").Random(0x5EED)
        vectors = []
        for i in range(20):
            raw = [rng.gauss(0, 1) for _ in range(8)]
            v = _unit(raw)
            vectors.append(v)
            db.store_embedding(nid, i, f"c{i}", _vec_blob(v), chunk_hash=f"h{i}")

        query = _unit([0.4, 0.2, -0.1, 0.05, 0.0, 0.7, -0.3, 0.1])

        vec_top = [r[0] for r in db.vec_search(query, 5)]

        numpy_conn = Connector(db, _StubEmbedder(), vector_backend="numpy")
        numpy_hits = numpy_conn._vector_candidates(query, limit=5)
        numpy_top = [h["embedding_id"] for h in numpy_hits]

        assert vec_top == numpy_top

    @needs_vec
    def test_deletes_mirror_into_vec_table(self, tmp_path):
        db = Database(str(tmp_path / "del.db"))
        nid = db.upsert_note(path="/d.md", title="D", content_hash="x")
        v = _unit([0.5, 0.5, 0.5, 0.5])
        db.store_embedding(nid, 0, "c0", _vec_blob(v), chunk_hash="h0")
        db.store_embedding(nid, 1, "c1", _vec_blob(v), chunk_hash="h1")

        assert len(db.vec_search(v, 5)) == 2
        db.delete_chunks(nid, [0])
        assert len(db.vec_search(v, 5)) == 1
        db.delete_note_embeddings(nid)
        assert db.vec_search(v, 5) == []

    @needs_vec
    def test_prune_clears_the_vec_mirror(self, tmp_path):
        """Pruning a note whose file disappeared must drop its vec rows too,
        not just the ``embeddings`` rows — otherwise orphaned vectors keep
        occupying KNN slots in vec_search and silently shrink results."""
        db = Database(str(tmp_path / "prune.db"))
        gone = db.upsert_note(path="/gone.md", title="Gone", content_hash="x")
        kept = db.upsert_note(path="/kept.md", title="Kept", content_hash="y")
        v = _unit([0.5, 0.5, 0.5, 0.5])
        db.store_embedding(gone, 0, "g0", _vec_blob(v), chunk_hash="g0")
        db.store_embedding(kept, 0, "k0", _vec_blob(v), chunk_hash="k0")
        assert len(db.vec_search(v, 5)) == 2

        # /gone.md is no longer on disk; only /kept.md survives the scan.
        assert db.prune_missing_notes(["/kept.md"]) == 1

        hits = db.vec_search(v, 5)
        assert [nid for _eid, nid, _txt, _score in hits] == [kept]

    @needs_vec
    def test_backfill_on_upgrade_path(self, tmp_path):
        """A pre-existing vault (built before sqlite-vec was installed)
        gets the vec table built and populated on the next Database
        instantiation."""
        # Bootstrap a DB with rows but no vec table.
        db = Database(str(tmp_path / "upgrade.db"))
        if db.vec_available:
            db.drop_vec_table()
        nid = db.upsert_note(path="/u.md", title="U", content_hash="x")
        v = _unit([1.0, 0.0, 0.0, 0.0])
        # Insert through a connection that bypasses the vec mirror.
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                "INSERT INTO embeddings (note_id, chunk_index, text_content, vector, chunk_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (nid, 0, "u0", _vec_blob(v), "h0"),
            )
            conn.commit()

        # Re-open → migration fires backfill.
        db2 = Database(str(tmp_path / "upgrade.db"))
        if not db2.vec_available:
            pytest.skip("vec backend disabled after re-open")
        results = db2.vec_search(v, 5)
        assert len(results) == 1
        assert results[0][2] == "u0"
