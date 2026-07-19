"""
Chunk-level incremental re-embedding.

The old code did ``DELETE FROM embeddings WHERE note_id = ?`` before
re-indexing on every content change, which forced a full re-embed even
when one paragraph of a 200-page doc was touched. The new path keys
each chunk by ``Embedder.chunk_hash(text, model)`` and only re-embeds
indices whose content actually changed.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock


from grimore.cognition.chunker import Chunk
from grimore.cognition.embedder import Embedder
from grimore.cognition.reembed import reembed_note
from grimore.memory.db import Database


# ── Fakes ────────────────────────────────────────────────────────────────


class _SpyEmbedder:
    """Deterministic mock embedder that tracks how often ``embed`` was called.

    Generates a fake 4-dim float32 vector keyed on the text so identical
    text round-trips to identical bytes (essential for the parity test).
    """

    def __init__(self, model: str = "fake-model"):
        self.model = model
        self.calls: list[str] = []

    def embed(self, text: str) -> Optional[list[float]]:
        self.calls.append(text)
        # Stable per-text vector: hash bytes into 4 floats in [0, 1].
        digest = abs(hash(text))
        return [
            ((digest >> (8 * i)) & 0xFF) / 255.0
            for i in range(4)
        ]

    @staticmethod
    def serialize_vector(vec: list[float]) -> bytes:
        return Embedder.serialize_vector(vec)


class _BatchSpyEmbedder(_SpyEmbedder):
    """Spy that also implements embed_batch, recording each batch call so a
    test can assert reembed_note takes the one-round-trip path."""

    def __init__(self, model: str = "fake-model"):
        super().__init__(model)
        self.batch_calls: list[list[str]] = []

    def embed_batch(self, texts):
        self.batch_calls.append(list(texts))
        return [self.embed(t) for t in texts]


def _make_db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "incremental.db"))


def _seed_note(db: Database, path: str = "/vault/long.md") -> int:
    return db.upsert_note(path=path, title="Long", content_hash="x")


# ── Embedder.chunk_hash ──────────────────────────────────────────────────


class TestChunkHash:
    def test_stable_for_same_text_and_model(self):
        a = Embedder.chunk_hash("hello world", "model-a")
        b = Embedder.chunk_hash("hello world", "model-a")
        assert a == b

    def test_changes_with_text(self):
        a = Embedder.chunk_hash("hello world", "model-a")
        b = Embedder.chunk_hash("HELLO world", "model-a")
        assert a != b

    def test_changes_with_model(self):
        # Model swap must implicitly invalidate every chunk — this is what
        # makes migrate-embeddings work without a separate "wipe hashes" pass.
        a = Embedder.chunk_hash("same text", "model-a")
        b = Embedder.chunk_hash("same text", "model-b")
        assert a != b

    def test_hex_and_truncated(self):
        h = Embedder.chunk_hash("anything", "model-a")
        assert len(h) == 32
        int(h, 16)  # raises if not hex


# ── DB helpers ───────────────────────────────────────────────────────────


class TestDbHelpers:
    def test_get_chunk_hashes_empty_note(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        assert db.get_chunk_hashes(note_id) == {}

    def test_get_chunk_hashes_returns_index_to_hash(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        db.store_embedding(note_id, 0, "a", b"\x00" * 4, chunk_hash="h0")
        db.store_embedding(note_id, 1, "b", b"\x00" * 4, chunk_hash="h1")
        assert db.get_chunk_hashes(note_id) == {0: "h0", 1: "h1"}

    def test_get_chunk_hashes_returns_none_for_legacy_rows(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        # Legacy call shape (pre-chunk_hash) omits the new kwarg.
        db.store_embedding(note_id, 0, "a", b"\x00" * 4)
        assert db.get_chunk_hashes(note_id) == {0: None}

    def test_delete_chunks_removes_only_given_indices(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        for i in range(4):
            db.store_embedding(note_id, i, f"c{i}", b"\x00" * 4, chunk_hash=f"h{i}")
        removed = db.delete_chunks(note_id, [1, 3])
        assert removed == 2
        remaining = db.get_chunk_hashes(note_id)
        assert set(remaining.keys()) == {0, 2}

    def test_delete_chunks_empty_list_is_noop(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        db.store_embedding(note_id, 0, "a", b"\x00" * 4, chunk_hash="h0")
        assert db.delete_chunks(note_id, []) == 0
        assert db.get_chunk_hashes(note_id) == {0: "h0"}

    def test_get_chunk_records_returns_hash_and_anchors(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        db.store_embedding(
            note_id, 0, "a", b"\x00" * 4, page=1, heading="Intro", chunk_hash="h0"
        )
        db.store_embedding(note_id, 1, "b", b"\x00" * 4, chunk_hash="h1")
        assert db.get_chunk_records(note_id) == {
            0: ("h0", 1, "Intro"),
            1: ("h1", None, None),
        }
        # get_chunk_hashes stays a thin view over the same rows.
        assert db.get_chunk_hashes(note_id) == {0: "h0", 1: "h1"}


# ── reembed_note ─────────────────────────────────────────────────────────


class TestReembedNote:
    def _chunks(self, *texts: str) -> list[Chunk]:
        return [Chunk(text=t) for t in texts]

    def test_first_scan_embeds_every_chunk(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3"))
        assert result.embedded == 3
        assert result.kept == 0
        assert result.removed == 0
        assert result.stored == 3
        assert emb.calls == ["p1", "p2", "p3"]

    def test_no_changes_skips_embed_calls(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        chunks = self._chunks("p1", "p2", "p3")

        # First pass populates hashes.
        reembed_note(db, emb, note_id, chunks)
        emb.calls.clear()

        # Second pass on identical chunks must not call embed.
        result = reembed_note(db, emb, note_id, chunks)
        assert emb.calls == []
        assert result.embedded == 0
        assert result.kept == 3
        assert result.removed == 0

    def test_edit_one_chunk_reembeds_only_that_one(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3"))
        emb.calls.clear()

        # Mid-doc edit.
        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2-edited", "p3"))
        assert emb.calls == ["p2-edited"]
        assert result.embedded == 1
        assert result.kept == 2
        # The stale row at index 1 is dropped before the new one lands.
        assert result.removed == 1
        assert result.stored == 3

    def test_shrink_doc_drops_surplus_chunks(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3", "p4"))
        emb.calls.clear()

        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2"))
        assert emb.calls == []  # nothing changed, just truncated
        assert result.embedded == 0
        assert result.kept == 2
        assert result.removed == 2

    def test_grow_doc_only_embeds_new_chunks(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, self._chunks("p1", "p2"))
        emb.calls.clear()

        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3", "p4"))
        assert emb.calls == ["p3", "p4"]
        assert result.embedded == 2
        assert result.kept == 2

    def test_model_swap_reembeds_everything(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb_a = _SpyEmbedder(model="model-a")
        reembed_note(db, emb_a, note_id, self._chunks("p1", "p2", "p3"))

        emb_b = _SpyEmbedder(model="model-b")
        result = reembed_note(db, emb_b, note_id, self._chunks("p1", "p2", "p3"))
        # Hash key includes the model, so every chunk is stale under a new model.
        assert emb_b.calls == ["p1", "p2", "p3"]
        assert result.embedded == 3
        assert result.removed == 3

    def test_legacy_null_hash_rows_back_fill(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        # Pre-existing rows from before the migration: no chunk_hash.
        db.store_embedding(note_id, 0, "p1", b"\x00" * 16)
        db.store_embedding(note_id, 1, "p2", b"\x00" * 16)
        emb = _SpyEmbedder()

        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2"))
        # Both rows must re-embed once because NULL hash counts as stale.
        assert emb.calls == ["p1", "p2"]
        assert result.embedded == 2
        # Second pass after back-fill is now a no-op.
        emb.calls.clear()
        result2 = reembed_note(db, emb, note_id, self._chunks("p1", "p2"))
        assert emb.calls == []
        assert result2.kept == 2

    def test_failed_embed_skips_chunk_without_corrupting_others(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)

        class _FlakyEmbedder(_SpyEmbedder):
            def embed(self, text: str):
                self.calls.append(text)
                if text == "broken":
                    return None
                return [0.1, 0.2, 0.3, 0.4]

        emb = _FlakyEmbedder()
        result = reembed_note(db, emb, note_id, self._chunks("ok1", "broken", "ok2"))
        assert result.embedded == 2  # one was skipped
        stored = db.get_chunk_hashes(note_id)
        # Index 1 stays absent (no row), but the others are present.
        assert 0 in stored and 2 in stored
        assert 1 not in stored

    def test_section_anchors_round_trip(self, tmp_path):
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        chunks = [
            Chunk(text="abc", page=1, heading="Intro"),
            Chunk(text="def", page=2, heading="Body"),
        ]
        reembed_note(db, emb, note_id, chunks)

        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            rows = conn.execute(
                "SELECT chunk_index, page, heading FROM embeddings "
                "WHERE note_id = ? ORDER BY chunk_index",
                (note_id,),
            ).fetchall()
        assert rows == [(0, 1, "Intro"), (1, 2, "Body")]

    def test_unchanged_rows_keep_their_primary_key(self, tmp_path):
        """The whole point: untouched chunks keep their ``id``, which means
        any downstream join (FTS, vec backend) that keyed on rowid stays
        valid across re-scans."""
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3"))
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            ids_before = dict(conn.execute(
                "SELECT chunk_index, id FROM embeddings WHERE note_id = ?",
                (note_id,),
            ).fetchall())

        reembed_note(db, emb, note_id, self._chunks("p1", "p2-edited", "p3"))
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            ids_after = dict(conn.execute(
                "SELECT chunk_index, id FROM embeddings WHERE note_id = ?",
                (note_id,),
            ).fetchall())
        assert ids_before[0] == ids_after[0]
        assert ids_before[2] == ids_after[2]
        # The edited chunk got a new id (it was deleted then re-inserted).
        assert ids_before[1] != ids_after[1]

    def test_moved_anchor_is_reanchored_not_reembedded(self, tmp_path):
        """A chunk whose text is unchanged but whose page/heading moved (a
        pagination reflow earlier in the doc) refreshes its stored anchors
        without paying for a re-embed — otherwise a citation like
        ``[[Title#p.42]]`` points at the wrong page."""
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, [Chunk(text="abc", page=1, heading="Intro")])
        emb.calls.clear()

        result = reembed_note(
            db, emb, note_id, [Chunk(text="abc", page=7, heading="Later")]
        )
        assert emb.calls == []          # text unchanged → embedding reused
        assert result.kept == 1
        assert result.embedded == 0
        with closing(sqlite3.connect(db.db_path)) as conn, conn:
            row = conn.execute(
                "SELECT page, heading FROM embeddings "
                "WHERE note_id = ? AND chunk_index = 0",
                (note_id,),
            ).fetchone()
        assert row == (7, "Later")      # anchor refreshed in place

    def test_unchanged_anchor_skips_the_update(self, tmp_path):
        """No anchor change → no spurious UPDATE on the kept row."""
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _SpyEmbedder()
        reembed_note(db, emb, note_id, [Chunk(text="abc", page=3, heading="H")])

        spy = MagicMock(wraps=db.update_chunk_anchors)
        db.update_chunk_anchors = spy
        reembed_note(db, emb, note_id, [Chunk(text="abc", page=3, heading="H")])
        spy.assert_not_called()

    def test_changed_chunks_embed_in_one_batch(self, tmp_path):
        """All stale chunks are embedded in a single batch call (not N serial
        round-trips) when the embedder exposes embed_batch."""
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _BatchSpyEmbedder()
        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3"))
        assert result.embedded == 3
        assert emb.batch_calls == [["p1", "p2", "p3"]]   # one batch, all chunks

    def test_batch_only_covers_stale_chunks(self, tmp_path):
        """Kept chunks aren't re-embedded — the batch holds only what changed."""
        db = _make_db(tmp_path)
        note_id = _seed_note(db)
        emb = _BatchSpyEmbedder()
        reembed_note(db, emb, note_id, self._chunks("p1", "p2", "p3"))
        emb.batch_calls.clear()
        result = reembed_note(db, emb, note_id, self._chunks("p1", "p2-edited", "p3"))
        assert result.embedded == 1
        assert emb.batch_calls == [["p2-edited"]]
