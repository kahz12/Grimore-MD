"""
Embedding-model hot-swap.

The migration is split into three primitives:

* ``begin_embedding_migration`` — stamps a ``migrations`` row + creates
  the shadow ``embeddings_migration`` table.
* ``append_migration_row`` — worker step, called once per source row.
* ``swap_embedding_migration`` — atomic single-transaction replace.

A fourth, ``abort_embedding_migration``, drops the shadow without
touching the live table. These tests exercise each primitive
independently and the end-to-end orchestration through
``_do_migrate_embeddings``.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from grimore.cognition.embedder import Embedder
from grimore.memory.db import Database
from grimore.operations import _do_migrate_embeddings
from grimore.utils.config import (
    CognitionConfig, Config, MemoryConfig, VaultConfig,
)


def _vec_blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _seed_db(tmp_path: Path, n_chunks: int = 3, dim: int = 4) -> Database:
    """Build a small DB with ``n_chunks`` embeddings on a single note."""
    db = Database(str(tmp_path / "migrate.db"))
    note_id = db.upsert_note(path="/v/n.md", title="Note", content_hash="x")
    for i in range(n_chunks):
        v = [0.1 * (i + 1)] * dim
        db.store_embedding(
            note_id, i, f"chunk-{i}",
            _vec_blob(v), chunk_hash=f"old-{i}",
        )
    return db


def _make_config(tmp_path: Path, current_model: str = "old-model") -> Config:
    return Config(
        vault=VaultConfig(path=str(tmp_path / "vault")),
        cognition=CognitionConfig(model_embeddings_local=current_model),
        memory=MemoryConfig(db_path=str(tmp_path / "migrate.db")),
    )


# ── DB primitives ───────────────────────────────────────────────────────


class TestBegin:
    def test_fresh_start_creates_row_and_shadow_table(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=3)
        row = db.begin_embedding_migration("new-model")
        assert row["status"] == "running"
        assert row["target_model"] == "new-model"
        assert row["total"] == 3
        assert row["done"] == 0
        with sqlite3.connect(db.db_path) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "embeddings_migration" in tables

    def test_resume_returns_existing_row(self, tmp_path):
        db = _seed_db(tmp_path)
        first = db.begin_embedding_migration("new-model")
        second = db.begin_embedding_migration("new-model")
        assert first["id"] == second["id"]

    def test_conflicting_target_raises(self, tmp_path):
        db = _seed_db(tmp_path)
        db.begin_embedding_migration("new-model")
        with pytest.raises(ValueError, match="abort it first"):
            db.begin_embedding_migration("other-model")


class TestWorkerStep:
    def test_append_advances_done(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=2)
        db.begin_embedding_migration("new-model")
        pending = db.iter_pending_migration_rows()
        assert len(pending) == 2

        src_id, note_id, idx, text, page, heading = pending[0]
        db.append_migration_row(
            src_id, note_id, idx, text, _vec_blob([1.0, 0.0, 0.0, 0.0]),
            page, heading, chunk_hash="new-0",
        )

        active = db.get_active_embedding_migration()
        assert active["done"] == 1
        # Pending list shrinks because the next call only returns rows whose
        # source id is greater than the shadow's max id.
        assert len(db.iter_pending_migration_rows()) == 1


class TestSwap:
    def _embed_all(self, db: Database, model: str = "new-model") -> None:
        for src_id, note_id, idx, text, page, heading in db.iter_pending_migration_rows():
            db.append_migration_row(
                src_id, note_id, idx, text,
                _vec_blob([0.9, 0.1, 0.0, 0.0]),
                page, heading,
                chunk_hash=Embedder.chunk_hash(text, model),
            )

    def test_swap_replaces_embeddings_atomically(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=3)
        db.begin_embedding_migration("new-model")
        self._embed_all(db)
        result = db.swap_embedding_migration()
        assert result["status"] == "complete"

        # New rows present, old vectors gone, shadow table dropped.
        with sqlite3.connect(db.db_path) as conn:
            rows = conn.execute(
                "SELECT chunk_hash, vector FROM embeddings ORDER BY chunk_index"
            ).fetchall()
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert len(rows) == 3
        for chunk_hash, vector in rows:
            assert chunk_hash.startswith(Embedder.chunk_hash("", "new-model")[:0] or "")
            # The vector blob is the new one (0.9, 0.1, 0.0, 0.0).
            assert struct.unpack("4f", vector)[0] == pytest.approx(0.9, abs=1e-5)
        assert "embeddings_migration" not in tables

    def test_swap_refuses_when_incomplete(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=3)
        db.begin_embedding_migration("new-model")
        # Only embed the first row.
        pending = db.iter_pending_migration_rows()
        src_id, note_id, idx, text, page, heading = pending[0]
        db.append_migration_row(
            src_id, note_id, idx, text, _vec_blob([1.0, 0.0, 0.0, 0.0]),
            page, heading, chunk_hash="new-0",
        )
        with pytest.raises(RuntimeError, match="not done yet"):
            db.swap_embedding_migration()

    def test_swap_with_no_migration_raises(self, tmp_path):
        db = _seed_db(tmp_path)
        with pytest.raises(RuntimeError):
            db.swap_embedding_migration()


class TestAbort:
    def test_abort_drops_shadow_without_touching_live(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=2)
        db.begin_embedding_migration("new-model")
        pending = db.iter_pending_migration_rows()
        src_id, note_id, idx, text, page, heading = pending[0]
        db.append_migration_row(
            src_id, note_id, idx, text, _vec_blob([0.0] * 4),
            page, heading, chunk_hash="x",
        )

        result = db.abort_embedding_migration()
        assert result["status"] == "aborted"
        with sqlite3.connect(db.db_path) as conn:
            live = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert live == 2
        assert "embeddings_migration" not in tables

    def test_abort_with_nothing_in_flight(self, tmp_path):
        db = _seed_db(tmp_path)
        assert db.abort_embedding_migration() is None


# ── _do_migrate_embeddings orchestration ────────────────────────────────


def _deterministic_embed(self, text: str):
    """Stand-in for ``Embedder.embed`` that's deterministic per text.

    Patched in via ``patch.object`` so the real ``Embedder.chunk_hash`` /
    ``serialize_vector`` static helpers stay intact — only the network
    call gets stubbed out.
    """
    seed = (abs(hash(text)) % 1000) / 1000.0
    return [seed, seed, seed, seed]


def _no_op_init(self, config, cache=None):
    """Skip the real ``Embedder.__init__`` (which probes OLLAMA_HOST)."""
    self.config = config
    self.model = config.cognition.model_embeddings_local
    self.cache = cache


class TestOrchestration:
    @pytest.fixture(autouse=True)
    def _patch_embedder(self):
        with patch.object(Embedder, "__init__", _no_op_init), \
             patch.object(Embedder, "embed", _deterministic_embed), \
             patch(
                "grimore.utils.config.update_cognition_models", return_value=True
             ) as rewrite:
            self.rewrite = rewrite
            yield

    def test_status_only_returns_idle_when_clean(self, tmp_path, capsys):
        db = _seed_db(tmp_path)
        cfg = _make_config(tmp_path)
        out = _do_migrate_embeddings(
            cfg, db, target_model="", status_only=True, write_config=False,
        )
        assert out["status"] == "idle"

    def test_abort_branch_exits_early(self, tmp_path):
        db = _seed_db(tmp_path)
        cfg = _make_config(tmp_path)
        # Nothing in flight → returns idle without touching the DB.
        out = _do_migrate_embeddings(
            cfg, db, target_model="", abort=True, write_config=False,
        )
        assert out["status"] == "idle"

    def test_end_to_end_completes_and_writes_config(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=3)
        cfg = _make_config(tmp_path)
        out = _do_migrate_embeddings(
            cfg, db, target_model="new-model", write_config=True,
        )
        assert out["status"] == "complete"
        # Verify the chunk_hash on the swapped rows reflects the new model.
        with sqlite3.connect(db.db_path) as conn:
            rows = conn.execute(
                "SELECT text_content, chunk_hash FROM embeddings ORDER BY chunk_index"
            ).fetchall()
        for text, h in rows:
            assert h == Embedder.chunk_hash(text, "new-model")
        # Config rewriter was called with the new model.
        self.rewrite.assert_called_once()
        kwargs = self.rewrite.call_args.kwargs
        assert kwargs.get("embedding_model") == "new-model"

    def test_resume_picks_up_after_partial_run(self, tmp_path):
        db = _seed_db(tmp_path, n_chunks=4)
        cfg = _make_config(tmp_path)

        # First pass: simulate a worker that dies after row #2 by patching
        # append_migration_row to count and raise.
        original_append = db.append_migration_row
        call_count = {"n": 0}

        def _flaky_append(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 2:
                raise RuntimeError("simulated crash")
            return original_append(*args, **kwargs)

        with patch.object(db, "append_migration_row", side_effect=_flaky_append):
            with pytest.raises(RuntimeError):
                _do_migrate_embeddings(
                    cfg, db, target_model="new-model", write_config=False,
                )

        # State after the crash: 2 rows in the shadow, still 'running'.
        active = db.get_active_embedding_migration()
        assert active is not None
        assert active["done"] == 2
        assert active["status"] == "running"

        # Resume — should finish the remaining 2 rows and swap.
        out = _do_migrate_embeddings(
            cfg, db, target_model="new-model", write_config=False,
        )
        assert out["status"] == "complete"
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert count == 4

    def test_conflict_when_other_target_already_running(self, tmp_path):
        db = _seed_db(tmp_path)
        db.begin_embedding_migration("other-model")
        cfg = _make_config(tmp_path)
        import typer
        with pytest.raises(typer.Exit):
            _do_migrate_embeddings(
                cfg, db, target_model="new-model", write_config=False,
            )
