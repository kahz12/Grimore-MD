"""Tests for periodic DB maintenance: VACUUM, WAL checkpoint, tag purge."""
import pytest

from grimore.memory.db import Database
from grimore.memory.maintenance import MaintenanceRunner, MaintenanceReport
from grimore.utils.config import MaintenanceConfig


def _make_db(tmp_path) -> Database:
    return Database(str(tmp_path / "grimore.db"))


def _add_note(db: Database, path: str) -> int:
    return db.upsert_note(path, title=path, content_hash="h-" + path)


class TestWalCheckpoint:
    def test_returns_stats_tuple(self, tmp_path):
        db = _make_db(tmp_path)
        _add_note(db, "a.md")  # force a write so the WAL is non-empty
        result = db.wal_checkpoint()
        assert set(result.keys()) == {"busy", "log_frames", "checkpointed_frames"}
        assert result["busy"] in (0, 1)

    def test_truncate_shrinks_wal_sidecar(self, tmp_path):
        db = _make_db(tmp_path)
        # Write enough to push some frames into the WAL.
        for i in range(20):
            _add_note(db, f"n{i}.md")
        wal = tmp_path / "grimore.db-wal"
        # Checkpoint with TRUNCATE should leave the -wal file at size 0 (or absent).
        db.wal_checkpoint("TRUNCATE")
        assert not wal.exists() or wal.stat().st_size == 0

    def test_rejects_bad_mode(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(ValueError):
            db.wal_checkpoint("nope")


class TestVacuum:
    def test_vacuum_returns_size_stats(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.vacuum()
        assert set(result.keys()) == {"before_bytes", "after_bytes", "reclaimed_bytes"}
        assert result["before_bytes"] >= 0
        assert result["reclaimed_bytes"] >= 0

    def test_vacuum_invariants_after_bulk_delete(self, tmp_path):
        """
        VACUUM should be monotonic (after_bytes <= before_bytes) and leave
        the DB queryable. We don't assert a strict size shrink here because
        FTS5 tombstones from the delete triggers can consume the space that
        VACUUM reclaims from the embeddings table — net size depends on the
        SQLite page allocator. The behaviour we care about is "doesn't grow
        and doesn't corrupt."
        """
        db = _make_db(tmp_path)
        big_blob = b"x" * 4096
        for i in range(100):
            nid = _add_note(db, f"n{i}.md")
            db.store_embedding(nid, 0, f"chunk {i}", big_blob)
        with db._get_connection() as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
        db.wal_checkpoint("TRUNCATE")

        result = db.vacuum()
        assert result["after_bytes"] <= result["before_bytes"]
        assert result["reclaimed_bytes"] == result["before_bytes"] - result["after_bytes"]
        # DB still usable.
        with db._get_connection() as conn:
            conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()


class TestMaintenanceRunner:
    def test_runs_all_enabled_tasks(self, tmp_path):
        db = _make_db(tmp_path)
        nid = _add_note(db, "a.md")
        db.upsert_tags(nid, ["used", "orphan"])
        # Detach "orphan" so purge_unused_tags has something to reclaim.
        with db._get_connection() as conn:
            conn.execute(
                "DELETE FROM note_tags WHERE tag_id = (SELECT id FROM tags WHERE name = 'orphan')"
            )
            conn.commit()

        runner = MaintenanceRunner(db, MaintenanceConfig())
        report = runner.run(reason="test")

        assert isinstance(report, MaintenanceReport)
        assert report.tags_purged == 1
        assert report.checkpoint  # populated dict
        assert report.vacuum      # populated dict
        assert report.skipped == []
        assert report.duration_s >= 0

    def test_respects_disabled_flags(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = MaintenanceConfig(vacuum=False, purge_tags=False, wal_checkpoint=False)
        report = MaintenanceRunner(db, cfg).run()
        assert set(report.skipped) == {"vacuum", "purge_tags", "wal_checkpoint"}
        assert report.tags_purged == 0
        assert report.checkpoint == {}
        assert report.vacuum == {}

    def test_individual_flag_toggles(self, tmp_path):
        db = _make_db(tmp_path)
        _add_note(db, "a.md")
        cfg = MaintenanceConfig(vacuum=True, purge_tags=False, wal_checkpoint=False)
        report = MaintenanceRunner(db, cfg).run()
        assert "purge_tags" in report.skipped
        assert "wal_checkpoint" in report.skipped
        assert "vacuum" not in report.skipped
        assert report.vacuum  # VACUUM ran

    def test_as_dict_shape(self, tmp_path):
        db = _make_db(tmp_path)
        report = MaintenanceRunner(db, MaintenanceConfig()).run()
        out = report.as_dict()
        assert set(out.keys()) == {
            "tags_purged", "checkpoint", "vacuum", "duration_s", "skipped"
        }
        assert isinstance(out["duration_s"], float)

    def test_failure_in_one_task_does_not_abort_others(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        _add_note(db, "a.md")

        def boom(self):
            raise RuntimeError("simulated")

        monkeypatch.setattr(Database, "purge_unused_tags", boom)
        report = MaintenanceRunner(db, MaintenanceConfig()).run()
        # Purge failed, but checkpoint and vacuum should still have run.
        assert "purge_tags" in report.skipped
        assert report.checkpoint
        assert report.vacuum
