"""Duplicate detection (``grimore dedupe``).

Covers the two signals separately — exact ``content_hash`` groups and
near-duplicate mean-vector pairs — plus the numpy/pure-Python parity,
the ragged-vector survival path, and the CLI/operations surface.
"""
import json
import struct

import pytest
from typer.testing import CliRunner

from grimore.cli import app
from grimore.cognition.dedupe import (
    find_exact_duplicates,
    find_near_duplicates,
)
from grimore.cognition.embedder import Embedder
from grimore.memory.db import Database
from grimore.operations import _do_dedupe
from grimore.session import Session
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


def _blob(values):
    return struct.pack(f"{len(values)}f", *values)


def _add_note(db, path, title, content_hash, vectors=()):
    """One note with ``vectors`` as its chunk embeddings (unit-normalized)."""
    note_id = db.upsert_note(path=path, title=title, content_hash=content_hash)
    for i, vec in enumerate(vectors):
        db.store_embedding(
            note_id, i, f"chunk-{i}", _blob(Embedder.normalize(vec)),
            chunk_hash=f"h{i}",
        )
    return note_id


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "grimore.db"))


# ── Exact duplicates ───────────────────────────────────────────────────────


class TestExactDuplicates:
    def test_shared_hash_grouped(self, db):
        a = _add_note(db, "/a.md", "A", "same-hash")
        b = _add_note(db, "/b.md", "B", "same-hash")
        _add_note(db, "/c.md", "C", "unique-hash")

        groups = find_exact_duplicates(db)
        assert len(groups) == 1
        assert groups[0].content_hash == "same-hash"
        assert [nid for nid, _p, _t in groups[0].notes] == [a, b]

    def test_unique_hashes_report_nothing(self, db):
        _add_note(db, "/a.md", "A", "h1")
        _add_note(db, "/b.md", "B", "h2")
        assert find_exact_duplicates(db) == []

    def test_empty_hash_never_groups(self, db):
        # Failed ingests may leave an empty hash; two of those are not dupes.
        _add_note(db, "/a.md", "A", "")
        _add_note(db, "/b.md", "B", "")
        assert find_exact_duplicates(db) == []

    def test_largest_group_first(self, db):
        for i in range(3):
            _add_note(db, f"/t{i}.md", f"T{i}", "triple")
        for i in range(2):
            _add_note(db, f"/p{i}.md", f"P{i}", "pair")

        groups = find_exact_duplicates(db)
        assert [g.content_hash for g in groups] == ["triple", "pair"]


# ── Near-duplicates ────────────────────────────────────────────────────────


class TestNearDuplicates:
    def test_similar_pair_found_orthogonal_excluded(self, db):
        a = _add_note(db, "/a.md", "A", "h1", [[1.0, 0.0, 0.0]])
        b = _add_note(db, "/b.md", "B", "h2", [[0.99, 0.01, 0.0]])
        _add_note(db, "/c.md", "C", "h3", [[0.0, 1.0, 0.0]])

        pairs = find_near_duplicates(db, threshold=0.9)
        assert [(p.a_id, p.b_id) for p in pairs] == [(a, b)]
        assert pairs[0].score == pytest.approx(1.0, abs=1e-3)
        assert pairs[0].a_path == "/a.md" and pairs[0].b_path == "/b.md"

    def test_threshold_excludes_and_limit_truncates(self, db):
        _add_note(db, "/a.md", "A", "h1", [[1.0, 0.0, 0.0]])
        _add_note(db, "/b.md", "B", "h2", [[0.9, 0.1, 0.0]])   # cos ≈ 0.994
        _add_note(db, "/c.md", "C", "h3", [[0.8, 0.2, 0.0]])   # cos(a,c) ≈ 0.970

        assert len(find_near_duplicates(db, threshold=0.9)) == 3
        assert find_near_duplicates(db, threshold=0.999) == []
        # Best score first, so the truncated set keeps the strongest pair.
        top = find_near_duplicates(db, threshold=0.9, limit=1)
        assert len(top) == 1
        assert top[0].score == pytest.approx(
            max(p.score for p in find_near_duplicates(db, threshold=0.9))
        )

    def test_mean_pooling_scores_the_note_not_the_chunks(self, db):
        """Two chunks along different axes must average before comparing:
        each chunk alone scores only ~0.707 against the diagonal note."""
        a = _add_note(db, "/multi.md", "M", "h1",
                      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = _add_note(db, "/diag.md", "D", "h2", [[1.0, 1.0, 0.0]])

        pairs = find_near_duplicates(db, threshold=0.95)
        assert [(p.a_id, p.b_id) for p in pairs] == [(a, b)]
        assert pairs[0].score == pytest.approx(1.0, abs=1e-3)

    def test_exact_hash_pairs_left_to_exact_report(self, db):
        _add_note(db, "/a.md", "A", "same", [[1.0, 0.0]])
        _add_note(db, "/b.md", "B", "same", [[1.0, 0.0]])

        assert find_near_duplicates(db, threshold=0.9) == []
        assert len(find_exact_duplicates(db)) == 1

    def test_empty_db_and_single_note_are_safe(self, db):
        assert find_near_duplicates(db, threshold=0.9) == []
        _add_note(db, "/a.md", "A", "h1", [[1.0, 0.0]])
        assert find_near_duplicates(db, threshold=0.9) == []

    def test_ragged_note_skipped_without_crashing(self, db):
        # Mixed dims inside one note (model swapped mid-scan) → skip it.
        _add_note(db, "/ragged.md", "R", "h1",
                  [[1.0, 0.0, 0.0], [1.0, 0.0]])
        a = _add_note(db, "/a.md", "A", "h2", [[1.0, 0.0, 0.0]])
        b = _add_note(db, "/b.md", "B", "h3", [[1.0, 0.0, 0.0]])

        pairs = find_near_duplicates(db, threshold=0.9)
        assert [(p.a_id, p.b_id) for p in pairs] == [(a, b)]

    def test_cross_dim_notes_are_incomparable(self, db):
        a = _add_note(db, "/a.md", "A", "h1", [[1.0, 0.0, 0.0]])
        b = _add_note(db, "/b.md", "B", "h2", [[1.0, 0.0, 0.0]])
        _add_note(db, "/other-model.md", "O", "h3", [[1.0, 0.0]])

        pairs = find_near_duplicates(db, threshold=0.9)
        assert [(p.a_id, p.b_id) for p in pairs] == [(a, b)]

    def test_python_fallback_matches_numpy(self, db, monkeypatch):
        _add_note(db, "/a.md", "A", "h1", [[1.0, 0.1, 0.0]])
        _add_note(db, "/b.md", "B", "h2", [[0.95, 0.15, 0.0]])
        _add_note(db, "/c.md", "C", "h3", [[0.0, 0.0, 1.0]])

        fast = find_near_duplicates(db, threshold=0.9)
        monkeypatch.setattr("grimore.cognition.dedupe._np", None)
        slow = find_near_duplicates(db, threshold=0.9)

        assert [(p.a_id, p.b_id) for p in fast] == [(p.a_id, p.b_id) for p in slow]
        for f, s in zip(fast, slow, strict=True):
            assert f.score == pytest.approx(s.score, abs=1e-4)


# ── Operations + CLI surface ───────────────────────────────────────────────


def _config(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(exist_ok=True)
    return Config(
        vault=VaultConfig(path=str(vault_dir), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


class TestDoDedupe:
    def test_stats_and_json_export(self, tmp_path):
        config = _config(tmp_path)
        db = Database(config.memory.db_path)
        _add_note(db, "/a.md", "A", "same")
        _add_note(db, "/b.md", "B", "same")
        _add_note(db, "/x.md", "X", "h1", [[1.0, 0.0]])
        _add_note(db, "/y.md", "Y", "h2", [[0.99, 0.01]])

        report_path = tmp_path / "report.json"
        with Session(config) as session:
            stats = _do_dedupe(session, threshold=0.9, export=report_path)

        assert stats == {"exact_groups": 1, "near_pairs": 1}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["threshold"] == 0.9
        assert [n["path"] for n in report["exact"][0]["notes"]] == ["/a.md", "/b.md"]
        assert report["near"][0]["score"] == pytest.approx(1.0, abs=1e-3)

    def test_clean_vault_reports_zero(self, tmp_path):
        config = _config(tmp_path)
        db = Database(config.memory.db_path)
        _add_note(db, "/a.md", "A", "h1", [[1.0, 0.0]])
        _add_note(db, "/b.md", "B", "h2", [[0.0, 1.0]])

        with Session(config) as session:
            stats = _do_dedupe(session)
        assert stats == {"exact_groups": 0, "near_pairs": 0}

    def test_config_threshold_is_the_default(self, tmp_path):
        config = _config(tmp_path)
        config.cognition.dedupe_threshold = 0.999
        db = Database(config.memory.db_path)
        _add_note(db, "/x.md", "X", "h1", [[1.0, 0.0, 0.0]])
        _add_note(db, "/y.md", "Y", "h2", [[0.9, 0.1, 0.0]])  # cos ≈ 0.994

        with Session(config) as session:
            assert _do_dedupe(session)["near_pairs"] == 0
            assert _do_dedupe(session, threshold=0.9)["near_pairs"] == 1


class TestCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_smoke_finds_seeded_duplicates(self, tmp_path, monkeypatch, runner):
        config = _config(tmp_path)
        db = Database(config.memory.db_path)
        _add_note(db, "/a.md", "A", "same")
        _add_note(db, "/b.md", "B", "same")
        monkeypatch.setattr("grimore.cli.load_config", lambda: config)

        result = runner.invoke(app, ["dedupe"])
        assert result.exit_code == 0
        assert "Exact duplicates" in result.output

    @pytest.mark.parametrize("bad", ["-0.5", "1.5"])
    def test_threshold_validation_reused(self, runner, bad):
        result = runner.invoke(app, ["dedupe", "--threshold", bad])
        assert result.exit_code != 0
        assert "must be in [0.0, 1.0]" in (result.output + (result.stderr or ""))
