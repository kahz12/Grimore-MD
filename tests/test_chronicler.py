"""Tests for the Chronicler.

We test three surfaces:
  * ``resolve_window_days`` — pure function, no DB.
  * ``Chronicler`` engine — exercises seed / list_stale / verify /
    check_decay against a real SQLite DB and a fake LLM.
  * Shell wiring — ``chronicler`` command is registered and dispatches.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from grimore.cognition.chronicler import (
    Chronicler,
    StaleNote,
    resolve_window_days,
)
from grimore.memory.db import Database
from grimore.session import Session
from grimore.shell import GrimoreShell
from grimore.utils.config import (
    ChroniclerConfig,
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def chronicler_config(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return Config(
        vault=VaultConfig(path=str(vault_dir), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


def _seed_note(db: Database, path: str, title: str, category: str | None,
               last_tagged_iso: str | None = None) -> int:
    """Insert a note and (optionally) backdate ``last_tagged``.

    The seed step uses ``COALESCE(last_tagged, last_seen)`` as the
    initial verification anchor. Backdating ``last_tagged`` is how we
    construct an overdue note without sleeping or freezing time inside
    the engine.
    """
    nid = db.upsert_note(path, title, content_hash="h-" + path)
    if category is not None:
        db.set_note_category(nid, category)
    if last_tagged_iso is not None:
        with db._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET last_tagged = ? WHERE id = ?",
                (last_tagged_iso, nid),
            )
            conn.commit()
    return nid


# ── resolve_window_days ──────────────────────────────────────────────────


class TestResolveWindow:
    def test_returns_zero_for_none_category(self):
        assert resolve_window_days(None, ChroniclerConfig()) == 0

    def test_returns_zero_for_unmatched_category(self):
        assert resolve_window_days("History/Rome", ChroniclerConfig()) == 0

    def test_matches_tech_prefix_case_insensitive(self):
        assert resolve_window_days("Tech/Linux/Termux", ChroniclerConfig()) == 90

    def test_dev_window_is_180(self):
        assert resolve_window_days("dev/python", ChroniclerConfig()) == 180

    def test_exempt_category_returns_zero(self):
        # concepts/ is mapped to 0 (never stale) in the suggested defaults
        assert resolve_window_days("concepts/causality", ChroniclerConfig()) == 0

    def test_first_matching_rule_wins(self):
        cfg = ChroniclerConfig(windows={"tech/": 30, "tech/legacy/": 365})
        # tech/ matches first by insertion order, even if a more specific
        # prefix is later in the dict.
        assert resolve_window_days("tech/legacy/cobol", cfg) == 30


# ── Chronicler.seed ──────────────────────────────────────────────────────


class TestSeed:
    def test_seed_creates_rows_for_matching_categories(self, chronicler_config):
        session = Session(chronicler_config)
        _seed_note(session.db, "/v/a.md", "A", "tech/linux")
        _seed_note(session.db, "/v/b.md", "B", "concepts/causality")  # exempt
        _seed_note(session.db, "/v/c.md", "C", "history/rome")  # unmatched

        report = Chronicler(session).seed()
        assert report.added == 1
        assert report.updated == 0
        assert report.removed == 0

        rows = session.db.list_freshness()
        assert rows == [("/v/a.md", 90)]

    def test_second_seed_is_idempotent(self, chronicler_config):
        session = Session(chronicler_config)
        _seed_note(session.db, "/v/a.md", "A", "tech/linux")
        chronicler = Chronicler(session)

        chronicler.seed()
        # Capture the timestamp the first seed committed.
        first = session.db.get_freshness_row("/v/a.md")
        assert first is not None
        first_last_verified = first[0]

        report = chronicler.seed()
        assert report.added == 0  # nothing new
        assert report.updated == 0
        # last_verified must be preserved across re-seeds.
        second = session.db.get_freshness_row("/v/a.md")
        assert second is not None
        assert second[0] == first_last_verified

    def test_seed_updates_window_when_config_changes(self, chronicler_config):
        session = Session(chronicler_config)
        _seed_note(session.db, "/v/a.md", "A", "tech/linux")

        Chronicler(session).seed()
        # Now narrow tech/ from 90d to 30d and re-seed.
        tighter = ChroniclerConfig(windows={"tech/": 30})
        report = Chronicler(session, cfg=tighter).seed()
        assert report.updated == 1

        rows = session.db.list_freshness()
        assert rows == [("/v/a.md", 30)]

    def test_seed_removes_rows_for_now_exempt_categories(self, chronicler_config):
        session = Session(chronicler_config)
        _seed_note(session.db, "/v/a.md", "A", "tech/linux")
        Chronicler(session).seed()

        # Re-categorize as exempt.
        with session.db._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET category = 'concepts/causality' WHERE path = ?",
                ("/v/a.md",),
            )
            conn.commit()

        report = Chronicler(session).seed()
        assert report.removed == 1
        assert session.db.list_freshness() == []


# ── Chronicler.list_stale ───────────────────────────────────────────────


class TestListStale:
    def test_returns_empty_when_nothing_overdue(self, chronicler_config):
        session = Session(chronicler_config)
        # Note tagged TODAY; tech/ window is 90d → not stale.
        _seed_note(session.db, "/v/a.md", "A", "tech/linux",
                   last_tagged_iso=datetime.now(timezone.utc).isoformat())

        stale = Chronicler(session).list_stale()
        assert stale == []

    def test_returns_overdue_notes(self, chronicler_config):
        session = Session(chronicler_config)
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        _seed_note(session.db, "/v/old.md", "Old", "tech/linux", last_tagged_iso=old)
        # Recent note — should NOT appear.
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        _seed_note(session.db, "/v/recent.md", "Recent", "tech/linux",
                   last_tagged_iso=recent)

        stale = Chronicler(session).list_stale()
        assert len(stale) == 1
        assert stale[0].path == "/v/old.md"
        assert stale[0].days_overdue >= 100  # 200d - 90d window

    def test_results_are_sorted_most_overdue_first(self, chronicler_config):
        session = Session(chronicler_config)
        old_a = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        old_b = (datetime.now(timezone.utc) - timedelta(days=300)).isoformat()
        _seed_note(session.db, "/v/a.md", "A", "tech/linux", last_tagged_iso=old_a)
        _seed_note(session.db, "/v/b.md", "B", "tech/linux", last_tagged_iso=old_b)

        stale = Chronicler(session).list_stale()
        assert [n.path for n in stale] == ["/v/b.md", "/v/a.md"]
        assert stale[0].days_overdue > stale[1].days_overdue


# ── Chronicler.verify ───────────────────────────────────────────────────


class TestVerify:
    def test_verify_advances_last_verified(self, chronicler_config):
        session = Session(chronicler_config)
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        _seed_note(session.db, "/v/a.md", "A", "tech/linux", last_tagged_iso=old)

        chronicler = Chronicler(session)
        # First list_stale auto-seeds.
        assert len(chronicler.list_stale()) == 1

        assert chronicler.verify("/v/a.md") is True
        # After verify the note is no longer overdue.
        assert chronicler.list_stale() == []

    def test_verify_unknown_path_returns_false(self, chronicler_config):
        session = Session(chronicler_config)
        # Note exists but its category is exempt → no freshness row.
        _seed_note(session.db, "/v/a.md", "A", "concepts/causality")
        Chronicler(session).seed()
        assert Chronicler(session).verify("/v/a.md") is False


# ── Chronicler.check_decay ──────────────────────────────────────────────


class TestCheckDecay:
    def _make_note_file(self, vault_dir: Path, name: str, body: str) -> str:
        path = vault_dir / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def test_returns_none_when_no_freshness_row(self, chronicler_config):
        session = Session(chronicler_config)
        path = self._make_note_file(
            Path(chronicler_config.vault.path), "x.md", "# X"
        )
        _seed_note(session.db, path, "X", "concepts/foo")  # exempt
        Chronicler(session).seed()

        assert Chronicler(session).check_decay(path) is None

    def test_persists_verdict_to_freshness_row(self, chronicler_config, monkeypatch):
        session = Session(chronicler_config)
        path = self._make_note_file(
            Path(chronicler_config.vault.path), "linux.md",
            "Use `sudo apt-get install python3` to install on Ubuntu 14.04.",
        )
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        _seed_note(session.db, path, "Linux", "tech/linux", last_tagged_iso=old)
        chronicler = Chronicler(session)
        chronicler.seed()

        fake_router = MagicMock()
        fake_router.complete.return_value = {
            "likely_stale": True,
            "reasons": ["References Ubuntu 14.04 — unsupported since 2019"],
        }
        # Replace the cached router on the live Session.
        session._router = fake_router

        result = chronicler.check_decay(path)
        assert result is not None
        assert result["likely_stale"] is True

        row = session.db.get_freshness_row(path)
        assert row is not None
        # row = (last_verified, window_days, decay_check_at, likely_stale)
        assert row[2] is not None  # decay_check_at populated
        assert row[3] == 1

    def test_returns_none_on_malformed_llm_payload(self, chronicler_config):
        session = Session(chronicler_config)
        path = self._make_note_file(
            Path(chronicler_config.vault.path), "x.md", "# X"
        )
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        _seed_note(session.db, path, "X", "tech/linux", last_tagged_iso=old)
        Chronicler(session).seed()

        fake_router = MagicMock()
        fake_router.complete.return_value = {"oops": "no likely_stale key"}
        session._router = fake_router

        assert Chronicler(session).check_decay(path) is None
        # Verdict was not persisted.
        row = session.db.get_freshness_row(path)
        assert row[3] is None


# ── Shell wiring ────────────────────────────────────────────────────────


class TestShellWiring:
    def test_chronicler_command_registered(self, chronicler_config):
        shell = GrimoreShell(Session(chronicler_config))
        assert "chronicler" in shell.commands

    def test_chronicler_help_text_documents_subcommands(self, chronicler_config):
        shell = GrimoreShell(Session(chronicler_config))
        text = shell._help_text["chronicler"]
        assert "list" in text and "check" in text and "verify" in text

    def test_chronicler_no_args_prints_error(self, chronicler_config, capsys):
        shell = GrimoreShell(Session(chronicler_config))
        shell.dispatch("chronicler")
        out = capsys.readouterr().out
        assert "missing subcommand" in out
        assert shell._running is True

    def test_chronicler_list_against_empty_vault(self, chronicler_config, capsys):
        shell = GrimoreShell(Session(chronicler_config))
        shell.dispatch("chronicler list")
        out = capsys.readouterr().out
        # No notes → success panel says vault is current.
        assert "current" in out.lower()
