"""Tests for the Mirror engine.

The pipeline is exercised end-to-end against a real SQLite DB but with
the LLM router and embedder mocked. The router returns canned
extraction payloads then canned contradiction verdicts in the order
Mirror.scan calls them; the embedder reads from a dict so similarity
is fully under the test's control.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from grimoire.cognition.embedder import Embedder
from grimoire.cognition.mirror import Mirror
from grimoire.session import Session
from grimoire.shell import GrimoireShell
from grimoire.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


# ── fakes ────────────────────────────────────────────────────────────────


class _RoutedRouter:
    """Routes ``complete`` calls into one of two queues based on the
    system prompt, so a test can supply extraction payloads and
    contradiction verdicts independently of call order."""

    EXTRACTION_MARKER = "atomic factual claims"
    CONTRADICTION_MARKER = "two atomic claims drawn from different"

    def __init__(self, *, extractions=None, verdicts=None):
        self.extractions = list(extractions or [])
        self.verdicts = list(verdicts or [])
        self.calls: list[dict] = []

    def complete(self, prompt, system_prompt="", json_format=True, model_override=None):
        self.calls.append({"prompt": prompt, "system": system_prompt})
        if self.EXTRACTION_MARKER in system_prompt:
            return self.extractions.pop(0) if self.extractions else None
        if self.CONTRADICTION_MARKER in system_prompt:
            return self.verdicts.pop(0) if self.verdicts else None
        return None


class _MapEmbedder:
    """Deterministic embedder driven by a {text: vector} dict."""

    def __init__(self, vectors):
        self.vectors = dict(vectors)

    def embed(self, text):
        return self.vectors.get(text)

    def serialize_vector(self, vec):
        return Embedder.serialize_vector(vec)

    @staticmethod
    def deserialize_vector(blob):
        return Embedder.deserialize_vector(blob)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture
def mirror_config(tmp_path, vault):
    return Config(
        vault=VaultConfig(path=str(vault), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimoire.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


def _write_note(vault: Path, name: str, body: str) -> str:
    p = vault / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def _seed_note(session: Session, path: str, title: str) -> int:
    """Mirror.scan walks the ``notes`` table, not the filesystem; we
    have to register each test note there so it gets picked up."""
    return session.db.upsert_note(path, title, content_hash="h-" + path)


def _wire(session, *, extractions, verdicts, vectors):
    """Inject the fakes onto the live Session."""
    session._router = _RoutedRouter(extractions=extractions, verdicts=verdicts)
    session._embedder = _MapEmbedder(vectors)
    return session


# ── empty-input behavior ─────────────────────────────────────────────────


class TestEmpty:
    def test_scan_empty_vault(self, mirror_config):
        session = _wire(Session(mirror_config), extractions=[], verdicts=[], vectors={})
        report = Mirror(session).scan()
        assert report.notes_scanned == 0
        assert report.claims_extracted == 0
        assert report.pairs_checked == 0
        assert report.contradictions_found == 0

    def test_list_open_no_claims(self, mirror_config):
        session = _wire(Session(mirror_config), extractions=[], verdicts=[], vectors={})
        assert Mirror(session).list_open() == []


# ── happy path ──────────────────────────────────────────────────────────


class TestHappyPath:
    def test_scan_records_contradiction_when_llm_says_so(self, mirror_config, vault):
        path_a = _write_note(vault, "a.md", "Python is interpreted.")
        path_b = _write_note(vault, "b.md", "Python is compiled, not interpreted.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        _seed_note(session, path_b, "B")

        # Embedder returns the *same* vector for both claims so they pass
        # the similarity floor.
        v = [1.0, 0.0, 0.0, 0.0]
        _wire(
            session,
            extractions=[
                {"claims": ["Python is interpreted."]},
                {"claims": ["Python is compiled, not interpreted."]},
            ],
            verdicts=[
                {"contradicts": True, "severity": "high",
                 "explanation": "Asserts opposite execution models."},
            ],
            vectors={
                "Python is interpreted.": v,
                "Python is compiled, not interpreted.": v,
            },
        )

        report = Mirror(session).scan()
        assert report.notes_scanned == 2
        assert report.claims_extracted == 2
        assert report.pairs_checked == 1
        assert report.contradictions_found == 1

        rows = Mirror(session).list_open()
        assert len(rows) == 1
        assert rows[0].severity == "high"
        assert "execution models" in rows[0].explanation

    def test_no_contradiction_when_llm_says_so(self, mirror_config, vault):
        path_a = _write_note(vault, "a.md", "Cats are mammals.")
        path_b = _write_note(vault, "b.md", "Dogs are mammals.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        _seed_note(session, path_b, "B")
        v = [1.0, 0.0, 0.0, 0.0]
        _wire(
            session,
            extractions=[
                {"claims": ["Cats are mammals."]},
                {"claims": ["Dogs are mammals."]},
            ],
            verdicts=[
                {"contradicts": False, "severity": "low",
                 "explanation": "Both can be true."},
            ],
            vectors={
                "Cats are mammals.": v,
                "Dogs are mammals.": v,
            },
        )

        report = Mirror(session).scan()
        assert report.pairs_checked == 1
        assert report.contradictions_found == 0
        assert Mirror(session).list_open() == []


# ── filtering ───────────────────────────────────────────────────────────


class TestFiltering:
    def test_similarity_floor_drops_unrelated_pairs(self, mirror_config, vault):
        path_a = _write_note(vault, "a.md", "Python is interpreted.")
        path_b = _write_note(vault, "b.md", "Bananas are yellow.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        _seed_note(session, path_b, "B")

        # Orthogonal vectors → cosine = 0 → below floor → no LLM call.
        _wire(
            session,
            extractions=[
                {"claims": ["Python is interpreted."]},
                {"claims": ["Bananas are yellow."]},
            ],
            verdicts=[],  # asserts no contradiction LLM calls
            vectors={
                "Python is interpreted.": [1.0, 0.0, 0.0, 0.0],
                "Bananas are yellow.":    [0.0, 1.0, 0.0, 0.0],
            },
        )

        report = Mirror(session).scan()
        assert report.claims_extracted == 2
        assert report.pairs_checked == 0
        assert report.contradictions_found == 0

    def test_same_note_pairs_are_skipped(self, mirror_config, vault):
        path_a = _write_note(vault, "a.md", "Two related lines about Python.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        v = [1.0, 0.0, 0.0, 0.0]
        _wire(
            session,
            extractions=[
                {"claims": ["Python is great.", "Python is interpreted."]},
            ],
            verdicts=[],  # no cross-note pairs available
            vectors={
                "Python is great.":       v,
                "Python is interpreted.": v,
            },
        )

        report = Mirror(session).scan()
        assert report.claims_extracted == 2
        assert report.pairs_checked == 0


# ── status workflow ─────────────────────────────────────────────────────


class TestStatusWorkflow:
    def _seed_open_contradiction(self, mirror_config, vault):
        path_a = _write_note(vault, "a.md", "Python is interpreted.")
        path_b = _write_note(vault, "b.md", "Python is compiled.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        _seed_note(session, path_b, "B")
        v = [1.0, 0.0, 0.0, 0.0]
        _wire(
            session,
            extractions=[
                {"claims": ["Python is interpreted."]},
                {"claims": ["Python is compiled."]},
            ],
            verdicts=[
                {"contradicts": True, "severity": "high", "explanation": "Opposite."},
            ],
            vectors={
                "Python is interpreted.": v,
                "Python is compiled.":    v,
            },
        )
        Mirror(session).scan()
        return session

    def test_dismiss_hides_from_list_open(self, mirror_config, vault):
        session = self._seed_open_contradiction(mirror_config, vault)
        mirror = Mirror(session)
        rows = mirror.list_open()
        assert len(rows) == 1
        cid = rows[0].id

        assert mirror.dismiss(cid) is True
        assert mirror.list_open() == []

    def test_resolve_hides_from_list_open(self, mirror_config, vault):
        session = self._seed_open_contradiction(mirror_config, vault)
        mirror = Mirror(session)
        cid = mirror.list_open()[0].id

        assert mirror.resolve(cid) is True
        assert mirror.list_open() == []

    def test_dismiss_unknown_id_returns_false(self, mirror_config):
        session = _wire(Session(mirror_config), extractions=[], verdicts=[], vectors={})
        assert Mirror(session).dismiss(999) is False

    def test_show_returns_full_detail(self, mirror_config, vault):
        session = self._seed_open_contradiction(mirror_config, vault)
        mirror = Mirror(session)
        cid = mirror.list_open()[0].id

        detail = mirror.show(cid)
        assert detail is not None
        assert detail.id == cid
        assert detail.severity == "high"
        assert detail.claim_a in {"Python is interpreted.", "Python is compiled."}
        assert detail.claim_b in {"Python is interpreted.", "Python is compiled."}
        assert detail.claim_a != detail.claim_b
        # Surrounding-paragraph context is recovered.
        assert detail.context_a is not None
        assert detail.context_b is not None

    def test_show_unknown_id_returns_none(self, mirror_config):
        session = _wire(Session(mirror_config), extractions=[], verdicts=[], vectors={})
        assert Mirror(session).show(999) is None


# ── dismissal-persistence (the load-bearing invariant) ──────────────────


class TestDismissalPersistence:
    def test_rescan_does_not_re_flag_dismissed_pair(self, mirror_config, vault):
        # First scan creates and flags the pair.
        path_a = _write_note(vault, "a.md", "Python is interpreted.")
        path_b = _write_note(vault, "b.md", "Python is compiled.")
        session = Session(mirror_config)
        _seed_note(session, path_a, "A")
        _seed_note(session, path_b, "B")
        v = [1.0, 0.0, 0.0, 0.0]
        _wire(
            session,
            extractions=[
                {"claims": ["Python is interpreted."]},
                {"claims": ["Python is compiled."]},
            ],
            verdicts=[
                {"contradicts": True, "severity": "high", "explanation": "Opposite."},
            ],
            vectors={
                "Python is interpreted.": v,
                "Python is compiled.":    v,
            },
        )
        mirror = Mirror(session)
        first = mirror.scan()
        assert first.contradictions_found == 1
        cid = mirror.list_open()[0].id

        # User dismisses it.
        assert mirror.dismiss(cid) is True

        # Re-wire fakes so a SECOND extraction returns the same texts (so
        # the claim ids stay the same — that's how persistence works).
        # The verdict queue is empty: the test asserts no LLM call is
        # made for the already-known pair.
        session._router = _RoutedRouter(
            extractions=[
                {"claims": ["Python is interpreted."]},
                {"claims": ["Python is compiled."]},
            ],
            verdicts=[],  # contradiction check should NOT happen
        )

        # Force re-extraction by bumping mtime.
        import os, time
        future = time.time() + 5
        os.utime(path_a, (future, future))
        os.utime(path_b, (future, future))

        second = Mirror(session).scan(full=False)
        assert second.contradictions_found == 0
        assert second.pairs_checked == 0
        # Dismissed contradiction is still in the table, still dismissed.
        assert Mirror(session).list_open() == []
        assert session.db.count_open_contradictions() == 0


# ── shell wiring ────────────────────────────────────────────────────────


class TestShellWiring:
    def test_mirror_command_registered(self, mirror_config):
        shell = GrimoireShell(Session(mirror_config))
        assert "mirror" in shell.commands

    def test_mirror_help_documents_subcommands(self, mirror_config):
        shell = GrimoireShell(Session(mirror_config))
        text = shell._help_text["mirror"]
        for keyword in ("scan", "show", "dismiss", "resolve"):
            assert keyword in text

    def test_bare_mirror_lists_or_explains(self, mirror_config, capsys):
        # No claims indexed — the panel directs the user to `mirror scan`.
        session = Session(mirror_config)
        session._router = _RoutedRouter()
        session._embedder = _MapEmbedder({})
        shell = GrimoireShell(session)
        shell.dispatch("mirror")
        out = capsys.readouterr().out.lower()
        assert "scan" in out or "consistent" in out

    def test_show_with_non_integer_id(self, mirror_config, capsys):
        shell = GrimoireShell(Session(mirror_config))
        shell.dispatch("mirror show oops")
        out = capsys.readouterr().out
        assert "must be an integer" in out
        assert shell._running is True
