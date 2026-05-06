"""Tests for the Synthesizer engine.

Exercise the engine end-to-end against a real SQLite DB plus a fake
LLM router and a deterministic embedder. Three layers:

  * Pure helpers — ``_slugify``, ``_centroid``, selector normalization.
  * ``Synthesizer.distill`` — selector resolution, generated-flag
    exclusion (the feedback-loop guard), centroid + top-K passage
    selection, output frontmatter, atomic write, dry-run path.
  * Shell wiring — the ``distill`` command is registered, dispatches
    cleanly, and prints helpful error text on bad inputs.
"""
from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from grimore.cognition.embedder import Embedder
from grimore.cognition.synthesizer import (
    GENERATED_FLAG_KEY,
    SYNTHESIS_DIRNAME,
    Synthesizer,
    _slugify,
)
from grimore.session import Session
from grimore.shell import GrimoreShell
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


# ── fakes ────────────────────────────────────────────────────────────────


class _QueuedRouter:
    """Returns each queued payload from ``complete`` in order. The
    Synthesizer only makes one LLM call per ``distill`` invocation, so a
    single-element queue covers the common case."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, prompt, system_prompt="", json_format=True, model_override=None):
        self.calls.append({"prompt": prompt, "system": system_prompt})
        return self.payloads.pop(0) if self.payloads else None


class _MapEmbedder:
    """Embedder fake driven by a {text: vector} dict.

    Synthesizer's ``_gather_sources`` only uses ``deserialize_vector``
    (chunks come from the DB) — we don't need ``embed`` itself for the
    distill path. Vectors are passed through ``Embedder.serialize_vector``
    when we seed the DB so deserialization mirrors production exactly.
    """

    def __init__(self, vectors=None):
        self.vectors = dict(vectors or {})

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
def synth_config(tmp_path, vault):
    return Config(
        vault=VaultConfig(path=str(vault), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


def _write_note(vault: Path, name: str, body: str, *, tags=None, category=None,
                generated: bool = False) -> str:
    """Materialize a markdown file with optional frontmatter."""
    meta_lines = []
    if tags:
        meta_lines.append(f"tags: [{', '.join(tags)}]")
    if category:
        meta_lines.append(f"category: {category}")
    if generated:
        meta_lines.append(f"{GENERATED_FLAG_KEY}: true")
    if meta_lines:
        front = "\n".join(meta_lines)
        text = f"---\n{front}\n---\n\n{body}"
    else:
        text = body
    p = vault / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def _seed(session: Session, path: str, title: str, *, tags=None, category=None,
          chunks=None) -> int:
    """Register a note in the DB exactly the way ``scan`` would: notes,
    tags, category, embeddings."""
    nid = session.db.upsert_note(path, title, content_hash="h-" + path)
    if tags:
        session.db.upsert_tags(nid, tags)
    if category:
        session.db.set_note_category(nid, category)
    if chunks:
        for idx, (text, vec) in enumerate(chunks):
            session.db.store_embedding(
                nid, idx, text, Embedder.serialize_vector(vec),
            )
    return nid


def _wire(session: Session, *, payloads=None, vectors=None) -> _QueuedRouter:
    router = _QueuedRouter(payloads or [])
    session._router = router
    session._embedder = _MapEmbedder(vectors)
    return router


# ── pure helpers ─────────────────────────────────────────────────────────


class TestSlugify:
    def test_replaces_path_separators(self):
        assert _slugify("Tech/Linux/Termux") == "Tech_Linux_Termux"

    def test_collapses_runs_of_special_chars(self):
        assert _slugify("a!!!  b???c") == "a_b_c"

    def test_strips_leading_and_trailing_underscores(self):
        assert _slugify("///foo///") == "foo"

    def test_falls_back_when_input_is_all_punctuation(self):
        # Empty after cleaning → fallback string, never empty.
        assert _slugify("///") == "synthesis"

    def test_caps_length(self):
        long_in = "abc" * 100
        assert len(_slugify(long_in)) <= 60


class TestCentroid:
    def test_empty_returns_none(self):
        assert Synthesizer._centroid([]) is None

    def test_single_vector_returns_unit_normed_copy(self):
        # Already unit-length input.
        v = [1.0, 0.0, 0.0]
        out = Synthesizer._centroid([v])
        assert out is not None
        assert pytest.approx(out[0], abs=1e-6) == 1.0
        assert pytest.approx(sum(x * x for x in out), abs=1e-6) == 1.0

    def test_two_orthogonal_vectors_average_then_normalize(self):
        out = Synthesizer._centroid([[1.0, 0.0], [0.0, 1.0]])
        assert out is not None
        # Mean is (0.5, 0.5); normalized → (1/√2, 1/√2).
        expect = 1 / (2 ** 0.5)
        assert pytest.approx(out[0], abs=1e-6) == expect
        assert pytest.approx(out[1], abs=1e-6) == expect

    def test_zero_vector_input_returns_none(self):
        # Anti-aligned unit vectors → mean is the zero vector.
        assert Synthesizer._centroid([[1.0, 0.0], [-1.0, 0.0]]) is None


class TestSelectorNormalization:
    def test_tag_only(self):
        assert Synthesizer._normalize_selector(tag="python", category=None) == ("tag", "python")

    def test_category_only_strips_trailing_slash(self):
        kind, value = Synthesizer._normalize_selector(tag=None, category="Tech/Linux/")
        assert kind == "category"
        assert value == "Tech/Linux"

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            Synthesizer._normalize_selector(tag=None, category=None)

    def test_both_raises(self):
        with pytest.raises(ValueError):
            Synthesizer._normalize_selector(tag="x", category="y")

    def test_blank_tag_raises(self):
        with pytest.raises(ValueError):
            Synthesizer._normalize_selector(tag="   ", category=None)


# ── distill — selector resolution ───────────────────────────────────────


class TestSelectorResolution:
    def test_no_match_skips_with_reason(self, synth_config):
        session = Session(synth_config)
        _wire(session)
        report = Synthesizer(session).distill(tag="nonexistent")
        assert report.notes_used == 0
        assert report.skipped_reason is not None
        assert "match" in report.skipped_reason.lower()

    def test_tag_resolves_notes_with_that_tag(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "# A\nLogs are useful.", tags=["python"])
        path_b = _write_note(vault, "b.md", "# B\nUnrelated.", tags=["other"])
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["python"], chunks=[("Logs are useful.", v)])
        _seed(session, path_b, "B", tags=["other"], chunks=[("Unrelated.", v)])
        _wire(session, payloads=[{"title": "Logs", "body": "## Summary\n[[A]] mentions logs."}])

        report = Synthesizer(session).distill(tag="python", dry_run=True)
        assert report.notes_used == 1
        assert report.sources == ["A"]


# ── distill — feedback-loop guard ───────────────────────────────────────


class TestGeneratedExclusion:
    def test_generated_notes_are_filtered_out(self, synth_config, vault):
        session = Session(synth_config)
        # Real source note.
        path_a = _write_note(vault, "a.md", "# A\nReal content.", tags=["topic"])
        # Synthesis output from a previous run — same tag, but should be skipped.
        path_synth = _write_note(
            vault, "_synthesis/topic_2026-04-01.md",
            "# Old synthesis\nbody",
            tags=["topic"], generated=True,
        )
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["topic"], chunks=[("Real content.", v)])
        _seed(session, path_synth, "Old synthesis",
              tags=["topic"], chunks=[("body", v)])
        _wire(session, payloads=[{"title": "T", "body": "## X\n[[A]] body"}])

        report = Synthesizer(session).distill(tag="topic", dry_run=True)
        assert report.notes_excluded_generated == 1
        assert report.notes_used == 1
        assert "A" in report.sources

    def test_only_generated_matches_skips_with_reason(self, synth_config, vault):
        session = Session(synth_config)
        path = _write_note(
            vault, "_synthesis/old.md",
            "# Old\nbody", tags=["solo"], generated=True,
        )
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path, "Old", tags=["solo"], chunks=[("body", v)])
        _wire(session, payloads=[])  # no LLM call expected

        report = Synthesizer(session).distill(tag="solo", dry_run=True)
        assert report.notes_used == 0
        assert "grimore_generated" in (report.skipped_reason or "")


# ── distill — happy path ────────────────────────────────────────────────


class TestHappyPath:
    def test_writes_synthesis_note_with_frontmatter_and_sources(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Logs are great.", tags=["py"])
        path_b = _write_note(vault, "b.md", "Logs need rotation.", tags=["py"])
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["py"], chunks=[("Logs are great.", v)])
        _seed(session, path_b, "B", tags=["py"], chunks=[("Logs need rotation.", v)])
        _wire(session, payloads=[{
            "title": "Logging in Python",
            "body": "## Summary\nLogs matter ([[A]], [[B]]).",
        }])

        report = Synthesizer(session).distill(tag="py", dry_run=False)
        assert report.output_path is not None
        # File ended up in the right place and has the generated flag.
        out = Path(report.output_path)
        assert out.parent.name == SYNTHESIS_DIRNAME
        assert out.exists()
        post = frontmatter.load(out)
        assert post.metadata[GENERATED_FLAG_KEY] is True
        assert post.metadata["selector"] == "tag:py"
        # Source paths are recorded so a future Mirror run can map back.
        assert sorted(post.metadata["sources"]) == sorted([path_a, path_b])
        # Title from LLM payload made it into the markdown body.
        assert "Logging in Python" in post.content
        assert "[[A]]" in post.content

    def test_dry_run_does_not_write_file(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Body.", tags=["t"])
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["t"], chunks=[("Body.", v)])
        _wire(session, payloads=[{"title": "T", "body": "## S\nBody."}])

        report = Synthesizer(session).distill(tag="t", dry_run=True)
        assert report.output_path is not None
        assert not Path(report.output_path).exists()

    def test_top_k_passages_picked_per_note(self, synth_config, vault):
        """Per-note passage selection should cap at ``passages_per_note``."""
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Long note.", tags=["t"])
        # Identical vectors so all chunks tie on similarity; the cap is
        # what's being asserted here, not the ranking.
        v = [1.0, 0.0, 0.0, 0.0]
        chunks = [(f"chunk {i}", v) for i in range(7)]
        _seed(session, path_a, "A", tags=["t"], chunks=chunks)
        router = _wire(session, payloads=[{"title": "T", "body": "## S\nb"}])

        Synthesizer(session).distill(tag="t", passages_per_note=2, dry_run=True)
        # Inspect the LLM prompt: at most 2 ``--- Source: [[A]] ---``
        # blocks worth of passages should appear.
        prompt = router.calls[0]["prompt"]
        # Each passage is wrapped in <passage> tags by SecurityGuard.
        assert prompt.count("<passage>") == 2


# ── distill — failure / no-op paths ─────────────────────────────────────


class TestSkipReasons:
    def test_no_embeddings_skips(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Body.", tags=["t"])
        # Note registered, but no chunks indexed yet.
        _seed(session, path_a, "A", tags=["t"], chunks=None)
        _wire(session, payloads=[])

        report = Synthesizer(session).distill(tag="t", dry_run=True)
        assert report.notes_used == 0
        assert "scan" in (report.skipped_reason or "").lower()

    def test_llm_returns_none_skips(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Body.", tags=["t"])
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["t"], chunks=[("Body.", v)])
        _wire(session, payloads=[None])

        report = Synthesizer(session).distill(tag="t", dry_run=True)
        assert report.output_path is None
        assert "no structured" in (report.skipped_reason or "").lower()

    def test_llm_payload_without_body_skips(self, synth_config, vault):
        session = Session(synth_config)
        path_a = _write_note(vault, "a.md", "Body.", tags=["t"])
        v = [1.0, 0.0, 0.0, 0.0]
        _seed(session, path_a, "A", tags=["t"], chunks=[("Body.", v)])
        _wire(session, payloads=[{"title": "Only title"}])

        report = Synthesizer(session).distill(tag="t", dry_run=True)
        assert report.output_path is None
        assert "body" in (report.skipped_reason or "").lower()


# ── shell wiring ────────────────────────────────────────────────────────


class TestShellWiring:
    def test_distill_command_registered(self, synth_config):
        shell = GrimoreShell(Session(synth_config))
        assert "distill" in shell.commands

    def test_distill_help_documents_options(self, synth_config):
        shell = GrimoreShell(Session(synth_config))
        text = shell._help_text["distill"]
        assert "--tag" in text and "--category" in text

    def test_distill_with_no_selector_prints_error(self, synth_config, capsys):
        shell = GrimoreShell(Session(synth_config))
        shell.dispatch("distill")
        out = capsys.readouterr().out
        assert "--tag" in out and "--category" in out
        assert shell._running is True

    def test_distill_with_both_selectors_prints_error(self, synth_config, capsys):
        shell = GrimoreShell(Session(synth_config))
        shell.dispatch("distill --tag python --category Tech")
        out = capsys.readouterr().out
        assert "pick one" in out.lower()
