"""
Semantic chunker.

The semantic chunker walks sentences, embeds each, and starts a new
chunk when the candidate sentence drifts below the running chunk's
mean similarity. Two safety nets matter as much as the happy path:

* A hard cap (``chunk_max_chars``) so a single runaway "topic" can't
  swallow the whole document.
* A graceful fallback to the markdown chunker when the embedder
  returns ``None`` for any sentence — retrieval recall is more
  important than chunker fidelity when Ollama is down.

We use a deterministic ``_FakeEmbedder`` so the tests don't need a
real model. Two-topic docs use orthogonal vectors so similarity is
exactly zero across the boundary; single-topic docs use a shared
vector so similarity is exactly one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from grimore.cognition.chunker import (
    Chunk,
    DEFAULT_SEMANTIC_THRESHOLD,
    build_candidate_chunks,
    chunk_semantic,
    chunk_semantic_sections,
    split_sentences,
)
from grimore.ingest.adapters.base import ExtractedSection


# ── Test doubles ────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Returns a unit vector keyed on the sentence's first token.

    The mapping is deterministic so the same input always yields the
    same chunk boundaries. ``orthogonal=True`` (the default) means
    different "topics" get fully orthogonal vectors so the cosine
    across them is 0 — well below any reasonable threshold.
    """

    def __init__(self, topic_map: dict[str, list[float]]):
        self.topic_map = topic_map
        self.calls: list[str] = []

    def embed(self, text: str) -> Optional[list[float]]:
        self.calls.append(text)
        # Pick the first matching topic key contained in the sentence.
        for key, vec in self.topic_map.items():
            if key.lower() in text.lower():
                return list(vec)
        # Unknown → default to first topic so untagged sentences stick
        # with whatever came before.
        return list(next(iter(self.topic_map.values())))


class _FailingEmbedder:
    """Returns ``None`` for everything — triggers the markdown fallback."""

    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> Optional[list[float]]:
        self.calls.append(text)
        return None


class _BatchFakeEmbedder(_FakeEmbedder):
    """_FakeEmbedder that also exposes embed_batch, recording each batch call
    so a test can assert the chunker takes the one-round-trip path."""

    def __init__(self, topic_map):
        super().__init__(topic_map)
        self.batch_calls: list[list[str]] = []

    def embed_batch(self, texts):
        self.batch_calls.append(list(texts))
        return [self.embed(t) for t in texts]


# ── split_sentences ─────────────────────────────────────────────────────


class TestSplitSentences:
    def test_basic_split(self):
        out = split_sentences("First sentence. Second sentence. Third one.")
        assert out == ["First sentence.", "Second sentence.", "Third one."]

    def test_abbreviation_does_not_split(self):
        # "Dr. Smith" should not produce a split between "Dr." and "Smith".
        out = split_sentences("Dr. Smith arrived. He sat down.")
        assert out == ["Dr. Smith arrived.", "He sat down."]

    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\n  ") == []

    def test_no_terminal_punctuation(self):
        # A single fragment with no period stays whole.
        assert split_sentences("just a fragment") == ["just a fragment"]


# ── chunk_semantic happy paths ──────────────────────────────────────────


class TestChunkSemantic:
    def test_two_distinct_topics_split_into_two_chunks(self):
        text = (
            "Gothic cathedrals used pointed arches. "
            "Ribbed vaults distributed weight efficiently. "
            "Flying buttresses transferred outward thrust. "
            "Bananas are a tropical fruit. "
            "They grow in large bunches. "
            "Most are yellow when ripe."
        )
        embedder = _FakeEmbedder({
            "gothic": [1.0, 0.0],
            "arches": [1.0, 0.0],
            "vaults": [1.0, 0.0],
            "buttresses": [1.0, 0.0],
            "bananas": [0.0, 1.0],
            "grow": [0.0, 1.0],
            "yellow": [0.0, 1.0],
        })
        chunks = chunk_semantic(text, embedder, max_chars=10_000, threshold=0.5)
        assert len(chunks) == 2
        assert "Gothic" in chunks[0] and "buttresses" in chunks[0]
        assert "Bananas" in chunks[1] and "yellow" in chunks[1]

    def test_uses_embed_batch_in_one_call_when_available(self):
        text = (
            "Gothic cathedrals used pointed arches. "
            "Ribbed vaults distributed weight. "
            "Bananas are a tropical fruit. "
            "Most are yellow when ripe."
        )
        topic_map = {
            "gothic": [1.0, 0.0], "arches": [1.0, 0.0], "vaults": [1.0, 0.0],
            "bananas": [0.0, 1.0], "yellow": [0.0, 1.0],
        }
        plain = chunk_semantic(text, _FakeEmbedder(topic_map),
                               max_chars=10_000, threshold=0.5)
        batched_emb = _BatchFakeEmbedder(topic_map)
        batched = chunk_semantic(text, batched_emb, max_chars=10_000, threshold=0.5)
        # One batch call covering every sentence — not N per-sentence requests.
        assert len(batched_emb.batch_calls) == 1
        assert len(batched_emb.batch_calls[0]) == len(split_sentences(text))
        # Identical boundaries to the per-sentence path.
        assert batched == plain

    def test_single_topic_stays_one_chunk(self):
        text = (
            "WPA2 is a security protocol. "
            "It supersedes WPA. "
            "WPA2 uses AES encryption. "
            "Most modern routers support WPA2."
        )
        embedder = _FakeEmbedder({"wpa": [1.0, 0.0]})
        chunks = chunk_semantic(text, embedder, max_chars=10_000, threshold=0.5)
        assert len(chunks) == 1
        # All four sentences ended up together.
        for needle in ("WPA2 is a security", "supersedes WPA", "AES encryption", "modern routers"):
            assert needle in chunks[0]

    def test_hard_cap_forces_split_even_on_one_topic(self):
        # 20 short same-topic sentences; a tight cap means we *must* see
        # multiple chunks regardless of (perfect) similarity.
        sentences = [f"Cats meow loudly at dawn number {i}." for i in range(20)]
        text = " ".join(sentences)
        embedder = _FakeEmbedder({"cats": [1.0, 0.0]})
        chunks = chunk_semantic(text, embedder, max_chars=80, threshold=0.5)
        assert len(chunks) > 1
        # Every chunk respects the cap (modulo single-sentence overflow,
        # which the sliding-window fallback handles — not exercised here
        # because each sentence < 80 chars).
        for c in chunks:
            assert len(c) <= 80 + 1  # +1 for joining slack

    def test_single_sentence_returns_one_chunk(self):
        text = "Just one sentence with no terminator"
        embedder = _FakeEmbedder({"sentence": [1.0]})
        # split_sentences sees no terminator → one sentence; chunker
        # returns it whole (cap permitting).
        assert chunk_semantic(text, embedder, max_chars=200) == [text]

    def test_empty_text(self):
        assert chunk_semantic("", _FakeEmbedder({"x": [1.0]})) == []
        assert chunk_semantic("   ", _FakeEmbedder({"x": [1.0]})) == []

    def test_failing_embedder_falls_back_to_markdown(self):
        text = (
            "First paragraph here. With two sentences.\n\n"
            "Second paragraph. Also two sentences."
        )
        embedder = _FailingEmbedder()
        chunks = chunk_semantic(text, embedder, max_chars=10_000)
        # Markdown chunker keeps both paragraphs in one ≤10k chunk.
        assert chunks  # non-empty
        # The fake was consulted at least once before falling back.
        assert embedder.calls

    def test_oversized_single_sentence_uses_sliding_window(self):
        # Single sentence that exceeds the cap → flushes via
        # _sliding_window so the caller never sees an >cap chunk.
        long_sent = "Cats " + ("meow " * 200) + "."
        embedder = _FakeEmbedder({"cats": [1.0]})
        chunks = chunk_semantic(long_sent, embedder, max_chars=120, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 120


# ── chunk_semantic_sections (anchor propagation) ────────────────────────


class TestSemanticSections:
    def test_each_section_propagates_its_anchors(self):
        sections = [
            ExtractedSection(text="WPA2 is a security protocol. AES is its cipher.", page=1, heading="Intro"),
            ExtractedSection(text="Gothic arches were pointed. Buttresses were flying.", page=42, heading="Cathedrals"),
        ]
        embedder = _FakeEmbedder({
            "wpa": [1.0, 0.0],
            "aes": [1.0, 0.0],
            "gothic": [0.0, 1.0],
            "buttresses": [0.0, 1.0],
        })
        chunks = chunk_semantic_sections(sections, embedder, max_chars=10_000)
        # Topic similarity within each section is 1.0, so each section
        # stays one chunk. The interesting assertion is anchor fidelity.
        assert len(chunks) == 2
        assert chunks[0].page == 1 and chunks[0].heading == "Intro"
        assert chunks[1].page == 42 and chunks[1].heading == "Cathedrals"

    def test_topic_shift_inside_section_keeps_same_anchors(self):
        # A drift within a single section produces multiple chunks, but
        # all of them must keep that section's anchors — otherwise a
        # citation would point at the wrong page.
        section = ExtractedSection(
            text="Cats meow loudly. Cats hunt at night. Tractors plow fields. Tractors are noisy.",
            page=7, heading="Mixed",
        )
        embedder = _FakeEmbedder({
            "cats": [1.0, 0.0],
            "tractors": [0.0, 1.0],
        })
        chunks = chunk_semantic_sections([section], embedder, max_chars=10_000, threshold=0.5)
        assert len(chunks) == 2
        for c in chunks:
            assert c.page == 7 and c.heading == "Mixed"


# ── build_candidate_chunks dispatch ─────────────────────────────────────


@dataclass
class _FakeNote:
    sections: list = None


@dataclass
class _FakeIngest:
    chunker: str = "markdown"
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    chunk_max_chars: int = 1500


@dataclass
class _FakeConfig:
    ingest: _FakeIngest = None

    def __post_init__(self):
        if self.ingest is None:
            self.ingest = _FakeIngest()


class TestBuildCandidateChunks:
    def test_markdown_default_no_sections(self):
        config = _FakeConfig()
        note = _FakeNote(sections=[])
        embedder = _FakeEmbedder({"x": [1.0]})
        body = "Paragraph one.\n\nParagraph two."
        chunks = build_candidate_chunks(note, body, embedder, config)
        assert all(isinstance(c, Chunk) for c in chunks)
        # Markdown path doesn't touch the embedder.
        assert embedder.calls == []

    def test_markdown_with_sections(self):
        config = _FakeConfig()
        note = _FakeNote(sections=[
            ExtractedSection(text="Page 1 content here.", page=1, heading="A"),
            ExtractedSection(text="Page 2 content here.", page=2, heading="B"),
        ])
        embedder = _FakeEmbedder({"x": [1.0]})
        chunks = build_candidate_chunks(note, "", embedder, config)
        assert {c.page for c in chunks} == {1, 2}
        assert embedder.calls == []  # markdown path is embed-free

    def test_semantic_no_sections_calls_embedder(self):
        config = _FakeConfig(ingest=_FakeIngest(chunker="semantic"))
        note = _FakeNote(sections=[])
        embedder = _FakeEmbedder({"cats": [1.0, 0.0], "trees": [0.0, 1.0]})
        body = "Cats are mammals. Cats sleep. Trees are plants. Trees grow tall."
        chunks = build_candidate_chunks(note, body, embedder, config)
        # Two topics → at least two chunks.
        assert len(chunks) >= 2
        # The semantic path did embed each sentence.
        assert len(embedder.calls) == 4

    def test_semantic_with_sections_calls_embedder_and_keeps_anchors(self):
        config = _FakeConfig(ingest=_FakeIngest(chunker="semantic"))
        note = _FakeNote(sections=[
            ExtractedSection(text="Cats are mammals. They meow.", page=3, heading="Cats"),
        ])
        embedder = _FakeEmbedder({"cats": [1.0]})
        chunks = build_candidate_chunks(note, "", embedder, config)
        assert len(chunks) == 1
        assert chunks[0].page == 3 and chunks[0].heading == "Cats"
        assert len(embedder.calls) == 2  # one per sentence

    def test_unknown_engine_falls_back_to_markdown(self):
        # An unknown chunker value shouldn't crash — defensively
        # treated as the markdown default.
        config = _FakeConfig(ingest=_FakeIngest(chunker="moonbeams"))
        note = _FakeNote(sections=[])
        embedder = _FakeEmbedder({"x": [1.0]})
        chunks = build_candidate_chunks(
            note, "Just a paragraph.", embedder, config,
        )
        assert chunks and embedder.calls == []
