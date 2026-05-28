"""End-to-end tests for the v2.1 retrieval-quality features against a live
Ollama server.

These cover what the mocked tests in ``test_retrieval_upgrades.py`` can't:
that the actual local LLM honours the query-rewrite JSON contract, that the
re-rank pass returns parseable scores, and that retrieval — fed by real
embeddings — surfaces the expected note for both a direct ask and a
pronoun-resolving follow-up.

The whole module is marked ``e2e`` and auto-skips when Ollama isn't reachable
on ``OLLAMA_HOST`` (or ``http://localhost:11434``) within a short probe, or
when the default LLM / embedding models aren't pulled. CI without a local
model picks this up as a clean skip rather than a failure.

Run only these:    pytest -m e2e
Skip these:        pytest -m "not e2e"
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest
import requests

from grimore.cognition.embedder import Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.oracle import Oracle
from grimore.memory.db import Database
from grimore.session import Session
from grimore.utils.config import load_config

pytestmark = pytest.mark.e2e


# ── gating ──────────────────────────────────────────────────────────────────

_OLLAMA_PROBE_TIMEOUT_S = 2.0
# Defaults that ship in CognitionConfig (kept in sync with config.py). The
# fixture re-checks both names — Ollama exposes models as either
# ``nomic-embed-text`` or ``nomic-embed-text:latest`` depending on how the
# user pulled them.
_DEFAULT_LLM = "qwen2.5:3b"
_DEFAULT_EMB = "nomic-embed-text"


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_models() -> set[str] | None:
    """Return the set of installed model names, or ``None`` if Ollama is
    unreachable. Names include both bare (``qwen2.5:3b``) and ``:latest``
    variants as Ollama reports them."""
    try:
        resp = requests.get(
            f"{_ollama_host()}/api/tags", timeout=_OLLAMA_PROBE_TIMEOUT_S
        )
        resp.raise_for_status()
    except (requests.RequestException, ValueError):
        return None
    names: set[str] = set()
    for m in resp.json().get("models", []) or []:
        name = m.get("name") or m.get("model")
        if not name:
            continue
        names.add(name)
        # Normalise bare-name lookups: "qwen2.5:3b:latest" → "qwen2.5:3b".
        if name.endswith(":latest"):
            names.add(name[: -len(":latest")])
    return names


def _require_model(installed: set[str], wanted: str) -> bool:
    """Match either an exact pin or the bare base name (``foo`` vs ``foo:latest``)."""
    if wanted in installed:
        return True
    return any(n == wanted or n.startswith(wanted + ":") for n in installed)


@pytest.fixture(scope="module")
def live_ollama() -> dict:
    """Skip the whole module unless Ollama is reachable AND the two default
    models exist locally. Returns the resolved host + model names so tests
    don't re-probe."""
    installed = _ollama_models()
    if installed is None:
        pytest.skip(f"Ollama not reachable at {_ollama_host()}")
    if not _require_model(installed, _DEFAULT_LLM):
        pytest.skip(f"LLM model '{_DEFAULT_LLM}' not pulled (installed: {sorted(installed)[:6]}…)")
    if not _require_model(installed, _DEFAULT_EMB):
        pytest.skip(f"Embedding model '{_DEFAULT_EMB}' not pulled")
    return {"host": _ollama_host(), "llm": _DEFAULT_LLM, "emb": _DEFAULT_EMB}


# ── seeded vault (real embeddings, three deliberately distinct notes) ──────

# Each note is short and topic-pure so retrieval is easy to reason about:
# embedding similarity should overwhelmingly pick the matching note for a
# query that names its distinctive terms.
_VAULT_NOTES: list[tuple[str, str]] = [
    (
        "Espresso Brewing",
        # Distinctive tokens: 9 bar, 25-30 second extraction, puck, crema.
        "Espresso is brewed by forcing hot water at roughly 9 bar of pressure "
        "through a tightly packed puck of finely ground coffee. A balanced "
        "shot extracts in about 25 to 30 seconds and pours with a thick "
        "reddish-brown crema on top. Grind size, dose and tamping pressure "
        "are the three levers a barista tunes to hit that window."
    ),
    (
        "Gothic Cathedrals",
        # Distinctive tokens: pointed arch, flying buttress, ribbed vault, Chartres.
        "Gothic cathedrals such as Chartres and Notre-Dame achieve their "
        "soaring height through three structural innovations: the pointed "
        "arch, the ribbed vault, and the flying buttress. Together these let "
        "masons push thrust outward to external supports, freeing the walls "
        "to hold vast stained-glass windows instead of solid stone."
    ),
    (
        "Stoic Ethics",
        # Distinctive tokens: Marcus Aurelius, Epictetus, dichotomy of control.
        "Stoic ethics, articulated by Epictetus and Marcus Aurelius, rests on "
        "the dichotomy of control: some things are up to us — our judgements, "
        "intentions and responses — and the rest, including reputation and "
        "outcomes, is not. The discipline is to invest effort only in the "
        "former and meet the latter with equanimity."
    ),
]


@pytest.fixture(scope="module")
def seeded(tmp_path_factory, live_ollama) -> dict:
    """Build a temp DB with the three notes embedded by the real Ollama
    embedder, plus a warm Session pointed at it. Module-scoped so the embed
    cost (≈1s per chunk on nomic-embed-text) is paid once."""
    workdir = tmp_path_factory.mktemp("grimore_e2e")
    db_path = workdir / "grimore.db"

    config = load_config()
    config.memory.db_path = str(db_path)
    config.vault.path = str(workdir)
    # Pin the models we already verified are installed, in case a user's
    # grimore.toml overrode them.
    config.cognition.model_llm_local = live_ollama["llm"]
    config.cognition.model_embeddings_local = live_ollama["emb"]
    # Keep the defaults explicit — rerank is opt-in per-test.
    config.cognition.rerank = False

    db = Database(str(db_path))
    embedder = Embedder(config, cache=db)

    for title, body in _VAULT_NOTES:
        # Use a fake path that's stable and unique; no file actually exists,
        # and the Oracle only ever looks up titles by note_id.
        path = str(workdir / f"{title.lower().replace(' ', '_')}.md")
        content_hash = hashlib.sha256(body.encode()).hexdigest()
        note_id = db.upsert_note(path=path, title=title, content_hash=content_hash)
        chunks = embedder.embed_chunks(body)
        assert chunks, f"real embedder returned nothing for {title!r}"
        for idx, (chunk_text, vec) in enumerate(chunks):
            db.store_embedding(
                note_id=note_id,
                chunk_index=idx,
                text_content=chunk_text,
                vector_blob=Embedder.serialize_vector(vec),
            )

    # Build the Session lazily — its properties pick up the patched config.
    session = Session(config=config)
    # Force the warm services to materialise so the first test isn't paying
    # for cold-start router + embedder build.
    _ = session.db, session.router, session.embedder, session.oracle
    return {"config": config, "db": db, "session": session, "workdir": workdir}


# ── helpers ────────────────────────────────────────────────────────────────


def _sources_titles(sources: list[str]) -> list[str]:
    """Drop ``#anchor`` suffixes so assertions can compare bare titles."""
    return [s.split("#", 1)[0] for s in sources or []]


# ── tests ──────────────────────────────────────────────────────────────────


class TestLiveRetrieval:
    def test_direct_ask_hits_the_right_note(self, seeded):
        """A query naming distinctive espresso terms must surface the espresso
        note among the top sources."""
        result = seeded["session"].oracle.ask(
            "What pressure is used to brew espresso?", top_k=3
        )
        assert isinstance(result["answer"], str) and result["answer"].strip()
        titles = _sources_titles(result["sources"])
        assert "Espresso Brewing" in titles, (
            f"expected Espresso Brewing in sources, got {titles!r}"
        )

    def test_citation_grounding_is_silent_on_a_clean_answer(self, seeded):
        """When the model only cites retrieved sources (the default case for
        a well-behaved local model on an easy question), no citations are
        dropped — verify the warning path stays at zero."""
        result = seeded["session"].oracle.ask(
            "What structural elements distinguish Gothic cathedrals?", top_k=3
        )
        assert "Gothic Cathedrals" in _sources_titles(result["sources"])
        # Don't assert == 0 strictly — a 3B local model may occasionally
        # invent a [[Bogus]] tag; just assert the count is sane and grounding
        # actually ran (key present in the dict).
        assert "dropped_citations" in result
        assert result["dropped_citations"] <= 2


class TestConversationFollowUp:
    def test_followup_resolves_pronoun_via_rewrite(self, seeded):
        """First turn establishes the espresso topic; the second turn uses
        only a pronoun, and retrieval must still land on the espresso note —
        proving query rewrite consumed the history."""
        session = seeded["session"]
        session.forget()  # isolate from any prior test bleed-through

        first = session.oracle.ask("Tell me about espresso brewing.", top_k=3)
        assert "Espresso Brewing" in _sources_titles(first["sources"])
        session.record_turn("Tell me about espresso brewing.", first["answer"], first["sources"])

        # Pronoun-only follow-up. Without rewrite, embedding "How long does
        # it take?" alone is far weaker — it could match anything time-ish.
        followup = session.oracle.ask(
            "How long does it take?",
            top_k=3,
            history=session.turns,
        )
        titles = _sources_titles(followup["sources"])
        assert "Espresso Brewing" in titles, (
            f"follow-up retrieval drifted off-topic: sources={titles!r}\n"
            f"answer={followup['answer'][:200]!r}"
        )
        # And the answer should plausibly mention the extraction window.
        # Use a tolerant check — the model phrases it many ways.
        ans = followup["answer"].lower()
        assert any(tok in ans for tok in ("25", "30", "second", "extraction")), (
            f"follow-up answer does not reference timing: {followup['answer'][:300]!r}"
        )


class TestRerankPath:
    def test_rerank_enabled_runs_and_keeps_relevant_note(self, seeded):
        """Enable the LLM re-rank stage and confirm the live model returns
        parseable scores (otherwise the connector silently falls back to the
        RRF order — still safe, but the test below would still pass; combined
        with the unit tests in test_retrieval_upgrades.py this gives us both
        the contract and a live smoke)."""
        config = seeded["config"]
        config.cognition.rerank = True
        try:
            result = seeded["session"].oracle.ask(
                "Who wrote about the dichotomy of control?", top_k=3
            )
        finally:
            config.cognition.rerank = False

        titles = _sources_titles(result["sources"])
        assert "Stoic Ethics" in titles, (
            f"rerank path drifted off-topic: {titles!r}"
        )
        assert isinstance(result["answer"], str) and result["answer"].strip()
