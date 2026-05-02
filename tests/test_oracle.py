"""B-07 regression: Oracle.ask caps the LLM context at _ORACLE_CONTEXT_MAX_CHARS."""
from unittest.mock import MagicMock

from grimoire.cognition.oracle import Oracle, _ORACLE_CONTEXT_MAX_CHARS


def _make_oracle():
    o = Oracle.__new__(Oracle)
    o.db = MagicMock()
    o.router = MagicMock()
    o.embedder = MagicMock()
    o.connector = MagicMock()
    o.system_prompt_template = "TEMPLATE: {context}"
    o.config = MagicMock()
    o.config.cognition.hybrid_search = False
    o.config.cognition.rrf_k = 60
    o.db.fts_available = False
    o.embedder.embed.return_value = [0.0] * 16
    o.db.get_note_title.side_effect = lambda nid: f"Note {nid}"
    return o


def _capture_system_prompt(o):
    seen = {}

    def capture(prompt, system_prompt):
        seen["system"] = system_prompt
        return {"answer": "OK"}

    o.router.complete.side_effect = capture
    return seen


def test_context_capped_when_too_many_sources():
    o = _make_oracle()
    seen = _capture_system_prompt(o)
    # 50 candidates × 500 chars ≈ 26 KB: well over the 16 KB cap.
    similar = [
        {"note_id": i, "text": "x" * 500, "score": 1.0 - i * 0.01}
        for i in range(50)
    ]
    o.connector.find_similar_notes.return_value = similar

    result = o.ask("hi", top_k=50)

    ctx = seen["system"].split("TEMPLATE: ", 1)[1]
    assert len(ctx) <= _ORACLE_CONTEXT_MAX_CHARS
    # Some sources must have been dropped.
    assert len(result["sources"]) < 50
    # And at least one was accepted.
    assert len(result["sources"]) > 0


def test_small_context_keeps_every_source():
    o = _make_oracle()
    seen = _capture_system_prompt(o)
    similar = [
        {"note_id": i, "text": "tiny body", "score": 0.9}
        for i in range(3)
    ]
    o.connector.find_similar_notes.return_value = similar

    result = o.ask("hi", top_k=3)

    ctx = seen["system"].split("TEMPLATE: ", 1)[1]
    assert len(ctx) <= _ORACLE_CONTEXT_MAX_CHARS
    assert len(result["sources"]) == 3


def test_oversized_first_source_is_skipped_then_smaller_ones_kept():
    """A single huge top-ranked source must not starve the rest of the context."""
    o = _make_oracle()
    seen = _capture_system_prompt(o)
    huge = "y" * (_ORACLE_CONTEXT_MAX_CHARS + 5_000)
    similar = [
        {"note_id": 0, "text": huge, "score": 0.99},  # too big on its own
        {"note_id": 1, "text": "small body 1", "score": 0.90},
        {"note_id": 2, "text": "small body 2", "score": 0.85},
    ]
    o.connector.find_similar_notes.return_value = similar

    result = o.ask("hi")

    # The huge one is skipped; the two smaller ones are accepted.
    assert "Note 0" not in result["sources"]
    assert "Note 1" in result["sources"]
    assert "Note 2" in result["sources"]
    ctx = seen["system"].split("TEMPLATE: ", 1)[1]
    assert len(ctx) <= _ORACLE_CONTEXT_MAX_CHARS
