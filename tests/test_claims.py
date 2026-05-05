"""Tests for ClaimExtractor.

We exercise text→claim conversion against a queued fake LLM. Offset
recovery (mapping the LLM's claim text back to a span in the source
note) is the trickiest part — covered with verbatim, case-shifted,
and paraphrased inputs.
"""
from __future__ import annotations

import pytest

from grimoire.cognition.claims import ClaimExtractor, ExtractedClaim


class _QueuedRouter:
    """Returns each queued payload from ``complete`` in order."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete(self, prompt, system_prompt="", json_format=True, model_override=None):
        self.calls.append({"prompt": prompt, "system": system_prompt})
        return self.payloads.pop(0) if self.payloads else None


# ── early-exit paths ─────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_body_returns_empty(self):
        router = _QueuedRouter([])
        assert ClaimExtractor(router).extract("") == []
        assert router.calls == []  # no LLM call

    def test_whitespace_body_returns_empty(self):
        router = _QueuedRouter([])
        assert ClaimExtractor(router).extract("   \n\t  \n") == []
        assert router.calls == []

    def test_none_payload_returns_empty(self):
        router = _QueuedRouter([None])
        assert ClaimExtractor(router).extract("Some content here.") == []

    def test_non_dict_payload_returns_empty(self):
        router = _QueuedRouter(["not a dict"])
        assert ClaimExtractor(router).extract("Some content here.") == []

    def test_missing_claims_key_returns_empty(self):
        router = _QueuedRouter([{"oops": []}])
        assert ClaimExtractor(router).extract("Some content here.") == []

    def test_claims_not_a_list_returns_empty(self):
        router = _QueuedRouter([{"claims": "this should be a list"}])
        assert ClaimExtractor(router).extract("Some content here.") == []


# ── claim normalization ──────────────────────────────────────────────────


class TestNormalization:
    def test_bare_strings_are_accepted(self):
        router = _QueuedRouter([{"claims": ["The sky is blue.", "Water boils at 100C."]}])
        out = ClaimExtractor(router).extract("Body text.")
        assert [c.text for c in out] == ["The sky is blue.", "Water boils at 100C."]

    def test_dict_with_text_key_works(self):
        router = _QueuedRouter([{"claims": [{"text": "X is Y."}, {"text": "A is B."}]}])
        out = ClaimExtractor(router).extract("Body.")
        assert [c.text for c in out] == ["X is Y.", "A is B."]

    def test_dedupes_repeated_claims(self):
        router = _QueuedRouter([{"claims": ["Same.", "Same.", {"text": "Same."}]}])
        out = ClaimExtractor(router).extract("Body.")
        assert [c.text for c in out] == ["Same."]

    def test_skips_blank_and_oversized_claims(self):
        oversized = "x" * 600
        router = _QueuedRouter([{"claims": ["", "   ", oversized, "Real claim."]}])
        out = ClaimExtractor(router).extract("Body.")
        assert [c.text for c in out] == ["Real claim."]

    def test_caps_at_max_per_note(self):
        many = [f"Claim {i}." for i in range(50)]
        router = _QueuedRouter([{"claims": many}])
        out = ClaimExtractor(router).extract("Body.")
        # Implementation cap is 10.
        assert len(out) == 10


# ── offset recovery ─────────────────────────────────────────────────────


class TestOffsetRecovery:
    def test_exact_substring_match(self):
        body = "Intro. The sky is blue today. Outro."
        router = _QueuedRouter([{"claims": ["The sky is blue today."]}])
        out = ClaimExtractor(router).extract(body)
        assert len(out) == 1
        c = out[0]
        assert c.char_start is not None
        assert body[c.char_start:c.char_end] == "The sky is blue today."

    def test_case_insensitive_fallback(self):
        body = "Intro. The Sky Is Blue. Outro."
        router = _QueuedRouter([{"claims": ["the sky is blue."]}])
        out = ClaimExtractor(router).extract(body)
        c = out[0]
        # Offsets cover the original-case substring.
        assert c.char_start is not None
        assert body[c.char_start:c.char_end].lower() == "the sky is blue."

    def test_paraphrased_claim_returns_none_offsets(self):
        body = "Intro. The cat sat on the mat. Outro."
        # LLM rewrote the claim — substring search will fail.
        router = _QueuedRouter([{"claims": ["A cat was on a mat at the time."]}])
        out = ClaimExtractor(router).extract(body)
        assert len(out) == 1
        assert out[0].char_start is None
        assert out[0].char_end is None


# ── frontmatter handling ────────────────────────────────────────────────


class TestFrontmatter:
    def test_frontmatter_stripped_before_llm_sees_body(self):
        body = (
            "---\n"
            "title: My Note\n"
            "tags: [a, b]\n"
            "---\n"
            "\n"
            "The actual claim is that water boils at 100 degrees Celsius.\n"
        )
        router = _QueuedRouter([{"claims": ["Water boils at 100 degrees Celsius."]}])
        ClaimExtractor(router).extract(body)
        # Confirm the LLM never saw the YAML frontmatter.
        assert "tags:" not in router.calls[0]["prompt"]
        assert "title:" not in router.calls[0]["prompt"]
        assert "water boils" in router.calls[0]["prompt"]

    def test_offsets_are_relative_to_full_body(self):
        body = (
            "---\n"
            "title: T\n"
            "---\n"
            "Water boils at 100 degrees Celsius.\n"
        )
        router = _QueuedRouter([{"claims": ["Water boils at 100 degrees Celsius."]}])
        out = ClaimExtractor(router).extract(body)
        c = out[0]
        assert c.char_start is not None
        # The match must point at the post-frontmatter body, not into it.
        assert body[c.char_start:c.char_end] == "Water boils at 100 degrees Celsius."
        assert c.char_start > body.find("---\n", 4)  # past the closing ---
