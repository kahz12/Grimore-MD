"""
ClaimExtractor — atomic factual claim extraction.

Phase 0 of the Black Mirror pipeline: turn each note into a small set
of atomic, declarative claims that can be embedded and pairwise-checked
for contradictions. The LLM produces just the text (paraphrased into a
single declarative sentence is fine); we derive the source offsets via
a substring search when possible, falling back to ``None`` when the
LLM rephrased too aggressively to map back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import frontmatter

from grimoire.cognition.llm_router import LLMRouter
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


# How many claims we ask the LLM to produce per note. Bounded so a long
# note doesn't blow the contradiction-check budget.
_MAX_CLAIMS_PER_NOTE = 10
# Hard ceiling on note bytes sent to the LLM in one call. The extractor
# is a one-shot per note (no chunking), so a too-long note gets truncated
# rather than producing inconsistent partial results across chunks.
_MAX_NOTE_CHARS = 8000


_SYSTEM = (
    "You extract atomic factual claims from a personal note.\n"
    "Output ONLY a JSON object of the form:\n"
    '  {"claims": [{"text": "<one declarative sentence>"}, ...]}\n'
    "Each claim MUST be:\n"
    "  • a single declarative statement that can be true or false\n"
    "  • self-contained (no pronouns referring back to other claims)\n"
    "  • copied verbatim from the note when possible — paraphrase only\n"
    "    to remove pronouns or fill in implicit context\n"
    "Skip questions, opinions, todos, hedges, and metadata.\n"
    f"Return at most {_MAX_CLAIMS_PER_NOTE} claims."
)


@dataclass
class ExtractedClaim:
    text: str
    char_start: Optional[int]
    char_end: Optional[int]


class ClaimExtractor:
    """LLM-driven claim extraction with offset-recovery."""

    def __init__(self, router: LLMRouter):
        self.router = router

    def extract(self, body: str) -> list[ExtractedClaim]:
        """Extract claims from ``body``. Frontmatter-aware.

        ``body`` is the raw file contents (with or without YAML
        frontmatter). The extractor strips the frontmatter so the LLM
        sees the actual prose, but the offset search uses the *full*
        body so callers can map a claim back to a span in the original
        file.

        Returns ``[]`` on:
          * empty/whitespace input
          * LLM circuit open
          * malformed payload (no ``claims`` array)
          * LLM call failure (LLMRouter returns None)
        """
        if not body or not body.strip():
            return []

        try:
            stripped = frontmatter.loads(body).content
        except Exception:
            stripped = body
        prompt_input = stripped[:_MAX_NOTE_CHARS].strip()
        if not prompt_input:
            return []

        prompt = (
            "NOTE CONTENT (between markers):\n"
            "--- BEGIN NOTE ---\n"
            f"{prompt_input}\n"
            "--- END NOTE ---"
        )
        result = self.router.complete(prompt, system_prompt=_SYSTEM, json_format=True)
        if not isinstance(result, dict):
            return []
        raw_claims = result.get("claims")
        if not isinstance(raw_claims, list):
            return []

        out: list[ExtractedClaim] = []
        seen_texts: set[str] = set()
        for entry in raw_claims[:_MAX_CLAIMS_PER_NOTE]:
            text = self._normalize_claim_text(entry)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            cs, ce = self._locate(body, text)
            out.append(ExtractedClaim(text=text, char_start=cs, char_end=ce))
        return out

    @staticmethod
    def _normalize_claim_text(entry: object) -> Optional[str]:
        """Pull ``text`` from a payload row.

        Tolerates the LLM emitting ``{"text": "..."}`` *or* a bare string
        — both have shown up in test runs against small models.
        """
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            value = entry.get("text")
            if not isinstance(value, str):
                return None
            text = value
        else:
            return None
        text = text.strip()
        if not text or len(text) > 500:
            return None
        return text

    @staticmethod
    def _locate(body: str, claim_text: str) -> tuple[Optional[int], Optional[int]]:
        """Best-effort offset recovery via substring search.

        Returns ``(None, None)`` when the LLM paraphrased beyond a
        substring match — Mirror's ``show`` falls back to displaying
        the claim text alone when offsets are absent.
        """
        if not body or not claim_text:
            return None, None
        idx = body.find(claim_text)
        if idx == -1:
            # Try a case-insensitive match before giving up.
            lower_idx = body.lower().find(claim_text.lower())
            if lower_idx == -1:
                return None, None
            return lower_idx, lower_idx + len(claim_text)
        return idx, idx + len(claim_text)
