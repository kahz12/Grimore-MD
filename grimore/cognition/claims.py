"""
ClaimExtractor — atomic factual claim extraction.

First stage of the Black Mirror pipeline: turn each note into a small
set of atomic, declarative claims that can be embedded and pairwise-
checked for contradictions. Long documents are split at source sections and
queried within a configurable budget. Source offsets are recovered within
each section when the claim is quoted verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import frontmatter

from grimore.cognition.llm_router import LLMRouter
from grimore.ingest.adapters.base import ExtractedSection
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)


# Defaults for each model call. The number of calls is bounded separately.
CLAIM_EXTRACTION_MAX_CLAIMS = 10
CLAIM_EXTRACTION_MAX_CHARS = 8000
CLAIM_EXTRACTION_MAX_SECTIONS = 8
_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


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
)


@dataclass
class ExtractedClaim:
    text: str
    char_start: Optional[int]
    char_end: Optional[int]


@dataclass(frozen=True)
class ClaimSectionRef:
    section: int
    part: int
    page: Optional[int]
    heading: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    reason: str = ""


@dataclass
class ClaimExtractionReport:
    claims: list[ExtractedClaim]
    analyzed_sections: list[ClaimSectionRef]
    unanalyzed_sections: list[ClaimSectionRef]
    total_sections: int

    @property
    def failed(self) -> bool:
        return any(ref.reason == "llm_failed" for ref in self.unanalyzed_sections)


@dataclass(frozen=True)
class _SourceSection:
    text: str
    start: Optional[int]
    page: Optional[int]
    heading: Optional[str]
    number: int


def _source_sections(body: str, stripped: str,
                     sections: list[ExtractedSection] | None) -> list[_SourceSection]:
    """Map adapter sections (or Markdown headings) back to body offsets."""
    base = body.find(stripped)
    if base < 0:
        base = 0
    if sections:
        out: list[_SourceSection] = []
        cursor = 0
        for number, section in enumerate(sections, 1):
            start = stripped.find(section.text, cursor)
            if start < 0:
                start = stripped.find(section.text)
            if start >= 0:
                cursor = start + len(section.text)
            out.append(_SourceSection(
                section.text, base + start if start >= 0 else None,
                section.page, section.heading, number,
            ))
        return out

    headings = list(_HEADING.finditer(stripped))
    if not headings:
        return [_SourceSection(stripped, base, None, None, 1)]
    out = []
    if headings[0].start() > 0:
        out.append(_SourceSection(stripped[:headings[0].start()], base, None, None, 1))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(stripped)
        out.append(_SourceSection(
            stripped[match.start():end], base + match.start(), None,
            match.group().lstrip("# ").strip(), len(out) + 1,
        ))
    return out


def _split_sections(sources: list[_SourceSection], max_chars: int) -> list[tuple[str, ClaimSectionRef]]:
    """Create non-overlapping, source-addressable prompt parts."""
    parts: list[tuple[str, ClaimSectionRef]] = []
    for source in sources:
        cursor = 0
        part_number = 0
        while cursor < len(source.text):
            while cursor < len(source.text) and source.text[cursor].isspace():
                cursor += 1
            if cursor >= len(source.text):
                break
            end = min(cursor + max_chars, len(source.text))
            if end < len(source.text):
                boundary = source.text.rfind("\n\n", cursor + max_chars // 2, end)
                if boundary < 0:
                    boundary = source.text.rfind(" ", cursor + max_chars // 2, end)
                if boundary > cursor:
                    end = boundary
            raw = source.text[cursor:end]
            text = raw.rstrip()
            if len(SecurityGuard.sanitize_prompt(text)) > max_chars:
                # Role-marker neutralization inserts characters. In the
                # pathological case of marker-dense text, shrink this exact
                # source span before sending it to the model.
                end = cursor + max_chars * 5 // 6
                text = source.text[cursor:end].rstrip()
            if text:
                part_number += 1
                start = source.start + cursor if source.start is not None else None
                stop = start + len(text) if start is not None else None
                parts.append((text, ClaimSectionRef(
                    source.number, part_number, source.page, source.heading,
                    start, stop,
                )))
            cursor = end
    return parts


class ClaimExtractor:
    """LLM-driven claim extraction with offset-recovery."""

    def __init__(
        self, router: LLMRouter, *,
        max_chars: int = CLAIM_EXTRACTION_MAX_CHARS,
        max_sections: int = CLAIM_EXTRACTION_MAX_SECTIONS,
        max_claims_per_section: int = CLAIM_EXTRACTION_MAX_CLAIMS,
    ):
        for name, value, lower, upper in (
            ("max_chars", max_chars, 500, 32_000),
            ("max_sections", max_sections, 1, 64),
            ("max_claims_per_section", max_claims_per_section, 1, 30),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        self.router = router
        self.max_chars = max_chars
        self.max_sections = max_sections
        self.max_claims_per_section = max_claims_per_section

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
        return self.extract_with_coverage(body).claims

    def extract_with_coverage(
        self, body: str, *, sections: list[ExtractedSection] | None = None,
    ) -> ClaimExtractionReport:
        """Extract by section and report every omitted or failed source part."""
        empty = ClaimExtractionReport([], [], [], 0)
        if not body or not body.strip():
            return empty

        try:
            stripped = frontmatter.loads(body).content
        except Exception:
            stripped = body
        if not stripped.strip():
            return empty
        parts = _split_sections(_source_sections(body, stripped, sections), self.max_chars)
        if not parts:
            return empty
        count = min(len(parts), self.max_sections)
        indices = (
            list(range(count)) if count == len(parts) else
            [i * (len(parts) - 1) // (count - 1) for i in range(count)]
            if count > 1 else [0]
        )
        selected = set(indices)
        unanalyzed = [
            ClaimSectionRef(
                ref.section, ref.part, ref.page, ref.heading,
                ref.char_start, ref.char_end, "budget",
            )
            for index, (_text, ref) in enumerate(parts) if index not in selected
        ]

        out: list[ExtractedClaim] = []
        analyzed: list[ClaimSectionRef] = []
        seen_texts: set[str] = set()
        for index in indices:
            part_text, ref = parts[index]
            source_context = ""
            if ref.heading:
                heading = SecurityGuard.sanitize_prompt(ref.heading)[:160]
                source_context = f"SOURCE HEADING (context only): {heading}\n"
            elif ref.page is not None:
                source_context = f"SOURCE PAGE: {ref.page}\n"
            prompt = (
                source_context +
                "NOTE CONTENT (between markers):\n"
                "--- BEGIN NOTE ---\n"
                f"{SecurityGuard.sanitize_prompt(part_text)}\n"
                "--- END NOTE ---"
            )
            result = self.router.complete(
                prompt,
                system_prompt=_SYSTEM + f"Return at most {self.max_claims_per_section} claims.",
                json_format=True,
            )
            raw_claims = result.get("claims") if isinstance(result, dict) else None
            if not isinstance(raw_claims, list):
                unanalyzed.append(ClaimSectionRef(
                    ref.section, ref.part, ref.page, ref.heading,
                    ref.char_start, ref.char_end, "llm_failed",
                ))
                continue
            analyzed.append(ref)
            for entry in raw_claims[:self.max_claims_per_section]:
                text = self._normalize_claim_text(entry)
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                cs, ce = self._locate(body, text, ref.char_start, ref.char_end)
                out.append(ExtractedClaim(text=text, char_start=cs, char_end=ce))
        return ClaimExtractionReport(out, analyzed, unanalyzed, len(parts))

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
    def _locate(
        body: str, claim_text: str, start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        """Best-effort offset recovery via substring search.

        Returns ``(None, None)`` when the LLM paraphrased beyond a
        substring match — Mirror's ``show`` falls back to displaying
        the claim text alone when offsets are absent.
        """
        if not body or not claim_text:
            return None, None
        if start is None or end is None:
            return None, None
        idx = body.find(claim_text, start, end)
        if idx == -1:
            # Try a case-insensitive match before giving up.
            lower_idx = body.lower().find(claim_text.lower(), start, end)
            if lower_idx == -1:
                return None, None
            return lower_idx, lower_idx + len(claim_text)
        return idx, idx + len(claim_text)
