"""
Note tagging, categorisation and summarisation.

Given a chunk of note content, the ``Tagger`` asks the local LLM for:

* a list of flat ``tags`` (reconciled against the vault's controlled vocabulary),
* a short ``summary`` (safe for YAML frontmatter), and
* a single hierarchical ``category`` path (validated against the vault's
  ``CategoryTree`` — unknown categories are dropped).
"""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from grimore.cognition.chunker import chunk_markdown
from grimore.cognition.llm_router import LLMRouter
from grimore.memory.taxonomy import CategoryTree, Taxonomy, VaultTaxonomy
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_TAG_MAX_LEN = 64
_SUMMARY_MAX = 300
_TAG_LIMIT = 16
_TAGGER_INPUT_MAX = 4000
_LONG_NOTE_MAX_PARTS = 12
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Function words that identify Spanish or English prose. With an English
# prompt and example, summaries of Spanish notes came back in English, so
# the tagger names the note's language; small models follow an explicit
# language far better than "the same language as the note". Words shared
# by both languages ("a", "no") are left out.
_FUNCTION_WORDS = {
    "Spanish": frozenset(
        "el la los las de del que y en un una por para con es se su al lo como más "
        "pero sus ya este esta entre cuando muy sin sobre también hasta hay donde "
        "desde son fue o ni".split()),
    "English": frozenset(
        "the of and to in is that for it as was with be by on this are or from at "
        "which but have an they were there been all their can has more would its "
        "also than what when who will into".split()),
}
_PROSE_WORD = re.compile(r"[a-záéíóúüñ]+")


def _note_language(text: str) -> Optional[str]:
    """Name the note's language when its function words clearly say so."""
    words = _PROSE_WORD.findall(text.lower())
    counts = {name: sum(w in vocab for w in words) for name, vocab in _FUNCTION_WORDS.items()}
    best = max(counts, key=counts.__getitem__)
    others = sum(counts.values()) - counts[best]
    if counts[best] < 5 or counts[best] < 2 * others:
        return None
    return best


def _language_rule(language: Optional[str]) -> str:
    """Summary in the note's language; tags in English for every note.

    Tags are one vocabulary across a mixed-language library: a Spanish note
    tagged "ranas" would never meet an English one tagged "frogs" in tag
    filters, distillation or connections. The language is given per field in
    a JSON template: naming only the summary language turned the tags
    Spanish, and one sentence naming both pulled qwen2.5:3b's summaries back
    to English (0/4), while the template held both (Spanish summaries for 4/4
    Spanish notes, English tags).
    """
    lang = language or "the note's language"
    return ('Fill this JSON: {"tags": [3-8 keywords in English], '
            f'"summary": "<one line in {lang}>", "category": "<path from the TAXONOMY>"}}')


@dataclass(frozen=True)
class _AnalysisPart:
    text: str
    section: int
    part: int
    page: int | None = None
    heading: str | None = None


def _analysis_parts(note, content: str) -> list[_AnalysisPart]:
    """Split natural sections first, then cap each model input without overlap."""
    sections = getattr(note, "sections", None) or []
    if sections:
        sources = [
            (section.text, index, section.page, section.heading)
            for index, section in enumerate(sections, 1)
        ]
    else:
        matches = list(_MARKDOWN_HEADING.finditer(content))
        sources = []
        if matches:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sources.append((preamble, 1, None, None))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
                sources.append((content[match.end():end], len(sources) + 1, None,
                                match.group(1).strip()))
        else:
            sources = [(content, 1, None, None)]

    parts: list[_AnalysisPart] = []
    for raw_text, section_number, page, heading in sources:
        safe_text = SecurityGuard.sanitize_prompt(raw_text)
        for part_number, piece in enumerate(
            chunk_markdown(safe_text, max_chars=_TAGGER_INPUT_MAX - 200, overlap=0), 1,
        ):
            parts.append(_AnalysisPart(piece, section_number, part_number, page, heading))
    return parts


def tag_document(tagger, note, clean_content: str) -> dict:
    """Keep the legacy single-call path for short notes and test doubles."""
    # A scanned PDF without OCR (or another textless source) still needs
    # an indexed record and extraction diagnosis, but there is no evidence
    # for an LLM to classify or summarise.
    if not clean_content.strip():
        return {"tags": [], "summary": "", "category": ""}
    if len(clean_content) <= _TAGGER_INPUT_MAX or not isinstance(tagger, Tagger):
        return tagger.tag_note(clean_content)
    return tagger.tag_long_note(note, clean_content)


def _sanitize_tag(raw, taxonomy: Taxonomy) -> str | None:
    """Normalise an LLM-emitted tag and reject if nothing usable remains."""
    if not isinstance(raw, str):
        return None
    candidate = taxonomy.normalize(raw.lstrip("#"))
    if not candidate or len(candidate) > _TAG_MAX_LEN:
        return None
    return candidate


def _sanitize_summary(raw) -> str:
    """Strip control chars, collapse whitespace, truncate to keep YAML clean."""
    if not isinstance(raw, str):
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_SUMMARY_MAX]


def _sanitize_category(raw, tree: CategoryTree) -> Optional[str]:
    """
    Resolve an LLM-emitted category against the CategoryTree and return the
    canonical display path, or None if it doesn't match any known node.
    """
    if not isinstance(raw, str):
        return None
    return tree.resolve(raw)


def _render_category_menu(tree: CategoryTree, max_paths: int = 80) -> str:
    """Render the category tree as a bulleted list the LLM can choose from."""
    paths = tree.paths()
    if not paths:
        return "(no categories configured — leave the field empty)"
    if len(paths) > max_paths:
        paths = paths[:max_paths]
    return "\n".join(f"- {p}" for p in paths)


class Tagger:
    """Tags, categorises and summarises a single note via the local LLM."""

    def __init__(self, config, router: LLMRouter, vault_tax: VaultTaxonomy):
        self.config = config
        self.router = router
        self.vault_tax = vault_tax
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self):
        prompt_path = Path(__file__).parent / "prompts" / "tagger.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _system_prompt(self) -> str:
        """Inject the live category menu into the static prompt template."""
        menu = _render_category_menu(self.vault_tax.categories)
        return self.system_prompt_template.replace("{categories}", menu)

    def tag_note(self, content: str, language: Optional[str] = None) -> dict:
        """
        Returns a dict with ``tags`` (list[str]), ``summary`` (str) and
        ``category`` (str or "" if the LLM's pick wasn't recognised).

        ``language`` names the note's language for the summary; when it is
        omitted it is detected from ``content``.
        """
        logger.info("tagging_note_start")

        if len(content) > _TAGGER_INPUT_MAX:
            logger.info(
                "tagger_input_truncated",
                kept=_TAGGER_INPUT_MAX,
                original=len(content),
            )
        rule = _language_rule(language or _note_language(content))
        prompt = f"Content:\n---\n{content[:_TAGGER_INPUT_MAX]}\n---\n{rule}"
        result = self.router.complete(prompt, system_prompt=self._system_prompt())

        if not result or "tags" not in result:
            logger.error("tagging_failed_invalid_response")
            return {"tags": [], "summary": "", "category": ""}

        raw_tags = result.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = []

        clean_tags = [
            t for t in (_sanitize_tag(r, self.vault_tax.tags) for r in raw_tags) if t
        ]
        clean_tags = clean_tags[:_TAG_LIMIT]
        reconciled_tags = self.vault_tax.tags.reconcile(clean_tags)

        category = _sanitize_category(result.get("category"), self.vault_tax.categories)
        if category is None:
            raw_cat = result.get("category")
            if raw_cat:
                logger.info("category_unrecognised", raw=str(raw_cat)[:80])
            category = ""

        return {
            "tags": reconciled_tags,
            "summary": _sanitize_summary(result.get("summary", "")),
            "category": category,
        }

    def tag_long_note(self, note, content: str) -> dict:
        """Analyze spread-out source parts and retain their provenance.

        The budget is intentionally bounded. When a document exceeds it, the
        coverage record explicitly reports the parts that were not analyzed.
        """
        parts = _analysis_parts(note, content)
        if not parts:
            return {"tags": [], "summary": "", "category": ""}
        # Detected once over the whole document: a part can be all code.
        language = _note_language(content)

        count = min(len(parts), _LONG_NOTE_MAX_PARTS)
        indices = (
            list(range(count)) if count == len(parts) else
            [i * (len(parts) - 1) // (count - 1) for i in range(count)]
        )
        if len(parts) > count:
            logger.warning(
                "tagger_partial_coverage",
                total_parts=len(parts), analyzed_parts=count,
            )
        evidence: list[dict] = []
        tags: list[str] = []
        categories: list[str] = []
        for index in indices:
            part = parts[index]
            source_id = f"s{index + 1}"
            heading = SecurityGuard.sanitize_prompt(part.heading or "")[:160]
            context = f"Heading: {heading}\n" if heading else ""
            result = self.tag_note(context + part.text, language=language)
            successful = bool(result["tags"] or result["summary"] or result["category"])
            evidence.append({
                "id": source_id,
                "section": part.section,
                "part": part.part,
                "page": part.page,
                "heading": part.heading,
                "status": "analyzed" if successful else "failed",
                "summary": result["summary"],
                "tags": result["tags"],
            })
            if successful:
                tags.extend(result["tags"])
                if result["category"]:
                    categories.append(result["category"])

        usable = [item for item in evidence if item["status"] == "analyzed"]
        summary = ""
        summary_sources: list[str] = []
        summaries = [item for item in usable if item["summary"]]
        if len(summaries) == 1:
            summary = summaries[0]["summary"]
            summary_sources = [summaries[0]["id"]]
        elif summaries:
            lines = "\n".join(
                f'{item["id"]}: {SecurityGuard.sanitize_prompt(item["summary"])}'
                for item in summaries
            )
            response = self.router.complete(
                "Combine the section summaries below into one concise summary "
                f"(under 150 characters) in {language or 'the language of the summaries'}. "
                "Use only facts present in them. "
                "Return JSON with keys summary (string) and sources (list of "
                "section IDs used). Do not add unsupported facts.\n\n" + lines,
                system_prompt="You summarize source-labeled document sections. Return only JSON.",
            )
            valid_ids = {item["id"] for item in summaries}
            proposed_sources = response.get("sources") if isinstance(response, dict) else None
            if isinstance(proposed_sources, list) and proposed_sources and all(
                isinstance(source, str) and source in valid_ids for source in proposed_sources
            ):
                summary = _sanitize_summary(response.get("summary"))
                if summary:
                    summary_sources = list(dict.fromkeys(proposed_sources))
            if not summary:
                summary = _sanitize_summary("; ".join(item["summary"] for item in summaries))
                summary_sources = [item["id"] for item in summaries]

        category = Counter(categories).most_common(1)[0][0] if categories else ""
        return {
            "tags": self.vault_tax.tags.reconcile(tags)[:_TAG_LIMIT],
            "summary": summary,
            "category": category,
            "section_analysis": {
                "total_parts": len(parts),
                "analyzed_parts": len(indices),
                "unanalyzed_parts": len(parts) - len(indices),
                "successful_parts": len(usable),
                "failed_parts": len(indices) - len(usable),
                "summary_sources": summary_sources,
                "sources": evidence,
            },
        }
