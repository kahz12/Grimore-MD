"""
Note tagging, categorisation and summarisation.

Given a chunk of note content, the ``Tagger`` asks the local LLM for:

* a list of flat ``tags`` (reconciled against the vault's controlled vocabulary),
* a short ``summary`` (safe for YAML frontmatter), and
* a single hierarchical ``category`` path (validated against the vault's
  ``CategoryTree`` — unknown categories are dropped).
"""
import re
from pathlib import Path
from typing import Optional

from grimoire.cognition.llm_router import LLMRouter
from grimoire.memory.taxonomy import CategoryTree, Taxonomy, VaultTaxonomy
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

_TAG_MAX_LEN = 64
_SUMMARY_MAX = 300
_TAG_LIMIT = 16


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

    def tag_note(self, content: str) -> dict:
        """
        Returns a dict with ``tags`` (list[str]), ``summary`` (str) and
        ``category`` (str or "" if the LLM's pick wasn't recognised).
        """
        logger.info("tagging_note_start")

        prompt = f"Content:\n---\n{content[:4000]}\n---"
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
