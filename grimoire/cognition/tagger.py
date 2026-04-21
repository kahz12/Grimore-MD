"""
Note tagging and summarization logic.
This module uses an LLM to generate relevant tags and a concise summary for
each note, ensuring tags adhere to the project's taxonomy.
"""
import re
from pathlib import Path
from grimoire.cognition.llm_router import LLMRouter
from grimoire.memory.taxonomy import Taxonomy
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for sanitization and limits
_TAG_MAX_LEN = 64
_SUMMARY_MAX = 300
_TAG_LIMIT = 16


def _sanitize_tag(raw, taxonomy: Taxonomy) -> str | None:
    """
    Normalise an LLM-emitted tag and reject it if nothing usable remains.
    Delegates to taxonomy.normalize to ensure consistent formatting.
    """
    if not isinstance(raw, str):
        return None
    candidate = taxonomy.normalize(raw.lstrip("#"))
    if not candidate or len(candidate) > _TAG_MAX_LEN:
        return None
    return candidate


def _sanitize_summary(raw) -> str:
    """
    Cleans up the LLM-generated summary to ensure it's safe for YAML frontmatter.
    Removes control characters and collapses extra whitespace.
    """
    if not isinstance(raw, str):
        return ""
    # Drop control chars and collapse whitespace to keep YAML well-formed.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_SUMMARY_MAX]


class Tagger:
    """
    Coordinates the tagging and summarization process for a single note.
    """
    def __init__(self, config, router: LLMRouter, taxonomy: Taxonomy):
        self.config = config
        self.router = router
        self.taxonomy = taxonomy
        self.system_prompt = self._load_prompt()

    def _load_prompt(self):
        """Loads the tagger system prompt from the prompts directory."""
        prompt_path = Path(__file__).parent / "prompts" / "tagger.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def tag_note(self, content: str) -> dict:
        """
        Calls the LLM to get tags and summary, then reconciles results with the taxonomy.
        Returns a dict with 'tags' (list) and 'summary' (str).
        """
        logger.info("tagging_note_start")
        
        # Truncate content to avoid exceeding context window limits
        prompt = f"Content:\n---\n{content[:4000]}\n---"
        
        result = self.router.complete(prompt, system_prompt=self.system_prompt)
        
        if not result or "tags" not in result:
            logger.error("tagging_failed_invalid_response")
            return {"tags": [], "summary": ""}
            
        raw_tags = result.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = []
            
        # Sanitize and limit the number of tags
        clean_tags = [
            t for t in (_sanitize_tag(r, self.taxonomy) for r in raw_tags) if t
        ]
        clean_tags = clean_tags[:_TAG_LIMIT]
        
        # Reconcile tags against existing vault taxonomy (merging synonyms, etc.)
        reconciled_tags = self.taxonomy.reconcile(clean_tags)

        return {
            "tags": reconciled_tags,
            "summary": _sanitize_summary(result.get("summary", "")),
        }
