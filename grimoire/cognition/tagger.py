import re
from pathlib import Path
from grimoire.cognition.llm_router import LLMRouter
from grimoire.memory.taxonomy import Taxonomy
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Conservative whitelist for tag tokens written into YAML frontmatter.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-/]{0,63}$")
_SUMMARY_MAX = 300
_TAG_LIMIT = 16


def _sanitize_tag(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lstrip("#")
    if not _TAG_RE.match(candidate):
        return None
    return candidate


def _sanitize_summary(raw) -> str:
    if not isinstance(raw, str):
        return ""
    # Drop control chars and collapse whitespace to keep YAML well-formed.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_SUMMARY_MAX]


class Tagger:
    def __init__(self, config, router: LLMRouter, taxonomy: Taxonomy):
        self.config = config
        self.router = router
        self.taxonomy = taxonomy
        self.system_prompt = self._load_prompt()

    def _load_prompt(self):
        prompt_path = Path(__file__).parent / "prompts" / "tagger.txt"
        with open(prompt_path, "r") as f:
            return f.read()

    def tag_note(self, content: str) -> dict:
        """
        Calls LLM to get tags and summary, then reconciles tags with taxonomy.
        """
        logger.info("tagging_note_start")
        
        # We might want to truncate content if it's too long
        prompt = f"Content:\n---\n{content[:4000]}\n---"
        
        result = self.router.complete(prompt, system_prompt=self.system_prompt)
        
        if not result or "tags" not in result:
            logger.error("tagging_failed_invalid_response")
            return {"tags": [], "summary": ""}
            
        raw_tags = result.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = []
        clean_tags = [t for t in (_sanitize_tag(r) for r in raw_tags) if t]
        clean_tags = clean_tags[:_TAG_LIMIT]
        reconciled_tags = self.taxonomy.reconcile(clean_tags)

        return {
            "tags": reconciled_tags,
            "summary": _sanitize_summary(result.get("summary", "")),
        }
