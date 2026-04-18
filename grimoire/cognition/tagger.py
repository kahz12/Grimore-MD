from pathlib import Path
from grimoire.cognition.llm_router import LLMRouter
from grimoire.memory.taxonomy import Taxonomy
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

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
        reconciled_tags = self.taxonomy.reconcile(raw_tags)
        
        return {
            "tags": reconciled_tags,
            "summary": result.get("summary", "")
        }
