import os
import json
from typing import Optional, Any
from grimoire.utils.http import build_session
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

logger = get_logger(__name__)

class LLMRouter:
    def __init__(self, config):
        self.config = config
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=config.cognition.allow_remote
        )
        self.session = build_session()

    def complete(self, prompt: str, system_prompt: str = "", model_override: str = None, json_format: bool = True) -> Any:
        """
        Routes the completion request to the appropriate backend.
        Default is local Ollama.
        """
        # For now, only Ollama is implemented as per Phase 2 local-first focus
        model = model_override or self.config.cognition.model_llm_local
        
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            if json_format:
                payload["format"] = "json"
            
            response = self.session.post(url, json=payload, timeout=60) # Increased timeout for reasoning
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get("response", "{}")
            
            if json_format:
                return json.loads(raw_response)
            return raw_response
            
        except Exception as e:
            logger.error("llm_call_failed", model=model, error=str(e))
            return None
