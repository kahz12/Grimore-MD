import json
import os
import re
import time
from typing import Any, Optional

from grimoire.utils.http import build_session
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

logger = get_logger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort parse of a JSON object embedded in LLM output.
    Strategies (in order): direct parse, fenced code block, bracket-balanced
    substring starting at the first '{'. Returns None if all fail.
    """
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_FENCE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
                break
    return None


class LLMRouter:
    # Circuit breaker: after N back-to-back failures, short-circuit calls
    # for COOLDOWN_SECONDS so we don't hammer a dead Ollama.
    _FAILURE_THRESHOLD = 5
    _COOLDOWN_SECONDS = 120

    def __init__(self, config):
        self.config = config
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=config.cognition.allow_remote
        )
        self.session = build_session()
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _circuit_open(self) -> bool:
        return time.monotonic() < self._open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._FAILURE_THRESHOLD and not self._circuit_open():
            self._open_until = time.monotonic() + self._COOLDOWN_SECONDS
            logger.warning(
                "llm_circuit_open",
                cooldown_s=self._COOLDOWN_SECONDS,
                failures=self._consecutive_failures,
            )

    def _record_success(self) -> None:
        if self._consecutive_failures or self._open_until:
            logger.info("llm_circuit_closed")
        self._consecutive_failures = 0
        self._open_until = 0.0

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: str = None,
        json_format: bool = True,
    ) -> Any:
        """
        Route the completion request to the local Ollama backend.
        When the circuit is open, returns None without contacting the model.
        """
        if self._circuit_open():
            logger.warning("llm_skipped_circuit_open")
            return None

        model = model_override or self.config.cognition.model_llm_local

        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
            }
            if json_format:
                payload["format"] = "json"

            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "")

            if json_format:
                parsed = _extract_json_object(raw_response)
                if parsed is None:
                    logger.warning("llm_json_parse_failed", raw=raw_response[:200])
                    self._record_failure()
                    return None
                self._record_success()
                return parsed

            self._record_success()
            return raw_response

        except Exception as e:
            logger.error("llm_call_failed", model=model, error=str(e))
            self._record_failure()
            return None
