"""
Local LLM Routing and JSON Extraction.
This module handles communication with the Ollama API, implements a circuit breaker
pattern for robustness, and provides robust JSON extraction from LLM responses.
"""
import json
import os
import re
import time
from typing import Any, Iterator, Optional

from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Regex to find JSON blocks inside Markdown code fences
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort parse of a JSON object embedded in LLM output.
    Strategies (in order):
    1. Direct parse of the entire text.
    2. Extraction from Markdown fenced code blocks (```json ... ```).
    3. Bracket-balanced substring starting at the first '{'.
    
    Returns None if all strategies fail.
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

    # Fallback: find the first '{' and try to find a balanced '}'
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
    """
    Routes completion requests to the local Ollama backend.
    Implements a circuit breaker to prevent hammering the service if it's down.
    """
    # Circuit breaker: after N back-to-back failures, short-circuit calls
    # for COOLDOWN_SECONDS so we don't hammer a dead Ollama.
    _FAILURE_THRESHOLD = 5
    _COOLDOWN_SECONDS = 120

    def __init__(self, config):
        self.config = config
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Ensure the host is valid and safe (especially if allow_remote is False)
        self.ollama_host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=config.cognition.allow_remote
        )
        self.session = build_session()
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _circuit_open(self) -> bool:
        """Returns True if the circuit breaker is currently open (cooldown active)."""
        return time.monotonic() < self._open_until

    def _record_failure(self) -> None:
        """Records a failure and potentially opens the circuit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._FAILURE_THRESHOLD and not self._circuit_open():
            self._open_until = time.monotonic() + self._COOLDOWN_SECONDS
            logger.warning(
                "llm_circuit_open",
                cooldown_s=self._COOLDOWN_SECONDS,
                failures=self._consecutive_failures,
            )

    def _record_success(self) -> None:
        """Resets the failure counter upon a successful call."""
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
        Sends a completion request to Ollama.
        If json_format=True, attempts to parse and return a dictionary.
        Returns None if the circuit is open or if the request fails.
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

            response = self.session.post(
                url, json=payload, timeout=self.config.cognition.request_timeout_s
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "")

            if json_format:
                parsed = _extract_json_object(raw_response)
                if parsed is None:
                    logger.warning(
                        "llm_json_parse_failed",
                        raw=SecurityGuard.redact_for_log(raw_response),
                    )
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

    def list_installed_models(self) -> list[dict]:
        """Return the models installed in the local Ollama (`/api/tags`).

        Each entry is ``{"name": str, "size": int}`` in the order Ollama
        reports them. Empty list on any failure (logged) so callers can
        handle "Ollama down" and "vault has no models" the same way.
        """
        try:
            url = f"{self.ollama_host}/api/tags"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", []) or []
            return [
                {"name": m.get("name", ""), "size": int(m.get("size", 0) or 0)}
                for m in models
                if m.get("name")
            ]
        except Exception as e:
            logger.error("ollama_list_failed", error=str(e))
            return []

    def complete_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield response chunks as Ollama emits them (NDJSON stream).

        Used by the interactive shell so the user sees the answer being
        typed instead of waiting for the full payload. Always plain text
        (no json_format) — the Oracle uses this only for the final answer
        rendering, never for structured calls like the tagger.

        Errors degrade silently: an empty iterator is returned and the
        circuit-breaker counter advances, so callers can fall back to
        the non-streaming path without special handling.
        """
        if self._circuit_open():
            logger.warning("llm_skipped_circuit_open")
            return

        model = model_override or self.config.cognition.model_llm_local
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
        }

        try:
            with self.session.post(
                url,
                json=payload,
                timeout=self.config.cognition.stream_timeout_s,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                got_anything = False
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = chunk.get("response", "")
                    if piece:
                        got_anything = True
                        yield piece
                    if chunk.get("done"):
                        break
            if got_anything:
                self._record_success()
            else:
                self._record_failure()
        except Exception as e:
            logger.error("llm_stream_failed", model=model, error=str(e))
            self._record_failure()
