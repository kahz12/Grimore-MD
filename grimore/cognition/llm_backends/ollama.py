"""Ollama backend — the v2.x default.

Talks the legacy ``/api/generate`` + ``/api/tags`` shape via the
``OLLAMA_HOST`` environment variable. Behaviour is exactly the v2.1
LLMRouter — the body was lifted into this class verbatim so the
extraction is a refactor and not a rewrite.

JSON parsing failures and stream errors don't raise here — they return
``None`` or yield nothing so the router's circuit breaker can advance.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)


class OllamaBackend:
    """Backend for a locally-running Ollama instance.

    Reads the host from ``OLLAMA_HOST`` (default ``http://localhost:11434``)
    once at construction and validates it through
    :func:`SecurityGuard.validate_llm_host` — same gate the v2.x router
    used, so loopback-only enforcement is preserved.
    """

    name = "ollama"

    def __init__(self, config):
        self.config = config
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=config.cognition.allow_remote
        )
        self.session = build_session()

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
        json_format: bool = True,
    ) -> Any:
        model = model_override or self.config.cognition.model_llm_local
        try:
            url = f"{self.host}/api/generate"
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
            return result.get("response", "")
        except Exception as e:
            logger.error("llm_call_failed", backend="ollama", model=model, error=str(e))
            return None

    def complete_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        model = model_override or self.config.cognition.model_llm_local
        url = f"{self.host}/api/generate"
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
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = chunk.get("response", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except Exception as e:
            logger.error("llm_stream_failed", backend="ollama", model=model, error=str(e))
            return

    def list_installed_models(self) -> list[dict]:
        try:
            url = f"{self.host}/api/tags"
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
