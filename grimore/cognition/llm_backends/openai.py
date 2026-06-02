"""OpenAI-compatible backend (``POST /v1/chat/completions``).

Targets anything that speaks the OpenAI HTTP shape — llama.cpp server,
vLLM, LM Studio, OpenRouter, Together, OpenAI proper. The backend is
chosen via ``[cognition].llm_backend = "openai"`` in ``grimore.toml``;
``[cognition].llm_base_url`` is the server root (no trailing slash, no
``/v1`` suffix — the backend appends it).

Auth: pulls a bearer token from the env var named in
``[cognition].llm_api_key_env`` (default ``GRIMORE_LLM_API_KEY``). When
the var is unset the Authorization header is omitted — fine for
loopback llama.cpp, required for hosted providers.

JSON mode and streaming both use the official OpenAI primitives:
``response_format={"type": "json_object"}`` and ``stream=true`` with
SSE-framed ``data:`` lines. Servers that don't support
``response_format`` will likely return text that still parses through
the router's bracket-balanced fallback.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# OpenAI's SSE framing prepends "data: " to every line; the terminator
# line is the literal "data: [DONE]".
_SSE_PREFIX = "data: "
_SSE_DONE = "[DONE]"


class OpenAICompatibleBackend:
    """Backend for any ``/v1/chat/completions`` server.

    Validation order at construction:

    1. Resolve the base URL (``[cognition].llm_base_url`` or
       ``OPENAI_BASE_URL`` env var). Default to
       ``http://localhost:8080`` so a fresh llama.cpp-server install
       works without further config.
    2. Run :func:`SecurityGuard.validate_llm_host` so non-loopback URLs
       require ``[cognition].allow_remote`` AND ``https`` — same gate
       Ollama goes through.
    3. Read the bearer token lazily (re-read on every call) so a
       long-running process picks up a rotated key without restart.
    """

    name = "openai"

    def __init__(self, config):
        self.config = config
        base = (
            getattr(config.cognition, "llm_base_url", None)
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8080"
        )
        allow_remote = config.cognition.allow_remote
        # SecurityGuard returns the URL untouched on success; mirror the
        # Ollama backend in storing the validated form.
        self.base_url = SecurityGuard.validate_llm_host(
            base, allow_remote=allow_remote
        ).rstrip("/")
        self._api_key_env = getattr(
            config.cognition, "llm_api_key_env", "GRIMORE_LLM_API_KEY"
        )
        # Pin loopback HTTP to the validated address (audit I1: DNS-rebinding).
        self.session = build_session(
            pins=SecurityGuard.loopback_pins(base, allow_remote=allow_remote)
        )

    def _headers(self) -> dict:
        """Build per-call headers, picking up the bearer token if set."""
        headers = {"Content-Type": "application/json"}
        token = os.getenv(self._api_key_env, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _messages(prompt: str, system_prompt: str) -> list[dict]:
        """Translate the router's ``(system, prompt)`` pair into the chat
        messages list. Empty system prompts are omitted so we don't send
        a stray empty ``system`` turn (some servers count those as part
        of the model's context budget)."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
        json_format: bool = True,
    ) -> Any:
        model = model_override or self.config.cognition.model_llm_local
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload: dict = {
                "model": model,
                "messages": self._messages(prompt, system_prompt),
                "stream": False,
            }
            if json_format:
                payload["response_format"] = {"type": "json_object"}

            response = self.session.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.config.cognition.request_timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            return message.get("content", "")
        except Exception as e:
            logger.error("llm_call_failed", backend="openai", model=model, error=str(e))
            return None

    def complete_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        model = model_override or self.config.cognition.model_llm_local
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": self._messages(prompt, system_prompt),
            "stream": True,
        }

        try:
            with self.session.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.config.cognition.stream_timeout_s,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if not raw.startswith(_SSE_PREFIX):
                        # Some servers emit raw JSON without the ``data:``
                        # prefix; tolerate it. The done sentinel is the
                        # canonical SSE shape, so missing prefix isn't fatal.
                        line = raw
                    else:
                        line = raw[len(_SSE_PREFIX):]
                    if line.strip() == _SSE_DONE:
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
                    if choices[0].get("finish_reason"):
                        break
        except Exception as e:
            logger.error("llm_stream_failed", backend="openai", model=model, error=str(e))
            return

    def list_installed_models(self) -> list[dict]:
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(
                url, headers=self._headers(), timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            entries = data.get("data") or []
            return [
                {"name": m.get("id", ""), "size": 0}
                for m in entries
                if m.get("id")
            ]
        except Exception as e:
            logger.error("openai_list_failed", error=str(e))
            return []
