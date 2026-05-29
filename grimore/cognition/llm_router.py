"""LLM router with circuit-breaker + pluggable backends.

The router itself is backend-agnostic: it owns the circuit-breaker,
the JSON-extraction fallback for non-strict servers, and the public
API that the rest of the codebase has used since v2.x.
The actual wire format lives in
:mod:`grimore.cognition.llm_backends.ollama` and
:mod:`grimore.cognition.llm_backends.openai`.

Backwards compat note: existing call sites
(``oracle.router.complete(...)``, ``tagger.router.complete(...)``,
``router.complete_streaming(...)``, ``router.list_installed_models()``)
keep working unchanged. Tests that mock ``LLMRouter`` continue to mock
the same public surface.
"""
import json
import re
import time
from typing import Any, Iterator, Optional

from grimore.cognition.llm_backends import LLMBackend, build_backend
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
    """Dispatch completion calls to the configured backend.

    Owns:

    * **Circuit breaker** — after ``_FAILURE_THRESHOLD`` back-to-back
      failures, calls short-circuit for ``_COOLDOWN_SECONDS`` so we
      don't hammer a dead backend.
    * **JSON extraction** — backends return raw strings; this class
      runs :func:`_extract_json_object` when ``json_format=True`` so
      servers that don't honour ``response_format`` still mostly work.

    The router holds the backend as ``self.backend``. The legacy
    attribute ``self.ollama_host`` is preserved (pointing at the
    backend's host) so any code that read it for logging keeps
    working — but no production code should reach into it.
    """
    # Circuit breaker: after N back-to-back failures, short-circuit calls
    # for COOLDOWN_SECONDS so we don't hammer a dead backend.
    _FAILURE_THRESHOLD = 5
    _COOLDOWN_SECONDS = 120

    def __init__(self, config, *, backend: Optional[LLMBackend] = None):
        self.config = config
        self.backend: LLMBackend = backend or build_backend(config)
        # Legacy alias kept for code that surfaced the host in logs or
        # error messages. Points at whatever URL the backend is using
        # — Ollama's host or the OpenAI base URL — so the field stays
        # informative without breaking the contract.
        self.ollama_host = getattr(
            self.backend, "host",
            getattr(self.backend, "base_url", ""),
        )
        self.session = getattr(self.backend, "session", None)
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
        """One-shot completion via the active backend.

        Returns:
            * the raw answer string when ``json_format`` is False,
            * a parsed ``dict`` when ``json_format`` is True and JSON
              extraction succeeds,
            * ``None`` on circuit-open, network failure, or unparseable
              JSON.
        """
        if self._circuit_open():
            logger.warning("llm_skipped_circuit_open")
            return None

        raw = self.backend.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model_override=model_override,
            json_format=json_format,
        )
        if raw is None:
            self._record_failure()
            return None

        if json_format:
            parsed = _extract_json_object(raw if isinstance(raw, str) else "")
            if parsed is None:
                logger.warning(
                    "llm_json_parse_failed",
                    raw=SecurityGuard.redact_for_log(raw if isinstance(raw, str) else ""),
                )
                self._record_failure()
                return None
            self._record_success()
            return parsed

        self._record_success()
        return raw

    def list_installed_models(self) -> list[dict]:
        """Pass-through to the backend's model listing.

        Each entry is ``{"name": str, "size": int}`` in whatever order
        the backend reports. Empty list on any failure so callers can
        handle "backend down" and "no models" the same way.
        """
        return self.backend.list_installed_models()

    def complete_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield response chunks as the backend emits them.

        Errors degrade silently: an empty iterator is returned and the
        circuit-breaker counter advances, so callers can fall back to
        the non-streaming path without special handling.
        """
        if self._circuit_open():
            logger.warning("llm_skipped_circuit_open")
            return

        got_anything = False
        try:
            for piece in self.backend.complete_streaming(
                prompt=prompt,
                system_prompt=system_prompt,
                model_override=model_override,
            ):
                if piece:
                    got_anything = True
                    yield piece
        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
            self._record_failure()
            return

        if got_anything:
            self._record_success()
        else:
            self._record_failure()
