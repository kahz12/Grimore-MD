"""LLM backend protocol + factory.

Decouples :class:`grimore.cognition.llm_router.LLMRouter` from Ollama's
wire format. Backends implement a tiny Protocol that mirrors the
router's existing public surface (so the rest of the codebase doesn't
change shape) while the router itself stays the single home of the
circuit-breaker and the JSON extraction fallback.

Two backends ship today:

* :class:`grimore.cognition.llm_backends.ollama.OllamaBackend` — the
  v2.x default. Talks to ``OLLAMA_HOST/api/{generate,tags,embeddings}``.
* :class:`grimore.cognition.llm_backends.openai.OpenAICompatibleBackend`
  — anything that speaks ``POST /v1/chat/completions``: llama.cpp
  server, vLLM, LM Studio, OpenRouter, OpenAI proper.

The Protocol is intentionally minimal: backends return strings (or
parsed dicts for the JSON path) and yield strings on stream, so
existing callers — Oracle, tagger, eval, reranker — stay unchanged.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """The contract every chat backend must implement.

    Designed to mirror the public surface of the v2.x LLMRouter so
    callers like ``Oracle.ask`` don't change. The router wraps each
    method with the circuit-breaker; backends themselves don't have to
    know the breaker exists.
    """

    name: str
    """Human-readable backend label used in logs and the ``status`` panel."""

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
        json_format: bool = True,
    ) -> Any:
        """Run a one-shot completion.

        Returns:
            * the raw answer string when ``json_format`` is False,
            * a parsed ``dict`` when ``json_format`` is True and parsing
              succeeds,
            * ``None`` on any failure (network, parse, model error).
        """
        ...

    def complete_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield answer fragments as the backend produces them.

        An empty iterator on failure is the documented "soft fail"
        signal — the router's circuit-breaker counter advances and the
        caller can fall back to the non-streaming path.
        """
        ...

    def list_installed_models(self) -> list[dict]:
        """Return the models the backend can serve.

        Each entry is ``{"name": str, "size": int}`` so the shell's
        ``/models`` command renders the same way regardless of backend.
        Backends that don't expose a size return ``0`` for that field.
        """
        ...


def build_backend(config) -> LLMBackend:
    """Build the backend selected by ``config.cognition.llm_backend``.

    Defaults to Ollama for backwards compatibility — a vault upgrading
    from v2.x sees zero behaviour change until they flip the knob.

    Unknown backend names degrade to Ollama with a warning rather than
    raising: a typo in ``grimore.toml`` shouldn't break the CLI before
    the user can fix it.
    """
    from grimore.utils.logger import get_logger

    logger = get_logger(__name__)
    name = getattr(config.cognition, "llm_backend", "ollama") or "ollama"
    if name == "openai":
        from grimore.cognition.llm_backends.openai import OpenAICompatibleBackend
        return OpenAICompatibleBackend(config)
    if name != "ollama":
        logger.warning("llm_backend_unknown", name=name, fallback="ollama")
    from grimore.cognition.llm_backends.ollama import OllamaBackend
    return OllamaBackend(config)
