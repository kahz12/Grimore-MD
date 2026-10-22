"""Describe whether cognition traffic stays on this device.

This module deliberately performs no DNS lookup and opens no connection.  It
is used by status surfaces, including the web UI, so inspecting privacy mode
must never create the network activity it is trying to describe.
"""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse


def _is_explicit_loopback(url: str) -> bool:
    """Return whether *url* names loopback without consulting DNS."""
    host = urlparse(url).hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _service_mode(url: str, *, allow_remote: bool) -> str:
    # With the remote gate closed, SecurityGuard will reject anything that
    # does not resolve entirely to loopback.  Reporting this as local is both
    # accurate for successful calls and avoids a DNS lookup in status paths.
    if not allow_remote:
        return "local"
    return "local" if _is_explicit_loopback(url) else "remote"


def cognition_connectivity(config, *, environ: dict[str, str] | None = None) -> dict:
    """Return a serialisable local/remote summary for chat and embeddings.

    The result describes the effective endpoints selected by the same config
    and environment-variable precedence as the runtime backends.  Endpoint
    URLs and credentials are intentionally omitted from the public shape.
    """
    env = os.environ if environ is None else environ
    cognition = config.cognition
    allow_remote = bool(getattr(cognition, "allow_remote", False))
    backend = (getattr(cognition, "llm_backend", "ollama") or "ollama").lower()
    if backend == "openai":
        chat_url = (
            getattr(cognition, "llm_base_url", None)
            or env.get("OPENAI_BASE_URL")
            or "http://localhost:8080"
        )
        chat_backend = "openai"
    else:
        # Unknown names fall back to Ollama in build_backend().
        chat_url = env.get("OLLAMA_HOST", "http://localhost:11434")
        chat_backend = "ollama"

    embedding_url = env.get("OLLAMA_HOST", "http://localhost:11434")
    chat_mode = _service_mode(chat_url, allow_remote=allow_remote)
    embedding_mode = _service_mode(embedding_url, allow_remote=allow_remote)
    if chat_mode == embedding_mode:
        overall = chat_mode
    else:
        overall = "mixed"

    return {
        "mode": overall,
        "remote_allowed": allow_remote,
        "chat": {"backend": chat_backend, "mode": chat_mode},
        "embeddings": {"backend": "ollama", "mode": embedding_mode},
    }
