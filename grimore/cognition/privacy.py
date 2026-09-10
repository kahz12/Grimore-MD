"""Per-document local-only services without changing shared configuration."""
from contextlib import contextmanager
from dataclasses import replace

from grimore.cognition.embedder import Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.tagger import Tagger


@contextmanager
def ingest_services(config, tagger, embedder, *, sensitive: bool):
    """Sensitive documents must use validated, pinned loopback clients.

    A remote-only configuration raises before any content is sent. Rebuilding
    with allow_remote=False also closes the DNS check/use gap for HTTP hosts;
    simply validating the existing remote-enabled client would not pin it.
    """
    if not sensitive or not config.cognition.allow_remote:
        yield tagger, embedder
        return
    local_config = replace(config, cognition=replace(config.cognition, allow_remote=False))
    router = local_embedder = None
    try:
        router = LLMRouter(local_config)
        local_embedder = Embedder(local_config, cache=embedder.cache)
        yield Tagger(local_config, router, tagger.vault_tax), local_embedder
    finally:
        for client in (router, local_embedder):
            session = getattr(client, "session", None)
            if session is not None:
                session.close()
