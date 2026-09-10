from unittest.mock import MagicMock

import pytest

from grimore.cognition.privacy import ingest_services
from grimore.utils.config import Config
from grimore.utils.http import _LoopbackPinAdapter


def test_sensitive_local_clients_are_pinned_and_closed(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    config = Config()
    config.cognition.allow_remote = True
    tagger, embedder = MagicMock(), MagicMock()
    with ingest_services(config, tagger, embedder, sensitive=True) as (local_tagger, local_embedder):
        assert local_tagger is not tagger and local_embedder is not embedder
        assert config.cognition.allow_remote is True
        assert local_tagger.config.cognition.allow_remote is False
        sessions = [local_tagger.router.session, local_embedder.session]
        for session in sessions:
            assert isinstance(session.get_adapter("http://localhost"), _LoopbackPinAdapter)
            monkeypatch.setattr(session, "close", MagicMock(wraps=session.close))
    for session in sessions:
        session.close.assert_called_once()


@pytest.mark.parametrize("remote", ["chat", "embeddings"])
def test_sensitive_content_rejects_each_remote_endpoint(monkeypatch, remote):
    config = Config()
    config.cognition.allow_remote = True
    config.cognition.llm_backend = "openai"
    config.cognition.llm_base_url = "https://8.8.8.8" if remote == "chat" else "http://127.0.0.1:8080"
    monkeypatch.setenv("OLLAMA_HOST", "https://8.8.8.8" if remote == "embeddings" else "http://127.0.0.1:11434")
    with pytest.raises(ValueError, match="non-loopback"):
        with ingest_services(config, MagicMock(), MagicMock(), sensitive=True):
            pytest.fail("a remote endpoint must not receive sensitive content")
