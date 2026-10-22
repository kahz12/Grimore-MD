from grimore.utils.config import Config
from grimore.utils.connectivity import cognition_connectivity


def _config(**overrides):
    config = Config()
    for key, value in overrides.items():
        setattr(config.cognition, key, value)
    return config


def test_default_mode_is_fully_local():
    result = cognition_connectivity(_config(), environ={})

    assert result == {
        "mode": "local",
        "remote_allowed": False,
        "chat": {"backend": "ollama", "mode": "local"},
        "embeddings": {"backend": "ollama", "mode": "local"},
    }


def test_openai_remote_chat_and_local_embeddings_is_mixed():
    config = _config(
        allow_remote=True,
        llm_backend="openai",
        llm_base_url="https://models.example/v1",
    )

    result = cognition_connectivity(
        config, environ={"OLLAMA_HOST": "http://127.0.0.1:11434"},
    )

    assert result["mode"] == "mixed"
    assert result["chat"] == {"backend": "openai", "mode": "remote"}
    assert result["embeddings"] == {"backend": "ollama", "mode": "local"}


def test_remote_ollama_applies_to_chat_and_embeddings():
    result = cognition_connectivity(
        _config(allow_remote=True),
        environ={"OLLAMA_HOST": "https://ollama.example"},
    )

    assert result["mode"] == "remote"
    assert result["chat"]["mode"] == "remote"
    assert result["embeddings"]["mode"] == "remote"


def test_status_classification_never_resolves_dns(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("status classification must not resolve DNS")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    result = cognition_connectivity(
        _config(
            allow_remote=True,
            llm_backend="openai",
            llm_base_url="https://models.example/v1",
        ),
        environ={},
    )

    assert result["mode"] == "mixed"
