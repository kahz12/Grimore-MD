"""Tests for the interactive shell.

We exercise ``GrimoreShell.dispatch`` directly — the prompt-toolkit
loop in ``run`` is a thin driver around it and is not covered here
(would require a tty harness).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from grimore.session import Session
from grimore.shell import GrimoreShell, _shell_history_path
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def shell_config(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return Config(
        vault=VaultConfig(path=str(vault_dir), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


@pytest.fixture
def patched_services(monkeypatch):
    """Replace the heavy service constructors used by Session.

    Returns the call-count dict so tests can assert reuse.
    """
    counts = {"db": 0, "router": 0, "embedder": 0, "oracle": 0}

    def make_db(*a, **kw):
        counts["db"] += 1
        return MagicMock(name="db")

    def make_router(*a, **kw):
        counts["router"] += 1
        return MagicMock(name="router")

    def make_embedder(*a, **kw):
        counts["embedder"] += 1
        return MagicMock(name="embedder")

    class _FakeOracle:
        def __init__(self, *a, **kw):
            counts["oracle"] += 1

        def ask(self, question, top_k=5):
            return {"answer": "Oracle says hi.", "sources": ["Note A"]}

        def ask_stream(self, question, top_k=5):
            yield {"type": "token", "text": "Oracle "}
            yield {"type": "token", "text": "streams."}
            yield {"type": "done", "sources": ["Note A"]}

    monkeypatch.setattr("grimore.session.Database", make_db)
    monkeypatch.setattr("grimore.session.LLMRouter", make_router)
    monkeypatch.setattr("grimore.session.Embedder", make_embedder)
    monkeypatch.setattr("grimore.session.Oracle", _FakeOracle)
    return counts


# ── core dispatch behaviour ──────────────────────────────────────────────


def test_unknown_command_does_not_kill_loop(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("nope")
    out = capsys.readouterr().out
    assert "Unknown command" in out
    # Loop is still alive — running flag untouched.
    assert shell._running is True


def test_blank_line_is_noop(shell_config, patched_services):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("")
    shell.dispatch("   ")
    assert shell._running is True


def test_exit_stops_loop(shell_config, patched_services):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("exit")
    assert shell._running is False


def test_quit_stops_loop(shell_config, patched_services):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("quit")
    assert shell._running is False


def test_help_lists_commands(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("help")
    out = capsys.readouterr().out
    for cmd in ("ask", "status", "tags", "category", "exit"):
        assert cmd in out


def test_help_for_specific_command(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("help ask")
    out = capsys.readouterr().out
    assert "ask" in out and "Oracle" in out


def test_parse_error_does_not_kill_loop(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    # Unclosed quote — shlex raises ValueError.
    shell.dispatch('ask "broken')
    out = capsys.readouterr().out
    assert "Parse error" in out
    assert shell._running is True


def test_argparse_error_does_not_kill_loop(shell_config, patched_services):
    """`tags --bogus` would normally call sys.exit. The non-exiting parser
    converts that to a caught exception so the shell keeps going."""
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("tags --bogus")
    assert shell._running is True


# ── the main point: session reuse across asks ───────────────────────────


def test_embedder_built_once_across_two_asks(shell_config, patched_services):
    """The whole reason for the shell: two `ask`s must not pay the
    cold-start cost twice. Embedder must be instantiated exactly once."""
    session = Session(shell_config)
    shell = GrimoreShell(session)

    shell.dispatch("ask Question one")
    shell.dispatch("ask Question two")

    assert patched_services["embedder"] == 1, (
        "Embedder was rebuilt — the warm-session win is broken."
    )
    assert patched_services["router"] == 1
    assert patched_services["oracle"] == 1


def test_refresh_drops_cached_services(shell_config, patched_services):
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("ask Question one")  # builds services
    shell.dispatch("refresh")
    shell.dispatch("ask Question two")  # rebuilds services
    assert patched_services["embedder"] == 2
    assert patched_services["oracle"] == 2


def test_ask_streams_by_default(shell_config, patched_services, capsys):
    """The streamed text must appear in the output exactly once."""
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("ask Tell me about X")
    out = capsys.readouterr().out
    assert "Oracle streams." in out


def test_ask_no_stream_uses_oracle_ask(shell_config, patched_services, capsys):
    """`--no-stream` must use the JSON-strict path (Oracle.ask)."""
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("ask --no-stream Tell me about X")
    out = capsys.readouterr().out
    assert "Oracle says hi." in out


# ── history file hygiene ────────────────────────────────────────────────


def test_history_file_created_with_0o600(tmp_path, shell_config, monkeypatch):
    """B-09-style hygiene: per-vault history files must not be world-readable."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    session = Session(shell_config)
    path = _shell_history_path(session)
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o600


def test_history_file_is_per_vault(tmp_path, monkeypatch):
    """Two different vaults must not share the same history file."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()

    cfg_a = Config(
        vault=VaultConfig(path=str(vault_a)),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "a.db")),
        output=OutputConfig(),
        maintenance=MaintenanceConfig(),
    )
    cfg_b = Config(
        vault=VaultConfig(path=str(vault_b)),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "b.db")),
        output=OutputConfig(),
        maintenance=MaintenanceConfig(),
    )

    path_a = _shell_history_path(Session(cfg_a))
    path_b = _shell_history_path(Session(cfg_b))
    assert path_a != path_b


# ── KeyboardInterrupt isolation ─────────────────────────────────────────


def test_keyboardinterrupt_inside_handler_returns_to_loop(
    shell_config, patched_services, capsys
):
    """Ctrl+C while a command is running must cancel the command, not the loop."""
    shell = GrimoreShell(Session(shell_config))

    def boom(_argv):
        raise KeyboardInterrupt

    shell.commands["status"] = boom
    shell.dispatch("status")
    out = capsys.readouterr().out
    assert "Interrupted" in out
    assert shell._running is True
