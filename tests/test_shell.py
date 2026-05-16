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


def test_unknown_slash_does_not_kill_loop(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/scna")
    out = capsys.readouterr().out
    assert "Unknown slash command" in out
    # difflib should propose /scan as the closest match.
    assert "/scan" in out
    # Loop is still alive — running flag untouched.
    assert shell._running is True


def test_freeform_text_routes_to_ask(shell_config, patched_services, capsys):
    """A non-slash, non-command line is treated as a question to the Oracle."""
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("what does my vault say about jung?")
    out = capsys.readouterr().out
    # Streaming Oracle output is rendered (the fake Oracle yields tokens).
    assert "Oracle streams." in out
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


# ── models command ──────────────────────────────────────────────────────


def _stub_models(shell, models):
    """Pin `router.list_installed_models()` to a fixed list for the test."""
    shell.session.router.list_installed_models = lambda: list(models)


def test_models_no_args_shows_current_and_list(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    _stub_models(shell, [
        {"name": "qwen2.5:3b", "size": 2_000_000_000},
        {"name": "nomic-embed-text", "size": 274_000_000},
    ])
    shell.dispatch("models")
    out = capsys.readouterr().out
    assert "qwen2.5:3b" in out and "nomic-embed-text" in out
    assert "Installed Ollama models" in out


def test_models_chat_by_name_updates_config(
    shell_config, patched_services, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)  # isolate from real grimore.toml
    session = Session(shell_config)
    shell = GrimoreShell(session)
    _stub_models(shell, [
        {"name": "qwen2.5:3b", "size": 1},
        {"name": "ministral-3:14b", "size": 2},
    ])
    shell.dispatch("models chat ministral-3:14b")
    assert session.config.cognition.model_llm_local == "ministral-3:14b"


def test_models_chat_by_index_updates_config(
    shell_config, patched_services, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session = Session(shell_config)
    shell = GrimoreShell(session)
    _stub_models(shell, [
        {"name": "qwen2.5:3b", "size": 1},
        {"name": "ministral-3:14b", "size": 2},
    ])
    shell.dispatch("models chat 2")
    assert session.config.cognition.model_llm_local == "ministral-3:14b"


def test_models_embed_resets_embedder_and_oracle(
    shell_config, patched_services, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session = Session(shell_config)
    shell = GrimoreShell(session)
    # Build embedder + oracle once via an ask.
    shell.dispatch("ask Question one")
    assert patched_services["embedder"] == 1
    assert patched_services["oracle"] == 1

    _stub_models(shell, [
        {"name": "nomic-embed-text", "size": 1},
        {"name": "mxbai-embed-large", "size": 2},
    ])
    shell.dispatch("models embed mxbai-embed-large")
    assert session.config.cognition.model_embeddings_local == "mxbai-embed-large"

    # Next ask must rebuild embedder + oracle (router is unaffected).
    shell.dispatch("ask Question two")
    assert patched_services["embedder"] == 2
    assert patched_services["oracle"] == 2
    assert patched_services["router"] == 1


def test_models_unknown_name_does_not_update(
    shell_config, patched_services, tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    session = Session(shell_config)
    original = session.config.cognition.model_llm_local
    shell = GrimoreShell(session)
    _stub_models(shell, [{"name": "qwen2.5:3b", "size": 1}])
    shell.dispatch("models chat does-not-exist")
    out = capsys.readouterr().out
    assert "no such model" in out
    assert session.config.cognition.model_llm_local == original


def test_models_chat_persists_to_toml(
    shell_config, patched_services, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "grimore.toml").write_text(
        "[cognition]\n"
        'model_llm_local = "qwen2.5:3b"\n'
        'model_embeddings_local = "nomic-embed-text"\n',
        encoding="utf-8",
    )
    shell = GrimoreShell(Session(shell_config))
    _stub_models(shell, [
        {"name": "qwen2.5:3b", "size": 1},
        {"name": "ministral-3:14b", "size": 2},
    ])
    shell.dispatch("models chat ministral-3:14b")

    text = (tmp_path / "grimore.toml").read_text(encoding="utf-8")
    assert 'model_llm_local = "ministral-3:14b"' in text
    # Embedding model untouched.
    assert 'model_embeddings_local = "nomic-embed-text"' in text


def test_status_reflects_session_model_swap(
    shell_config, patched_services, tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    session = Session(shell_config)
    shell = GrimoreShell(session)
    _stub_models(shell, [
        {"name": "qwen2.5:3b", "size": 1},
        {"name": "ministral-3:14b", "size": 2},
    ])
    shell.dispatch("models chat ministral-3:14b")
    capsys.readouterr()  # drop the swap confirmation

    # Patch out the DB stats call — the patched session.db is already a
    # MagicMock, but `_render_status_dashboard` makes its own Database().
    monkeypatch.setattr(
        "grimore.cli.Database",
        lambda *a, **kw: MagicMock(
            get_dashboard_stats=lambda: {
                "total_notes": 0, "tagged_notes": 0, "total_embeddings": 0,
                "cached_embeddings": 0, "categorised_notes": 0,
            },
            get_tag_count=lambda: 0,
            get_category_frequency=lambda: [],
        ),
    )

    shell.dispatch("status")
    out = capsys.readouterr().out
    assert "ministral-3:14b" in out


def test_models_unknown_subcommand_does_not_kill_loop(
    shell_config, patched_services, capsys,
):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("models bogus")
    out = capsys.readouterr().out
    assert "Unknown models subcommand" in out
    assert shell._running is True


def test_models_chat_without_name_prints_hint(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    _stub_models(shell, [{"name": "qwen2.5:3b", "size": 1}])
    shell.dispatch("models chat")
    out = capsys.readouterr().out
    assert "Installed Ollama models" in out
    assert "models chat" in out  # the hint mentions the command


def test_models_when_ollama_unreachable_reports_error(
    shell_config, patched_services, capsys,
):
    shell = GrimoreShell(Session(shell_config))
    _stub_models(shell, [])
    shell.dispatch("models chat 1")
    out = capsys.readouterr().out
    assert "Ollama" in out
    assert shell._running is True


# ── redesign: slash dispatch + @-mentions + new commands ───────────────


def test_slash_routes_to_same_handler_as_word(shell_config, patched_services, capsys):
    """/help and `help` must produce the same output."""
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/help")
    slash_out = capsys.readouterr().out
    shell.dispatch("help")
    word_out = capsys.readouterr().out
    # Both list at least one command line.
    assert "/status" in slash_out and "/status" in word_out


def test_slash_did_you_mean(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/scna")
    out = capsys.readouterr().out
    assert "Unknown slash command" in out and "/scan" in out


def test_slash_empty_is_handled(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/")
    out = capsys.readouterr().out
    assert "Empty slash command" in out
    assert shell._running is True


def test_freeform_question_logged_in_session(shell_config, patched_services):
    """Asking a question must populate last_question/last_answer/question_log."""
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("what is jung?")
    assert session.last_question == "what is jung?"
    assert session.last_answer is not None
    assert session.last_answer["answer"] == "Oracle streams."
    assert session.question_log == ["what is jung?"]


def test_again_repeats_last_question(shell_config, patched_services, capsys):
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("first question")
    capsys.readouterr()
    shell.dispatch("/again")
    # log now has the same question twice.
    assert session.question_log == ["first question", "first question"]


def test_again_with_no_history_is_noop(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/again")
    out = capsys.readouterr().out
    assert "Nothing to repeat" in out


def test_history_lists_recent_questions(shell_config, patched_services, capsys):
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("q one")
    shell.dispatch("q two")
    capsys.readouterr()
    shell.dispatch("/history 5")
    out = capsys.readouterr().out
    assert "q one" in out and "q two" in out


def test_why_prints_sources_of_last_answer(shell_config, patched_services, capsys):
    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("anything")
    capsys.readouterr()
    shell.dispatch("/why")
    out = capsys.readouterr().out
    assert "Note A" in out  # _FakeOracle reports a single source


def test_mention_extraction_attaches_note(tmp_path, shell_config, patched_services):
    note = tmp_path / "vault" / "jung-shadow.md"
    note.write_text("Body about jung shadow.", encoding="utf-8")

    session = Session(shell_config)
    captured = {}
    real_ask = session.oracle.ask_stream  # _FakeOracle stream

    def spy_stream(question, top_k=5, extra_sources=None):
        captured["extras"] = extra_sources
        yield from real_ask(question, top_k=top_k)

    session.oracle.ask_stream = spy_stream

    shell = GrimoreShell(session)
    shell.dispatch("what does @jung-shadow say?")
    assert captured["extras"] is not None
    titles = [t for t, _ in captured["extras"]]
    assert "jung-shadow" in titles


def test_pin_persists_across_asks(tmp_path, shell_config, patched_services):
    note = tmp_path / "vault" / "pinned-note.md"
    note.write_text("Pinned body.", encoding="utf-8")

    session = Session(shell_config)
    captured = []
    real_ask = session.oracle.ask_stream

    def spy_stream(question, top_k=5, extra_sources=None):
        captured.append([t for t, _ in (extra_sources or [])])
        yield from real_ask(question, top_k=top_k)

    session.oracle.ask_stream = spy_stream
    shell = GrimoreShell(session)
    shell.dispatch("/pin @pinned-note")
    shell.dispatch("first ask")
    shell.dispatch("second ask")
    assert captured == [["pinned-note"], ["pinned-note"]]


def test_unpin_clears_all(shell_config, patched_services):
    from grimore.session import NoteAttachment

    session = Session(shell_config)
    shell = GrimoreShell(session)
    # Bypass the resolver by appending an attachment directly — the test
    # is about /unpin clearing state, not about how it got there.
    session.pinned_notes.append(NoteAttachment(
        title="x", path=Path("/tmp/x.md"), content="x",
    ))
    shell.dispatch("/unpin")
    assert session.pinned_notes == []


def test_save_writes_transcript_inside_vault(shell_config, patched_services, tmp_path):
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("first")
    shell.dispatch("/save my-transcript.md")
    target = Path(shell_config.vault.path) / "my-transcript.md"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "Q1. first" in text


def test_save_rejects_path_outside_vault(shell_config, patched_services, tmp_path, capsys):
    session = Session(shell_config)
    shell = GrimoreShell(session)
    shell.dispatch("anything")
    capsys.readouterr()
    # Path traversal attempt.
    shell.dispatch("/save ../escape.md")
    out = capsys.readouterr().out
    assert "outside the vault" in out


def test_approval_prompt_blocks_on_no(shell_config, patched_services, monkeypatch, capsys):
    """`/scan --no-dry-run` without --yes prompts; answering N bails out."""
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: "n")
    session = Session(shell_config)
    shell = GrimoreShell(session)

    called = {"scan": False}

    def fake_scan(vault_path=None, dry_run=None, json_logs=False):
        called["scan"] = True

    monkeypatch.setattr("grimore.cli.scan", fake_scan)
    shell.dispatch("/scan --no-dry-run")
    assert called["scan"] is False


def test_approval_prompt_yes_flag_bypasses(shell_config, patched_services, monkeypatch):
    """`--yes` must skip the input() call and run the action."""
    def explode(*_a, **_kw):
        raise AssertionError("input() should not be called when --yes is set")
    monkeypatch.setattr("builtins.input", explode)

    called = {"scan": False}
    def fake_scan(vault_path=None, dry_run=None, json_logs=False):
        called["scan"] = True
    monkeypatch.setattr("grimore.cli.scan", fake_scan)

    shell = GrimoreShell(Session(shell_config))
    shell.dispatch("/scan --no-dry-run --yes")
    assert called["scan"] is True


def test_at_mention_outside_vault_is_rejected(tmp_path, shell_config, patched_services):
    """A `@../escape` token must NOT resolve to a path outside the vault."""
    # Make a file outside the vault.
    escape = tmp_path / "escape.md"
    escape.write_text("nope", encoding="utf-8")

    session = Session(shell_config)
    captured = {"extras": None}
    real_ask = session.oracle.ask_stream

    def spy_stream(question, top_k=5, extra_sources=None):
        captured["extras"] = extra_sources
        yield from real_ask(question, top_k=top_k)

    session.oracle.ask_stream = spy_stream

    shell = GrimoreShell(session)
    shell.dispatch("look at @../escape")
    # Whether `_do_ask` got called or not, no attachment for the escape
    # file may have been forwarded.
    titles = [t for t, _ in (captured["extras"] or [])]
    assert "escape" not in titles
