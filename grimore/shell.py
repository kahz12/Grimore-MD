"""
Interactive REPL — ``grimore shell``.

The win is keeping :class:`~grimore.session.Session` warm across calls so
consecutive ``ask`` invocations skip the cold-start of Embedder + LLMRouter.
``ask`` runs through :func:`grimore.operations._do_ask` with the live
session; everything else delegates to the existing Typer command callbacks
to keep the CLI and the shell exercising the same code.

History is per-vault: questions never bleed between vaults.
"""
from __future__ import annotations

import argparse
import difflib
import re
import shlex
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import typer
from rich.text import Text

from grimore.operations import (
    _do_ask,
    _do_chronicler_check,
    _do_chronicler_list,
    _do_chronicler_verify,
    _do_distill,
    _do_mirror_dismiss,
    _do_mirror_list,
    _do_mirror_resolve,
    _do_mirror_scan,
    _do_mirror_show,
)
from grimore.session import NoteAttachment, Session
from grimore.utils import ui
from grimore.utils.config import is_ignored_path
from grimore.utils.logger import get_logger
from grimore.utils.paths import shell_history_path
from grimore.utils.security import SecurityGuard

console = ui.console
logger = get_logger(__name__)

# Max bytes pulled from a single @-mentioned note. Matches the embedder's
# EMBED_MAX_CHARS so the same defence applies to user-pinned attachments.
_MENTION_MAX_CHARS = 32_000

# Pattern that finds `@token` segments inside an input line. Token can hold
# letters, digits, dashes, underscores, dots and slashes — enough for note
# paths like ``@notes/jung-shadow`` while still preventing shell metachars
# from slipping into the captured group.
_MENTION_RE = re.compile(r"(?:^|\s)@([\w./-]+)")


def _format_bytes(n: int) -> str:
    """Compact human-readable byte size (e.g. ``"4.7 GB"``)."""
    if n <= 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


class _ShellArgError(Exception):
    """Raised when argparse would call sys.exit on a parse error.

    Caught at the dispatch boundary so a typo doesn't kill the loop.
    """


class _NonExitingArgParser(argparse.ArgumentParser):
    """argparse calls ``sys.exit`` on errors, which would kill the shell.

    Override ``error`` to raise instead, and ``exit`` to be a no-op so
    ``-h``/``--help`` does not terminate the process either.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ShellArgError(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:  # type: ignore[override]
        if message:
            console.print(message)
        raise _ShellArgError("help" if status == 0 else (message or ""))


def _shell_history_path(session: Session) -> Path:
    """Per-vault history file in the platform's user cache directory.

    Thin wrapper around :func:`grimore.utils.paths.shell_history_path` —
    kept for back-compat with tests that import it by name.
    """
    return shell_history_path(session.vault_root)


class GrimoreShell:
    """Dispatches one line at a time. The prompt-toolkit loop in
    :meth:`run` is just a thin driver around :meth:`dispatch`, which is
    what tests exercise.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._running = True
        # Legacy word commands — first token matches one of these → run it.
        # Kept for Phase-1 backwards compatibility with the pre-redesign
        # shell. New users should reach for ``/<name>`` instead.
        self.commands: dict[str, Callable[[Sequence[str]], None]] = {
            "ask": self._cmd_ask,
            "status": self._cmd_status,
            "tags": self._cmd_tags,
            "scan": self._cmd_scan,
            "connect": self._cmd_connect,
            "prune": self._cmd_prune,
            "category": self._cmd_category,
            "chronicler": self._cmd_chronicler,
            "mirror": self._cmd_mirror,
            "distill": self._cmd_distill,
            "refresh": self._cmd_refresh,
            "models": self._cmd_models,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "clear": self._cmd_clear,
        }
        # Slash namespace. Mirrors every legacy command and adds the new
        # conversational helpers (/again, /why, /pin, /unpin, /save,
        # /history). The slash dispatcher uses difflib to suggest close
        # matches when the user mistypes.
        self.slash_commands: dict[str, Callable[[Sequence[str]], None]] = {
            **self.commands,
            "again": self._cmd_again,
            "why": self._cmd_why,
            "pin": self._cmd_pin,
            "unpin": self._cmd_unpin,
            "save": self._cmd_save,
            "history": self._cmd_history,
        }
        # Built lazily on first @-mention; reset by /refresh so a vault
        # scan from another terminal becomes visible.
        self._vault_index: Optional[list[tuple[str, Path]]] = None

    # ── public API ─────────────────────────────────────────────────────

    def dispatch(self, line: str) -> None:
        """Parse one input line and run the matching handler.

        Routing order:

        1. Empty / whitespace-only → no-op (never reaches the LLM).
        2. Starts with ``/`` → slash command (with did-you-mean for typos).
        3. First token matches a legacy word command → run it.
        4. Otherwise → freeform question for the Oracle. Any ``@token``
           segments are resolved to vault notes and ride as attachments.

        Tolerates unknown commands, parse errors and ``typer.Exit``
        without killing the loop. ``KeyboardInterrupt`` inside a handler
        cancels the command and returns to the prompt.
        """
        line = line.strip()
        if not line:
            return

        # Slash path — handled before shlex because we want exact text
        # ("/scna" should suggest "/scan", not be split first).
        if line.startswith("/"):
            return self._dispatch_slash(line[1:])

        try:
            argv = shlex.split(line)
        except ValueError as e:
            console.print(Text(f"Parse error: {e}", style="grimore.danger"))
            return
        if not argv:
            return
        cmd, *rest = argv
        handler = self.commands.get(cmd)
        if handler is not None:
            self._run_handler(handler, rest, cmd_name=cmd)
            return

        # No word command matched → treat the whole line as a question.
        self._run_ask_freeform(line)

    def _dispatch_slash(self, body: str) -> None:
        """Run ``/cmd [args]`` after the leading slash has been stripped."""
        body = body.strip()
        if not body:
            console.print(Text("Empty slash command.", style="grimore.danger"))
            ui.tip("Type [cyan]/help[/] to see the list.")
            return
        try:
            argv = shlex.split(body)
        except ValueError as e:
            console.print(Text(f"Parse error: {e}", style="grimore.danger"))
            return
        cmd, *rest = argv
        handler = self.slash_commands.get(cmd)
        if handler is None:
            self._suggest_slash(cmd)
            return
        self._run_handler(handler, rest, cmd_name=f"/{cmd}")

    def _suggest_slash(self, cmd: str) -> None:
        """Print a 'did you mean /xxx?' hint via difflib."""
        matches = difflib.get_close_matches(
            cmd, list(self.slash_commands), n=3, cutoff=0.5
        )
        console.print(Text.assemble(
            ("Unknown slash command: ", "grimore.danger"),
            (f"/{cmd}", "grimore.primary"),
        ))
        if matches:
            suggestion = " · ".join(f"/{m}" for m in matches)
            ui.tip(f"Did you mean: [cyan]{suggestion}[/]?")
        else:
            ui.tip("Type [cyan]/help[/] to see the list.")

    def _run_handler(
        self,
        handler: Callable[[Sequence[str]], None],
        argv: Sequence[str],
        *,
        cmd_name: str,
    ) -> None:
        """Shared try/except guard for both word- and slash-routed calls."""
        try:
            handler(argv)
        except KeyboardInterrupt:
            console.print()
            console.print(Text("Interrupted.", style="grimore.muted"))
        except _ShellArgError:
            return
        except typer.Exit:
            return
        except Exception as e:
            logger.exception("shell_command_failed", command=cmd_name)
            console.print(Text.assemble(
                ("Error: ", "grimore.danger"),
                (str(e), "grimore.muted"),
            ))

    def _run_ask_freeform(self, line: str) -> None:
        """Treat the whole line as a question, extract @-mentions, ask.

        Calls into the ask flow directly (no argparse) so that flag-like
        substrings inside a free-form question (e.g. ``--main thing``) are
        not mistaken for options. Wrapped in :meth:`_run_handler` so that
        ``Ctrl+C`` during a long Ollama wait (thinking-mode models often
        take 30+ seconds before the first token) cancels the question
        cleanly instead of killing the REPL.
        """
        cleaned, attachments = self._extract_mentions(line)
        if not cleaned:
            console.print(Text(
                "Type a question — @mentions on their own aren't asked.",
                style="grimore.muted",
            ))
            return
        self.session.staged_attachments = attachments
        self._run_handler(
            lambda _argv: self._ask(cleaned, top_k=5, export=None, stream=True),
            [],
            cmd_name="ask",
        )

    def run(self) -> None:
        """Drive the REPL loop until the user types ``/exit`` or hits Ctrl+D.

        Wires prompt_toolkit with:

        - A slash- and ``@``-aware completer (popup as soon as the trigger
          character is typed).
        - A bottom toolbar that renders the live state — vault name,
          chat model, embedding model, dry-run badge, pin count.
        - Multi-line submit semantics: plain ``Enter`` submits; ``Esc``-
          then-``Enter`` (also ``Alt+Enter``) inserts a newline; lines
          ending in a bare backslash continue onto the next line.
        - Optional vi-mode editing, gated on ``[shell] vi_mode`` in
          ``grimore.toml``.
        """
        from prompt_toolkit import PromptSession
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.history import FileHistory

        kb = self._build_keybindings()
        completer = self._build_completer()

        history = FileHistory(str(_shell_history_path(self.session)))
        editing_mode = (
            EditingMode.VI
            if self.session.config.shell.vi_mode
            else EditingMode.EMACS
        )
        ptk = PromptSession(
            history=history,
            completer=completer,
            complete_while_typing=True,
            bottom_toolbar=self._bottom_toolbar,
            key_bindings=kb,
            multiline=True,
            editing_mode=editing_mode,
        )

        self._print_banner()
        while self._running:
            try:
                line = ptk.prompt(self._prompt_glyph)
            except KeyboardInterrupt:
                continue  # Ctrl+C clears the buffer, doesn't exit
            except EOFError:
                break  # Ctrl+D exits
            self.dispatch(self._collapse_continuations(line))
        self.session.close()
        console.print(Text("Bye.", style="grimore.muted"))

    def _build_keybindings(self):
        """Custom keymap: Enter submits unless a trailing backslash
        continuation is active; Esc+Enter and Alt+Enter both insert a
        newline. We keep the rest of prompt_toolkit's defaults so users
        get history search (Ctrl+R), word navigation, etc. for free."""
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event):
            buf = event.current_buffer
            text = buf.text
            # Trailing backslash continuation → newline instead of submit.
            if text.rstrip(" \t").endswith("\\"):
                buf.insert_text("\n")
                return
            buf.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline_esc(event):
            event.current_buffer.insert_text("\n")

        # Alt+Enter — same effect as Esc+Enter for keyboards that emit it.
        @kb.add("escape", "\r")
        def _newline_alt(event):
            event.current_buffer.insert_text("\n")

        return kb

    def _build_completer(self):
        """Return a prompt_toolkit completer that surfaces:

        - ``/<word>`` slash commands when the line starts with ``/``;
        - ``@<title>`` vault notes (fuzzy via rapidfuzz) when the user
          is mid-mention.

        For everything else we yield no completions — typing a question
        shouldn't trigger a popup.
        """
        from prompt_toolkit.completion import Completer, Completion

        shell = self  # capture for the nested class

        class _ShellCompleter(Completer):
            def get_completions(self, document, complete_event):
                text_before = document.text_before_cursor
                word = document.get_word_before_cursor(WORD=True)

                # @-mentions: complete vault note titles.
                if word.startswith("@"):
                    fragment = word[1:]
                    yield from shell._mention_completions(fragment, word)
                    return

                # Slash commands: only when the line starts with `/` and
                # we're still on the first token.
                stripped = text_before.lstrip()
                if stripped.startswith("/") and " " not in stripped:
                    typed = stripped[1:]
                    for name in sorted(shell.slash_commands):
                        if name.startswith(typed):
                            yield Completion(
                                f"/{name}",
                                start_position=-len(stripped),
                                display=f"/{name}",
                            )

        return _ShellCompleter()

    def _mention_completions(self, fragment: str, word: str):
        """Yield Completion objects for `@fragment` over the vault index."""
        from prompt_toolkit.completion import Completion

        index = self._ensure_vault_index()
        if not index:
            return
        titles = [t for t, _ in index]
        if not fragment:
            ranked = [(t, 100) for t in titles[:20]]
        else:
            try:
                from rapidfuzz import process, fuzz
            except ImportError:  # pragma: no cover
                ranked = [(t, 100) for t in titles if fragment.lower() in t.lower()][:20]
            else:
                threshold = max(0, min(100, self.session.config.shell.fuzzy_threshold))
                hits = process.extract(
                    fragment, titles, scorer=fuzz.WRatio,
                    limit=20, score_cutoff=threshold,
                )
                ranked = [(t, int(s)) for t, s, _ in hits]
        for title, score in ranked:
            yield Completion(
                f"@{title}",
                start_position=-len(word),
                display=f"@{title}",
                display_meta=f"{score}",
            )

    def _collapse_continuations(self, line: str) -> str:
        """Strip the trailing ``\\`` plus newline from each continuation
        so the dispatcher receives a single logical line."""
        return re.sub(r"\\\s*\n", " ", line)

    # ── prompt + banner ────────────────────────────────────────────────

    def _prompt_glyph(self) -> str:
        """Single-character prompt that reflects the current vi mode.

        Insert / emacs: ``❯ ``
        Vi-normal:      ``∙ ``
        """
        try:
            from prompt_toolkit.application.current import get_app
            from prompt_toolkit.key_binding.vi_state import InputMode
            app = get_app()
            if app.editing_mode.name == "VI" and app.vi_state.input_mode == InputMode.NAVIGATION:
                return "∙ "
        except Exception:
            pass
        return "❯ "

    def _bottom_toolbar(self):
        """Persistent state line — vault · chat model · embed model · dry-run
        · pin count. Re-read every render so changes from /models or /pin
        are immediately visible."""
        cfg = self.session.config
        vault_label = (
            cfg.vault.display_name
            or self.session.vault_root.name
            or "vault"
        )
        bits = [
            f"vault: {vault_label}",
            f"chat: {cfg.cognition.model_llm_local}",
            f"embed: {cfg.cognition.model_embeddings_local}",
        ]
        if cfg.output.dry_run:
            bits.append("dry-run")
        if self.session.pinned_notes:
            bits.append(f"pins: {len(self.session.pinned_notes)}")
        return "  •  ".join(bits)

    def _print_banner(self) -> None:
        console.print(ui.render_banner())
        console.print(ui.info_panel(
            "Interactive shell. Type a question, or [cyan]/help[/] for commands.\n"
            "[cyan]@note[/] attaches a vault note · [cyan]/exit[/] or Ctrl+D to leave.",
            title="Shell ready",
        ))

    # ── built-in commands ──────────────────────────────────────────────

    def _cmd_help(self, argv: Sequence[str]) -> None:
        if argv:
            target = argv[0].lstrip("/")
            doc = self._help_text.get(target)
            if doc:
                console.print(Text(doc, style="grimore.muted"))
                return
            console.print(Text.assemble(
                ("No help for ", "grimore.danger"),
                (target, "grimore.primary"),
            ))
            return
        ui.section("Commands")
        console.print(Text(
            "Type a question to consult the Oracle. Use @notes to attach,\n"
            "/cmd for actions, /help <cmd> for detail.",
            style="grimore.muted",
        ))
        console.print()
        for name in sorted(self.slash_commands):
            short = self._help_text.get(name, "").splitlines()[0] if self._help_text.get(name) else ""
            console.print(Text.assemble(
                ("  ", ""),
                (f"/{name}".ljust(12), "grimore.primary"),
                ("  ", ""),
                (short, "grimore.muted"),
            ))

    def _cmd_exit(self, argv: Sequence[str]) -> None:
        self._running = False

    def _cmd_clear(self, argv: Sequence[str]) -> None:
        console.clear()

    def _cmd_refresh(self, argv: Sequence[str]) -> None:
        self.session.refresh()
        self._vault_index = None  # rebuild on next @-mention
        console.print(Text("Session refreshed — services will rebuild on next use.",
                           style="grimore.success"))

    # ── wrapped operations ─────────────────────────────────────────────

    def _cmd_ask(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="ask", add_help=True)
        parser.add_argument("question", nargs="+")
        parser.add_argument("-k", "--top-k", type=int, default=5)
        parser.add_argument("-e", "--export", type=Path, default=None)
        parser.add_argument("--no-stream", action="store_true",
                            help="Render the full answer at once instead of token-by-token.")
        args = parser.parse_args(argv)
        question = " ".join(args.question)
        # Re-extract mentions: a user typing `/ask @foo what …` should
        # still get the @foo attached. Free-form text goes through
        # _run_ask_freeform which already extracted them.
        cleaned, attachments = self._extract_mentions(question)
        if cleaned:
            question = cleaned
        if attachments and not self.session.staged_attachments:
            self.session.staged_attachments = attachments
        self._ask(
            question,
            top_k=args.top_k,
            export=args.export,
            stream=(args.export is None) and (not args.no_stream),
        )

    def _ask(
        self,
        question: str,
        *,
        top_k: int,
        export: Optional[Path],
        stream: bool,
    ) -> None:
        """Single entry point for every ask path — argparse, freeform,
        ``/again``. Centralises the bookkeeping (history, last_*) and
        attachment-budget reset so each caller can't forget it."""
        extras: list[tuple[str, str]] = []
        for att in (*self.session.pinned_notes, *self.session.staged_attachments):
            extras.append((att.title, att.content))
        try:
            result = _do_ask(
                self.session,
                question,
                top_k=top_k,
                export=export,
                stream=stream,
                extra_sources=extras or None,
            )
        finally:
            # Always clear the one-shot staging, even if _do_ask raised —
            # otherwise a failed call would leave attachments stuck on
            # the next, unrelated question.
            self.session.staged_attachments = []
        if result is not None:
            self.session.last_question = question
            self.session.last_answer = result
            self.session.question_log.append(question)

    def _cmd_status(self, argv: Sequence[str]) -> None:
        from grimore.cli import _render_status_dashboard
        _render_status_dashboard(self.session.config)

    def _cmd_tags(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="tags", add_help=True)
        parser.add_argument("-n", "--limit", type=int, default=30)
        args = parser.parse_args(argv)
        from grimore.cli import tags
        tags(limit=args.limit)

    def _cmd_scan(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="scan", add_help=True)
        parser.add_argument("-p", "--vault-path", type=Path, default=None)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.add_argument("--json", dest="json_logs", action="store_true")
        parser.add_argument("--yes", action="store_true",
                            help="Skip the approval prompt for write mode.")
        parser.set_defaults(dry_run=None, json_logs=False)
        args = parser.parse_args(argv)
        if args.dry_run is False and not self._confirm(
            "scan --no-dry-run will write frontmatter to every changed note.",
            yes=args.yes,
        ):
            return
        from grimore.cli import scan
        scan(vault_path=args.vault_path, dry_run=args.dry_run, json_logs=args.json_logs)

    def _cmd_connect(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="connect", add_help=True)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.add_argument("-t", "--threshold", type=float, default=None)
        parser.add_argument("--yes", action="store_true",
                            help="Skip the approval prompt for write mode.")
        parser.set_defaults(dry_run=None)
        args = parser.parse_args(argv)
        if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
            console.print(Text("--threshold must be in [0.0, 1.0]", style="grimore.danger"))
            return
        if args.dry_run is False and not self._confirm(
            "connect --no-dry-run will inject wikilinks into matching notes.",
            yes=args.yes,
        ):
            return
        from grimore.cli import connect
        connect(dry_run=args.dry_run, threshold=args.threshold)

    def _cmd_prune(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="prune", add_help=True)
        parser.add_argument("-p", "--vault-path", type=Path, default=None)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.add_argument("--yes", action="store_true",
                            help="Skip the approval prompt for write mode.")
        parser.set_defaults(dry_run=True)
        args = parser.parse_args(argv)
        if args.dry_run is False and not self._confirm(
            "prune --no-dry-run will delete DB rows for vanished notes.",
            yes=args.yes,
        ):
            return
        from grimore.cli import prune
        prune(vault_path=args.vault_path, dry_run=args.dry_run)

    def _cmd_category(self, argv: Sequence[str]) -> None:
        if not argv:
            console.print(Text("category: missing subcommand (list · add · rm · notes)",
                               style="grimore.danger"))
            return
        sub, *rest = argv
        from grimore.cli import category_add, category_list, category_notes, category_rm
        if sub == "list":
            category_list(vault_path=None)
        elif sub == "add":
            if not rest:
                console.print(Text("category add: missing path", style="grimore.danger"))
                return
            category_add(path=rest[0], vault_path=None)
        elif sub == "rm":
            parser = _NonExitingArgParser(prog="category rm", add_help=True)
            parser.add_argument("path")
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("--yes", action="store_true",
                                help="Skip the approval prompt.")
            args = parser.parse_args(rest)
            if not self._confirm(
                f"category rm will delete the {args.path!r} category"
                + (" and re-parent its descendants." if args.force
                   else " (must already be empty)."),
                yes=args.yes,
            ):
                return
            category_rm(path=args.path, force=args.force, vault_path=None)
        elif sub == "notes":
            parser = _NonExitingArgParser(prog="category notes", add_help=True)
            parser.add_argument("path")
            parser.add_argument("--flat", dest="recursive", action="store_false")
            parser.set_defaults(recursive=True)
            args = parser.parse_args(rest)
            category_notes(path=args.path, recursive=args.recursive, vault_path=None)
        else:
            console.print(Text.assemble(
                ("Unknown category subcommand: ", "grimore.danger"),
                (sub, "grimore.primary"),
            ))

    def _cmd_chronicler(self, argv: Sequence[str]) -> None:
        if not argv:
            console.print(Text("chronicler: missing subcommand (list · check · verify)",
                               style="grimore.danger"))
            return
        sub, *rest = argv
        if sub == "list":
            parser = _NonExitingArgParser(prog="chronicler list", add_help=True)
            parser.add_argument("--decay", action="store_true")
            args = parser.parse_args(rest)
            _do_chronicler_list(self.session, decay=args.decay)
        elif sub == "check":
            if not rest:
                console.print(Text("chronicler check: missing path", style="grimore.danger"))
                return
            _do_chronicler_check(self.session, rest[0])
        elif sub == "verify":
            if not rest:
                console.print(Text("chronicler verify: missing path", style="grimore.danger"))
                return
            _do_chronicler_verify(self.session, rest[0])
        else:
            console.print(Text.assemble(
                ("Unknown chronicler subcommand: ", "grimore.danger"),
                (sub, "grimore.primary"),
            ))

    def _cmd_mirror(self, argv: Sequence[str]) -> None:
        if not argv:
            _do_mirror_list(self.session)
            return
        sub, *rest = argv
        if sub == "scan":
            parser = _NonExitingArgParser(prog="mirror scan", add_help=True)
            parser.add_argument("-k", "--top-k", type=int, default=5)
            parser.add_argument("--full", action="store_true")
            args = parser.parse_args(rest)
            _do_mirror_scan(self.session, top_k=args.top_k, full=args.full)
        elif sub == "show":
            if not rest:
                console.print(Text("mirror show: missing id", style="grimore.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror show: id must be an integer, got {rest[0]!r}",
                                   style="grimore.danger"))
                return
            _do_mirror_show(self.session, cid)
        elif sub == "dismiss":
            if not rest:
                console.print(Text("mirror dismiss: missing id", style="grimore.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror dismiss: id must be an integer, got {rest[0]!r}",
                                   style="grimore.danger"))
                return
            _do_mirror_dismiss(self.session, cid)
        elif sub == "resolve":
            if not rest:
                console.print(Text("mirror resolve: missing id", style="grimore.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror resolve: id must be an integer, got {rest[0]!r}",
                                   style="grimore.danger"))
                return
            _do_mirror_resolve(self.session, cid)
        else:
            console.print(Text.assemble(
                ("Unknown mirror subcommand: ", "grimore.danger"),
                (sub, "grimore.primary"),
            ))

    def _cmd_models(self, argv: Sequence[str]) -> None:
        """List Ollama models and switch chat/embedding for this session.

        Subcommands:
          models                       — list + show current selection
          models chat  <name|index>    — set the LLM model
          models embed <name|index>    — set the embedding model
          models chat                  — list with a hint to pass a name
          models embed                 — same, embedding side
        """
        if not argv:
            self._models_show()
            return
        sub, *rest = argv
        if sub not in ("chat", "embed"):
            console.print(Text.assemble(
                ("Unknown models subcommand: ", "grimore.danger"),
                (sub, "grimore.primary"),
            ))
            ui.tip("Use [cyan]models[/], [cyan]models chat <name>[/] or [cyan]models embed <name>[/].")
            return

        models = self.session.router.list_installed_models()
        if not models:
            console.print(Text("No models reported by Ollama (or it's unreachable).",
                               style="grimore.danger"))
            return

        if not rest:
            self._models_print_table(models, hint=sub)
            return

        chosen = self._models_resolve(rest[0], models)
        if chosen is None:
            console.print(Text(f"models: no such model {rest[0]!r}",
                               style="grimore.danger"))
            ui.tip("Run [cyan]models[/] to see what's installed.")
            return

        from grimore.utils.config import update_cognition_models

        if sub == "chat":
            self.session.set_chat_model(chosen)
            persisted = update_cognition_models(chat_model=chosen)
            label = "Chat model set to "
        else:
            self.session.set_embedding_model(chosen)
            persisted = update_cognition_models(embedding_model=chosen)
            label = "Embedding model set to "

        suffix = (
            " — saved to grimore.toml."
            if persisted
            else " — no grimore.toml found, this session only."
        )
        console.print(Text.assemble(
            (label, "grimore.success"),
            (chosen, "grimore.primary"),
            (suffix, "grimore.muted"),
        ))

    def _models_show(self) -> None:
        cog = self.session.config.cognition
        console.print(ui.kv_table([
            ("Chat model", cog.model_llm_local),
            ("Embedding model", cog.model_embeddings_local),
        ]))
        models = self.session.router.list_installed_models()
        if not models:
            console.print(Text("No models reported by Ollama (or it's unreachable).",
                               style="grimore.danger"))
            return
        self._models_print_table(models, hint=None)

    def _models_print_table(self, models: list[dict], hint: str | None) -> None:
        ui.section("Installed Ollama models")
        cog = self.session.config.cognition
        for i, m in enumerate(models, start=1):
            name = m["name"]
            badges = []
            if name == cog.model_llm_local:
                badges.append("chat")
            if name == cog.model_embeddings_local:
                badges.append("embed")
            badge = f"  ← {' & '.join(badges)}" if badges else ""
            console.print(Text.assemble(
                (f"  {i:>2}. ", "grimore.muted"),
                (name, "grimore.primary"),
                (f"  ({_format_bytes(m['size'])})", "grimore.muted"),
                (badge, "grimore.accent"),
            ))
        if hint == "chat":
            ui.tip("Pick one: [cyan]models chat <name>[/] or [cyan]models chat <index>[/].")
        elif hint == "embed":
            ui.tip("Pick one: [cyan]models embed <name>[/] or [cyan]models embed <index>[/].")

    @staticmethod
    def _models_resolve(token: str, models: list[dict]) -> str | None:
        """Resolve a user token (name or 1-based index) to a model name."""
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(models):
                return models[idx - 1]["name"]
            return None
        for m in models:
            if m["name"] == token:
                return token
        return None

    def _cmd_distill(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="distill", add_help=True)
        parser.add_argument("-t", "--tag", default=None)
        parser.add_argument("-c", "--category", default=None)
        parser.add_argument("-p", "--passages", type=int, default=3)
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(argv)
        if not args.tag and not args.category:
            console.print(Text(
                "distill: provide --tag <name> or --category <path>",
                style="grimore.danger",
            ))
            return
        if args.tag and args.category:
            console.print(Text(
                "distill: pick one selector — not both --tag and --category",
                style="grimore.danger",
            ))
            return
        _do_distill(
            self.session,
            tag=args.tag,
            category=args.category,
            passages_per_note=args.passages,
            dry_run=args.dry_run,
        )

    # ── new conversational commands ────────────────────────────────────

    def _cmd_again(self, argv: Sequence[str]) -> None:
        """Re-run the previous question with the same flags as default."""
        if not self.session.last_question:
            console.print(Text("Nothing to repeat — ask a question first.",
                               style="grimore.muted"))
            return
        self._ask(
            self.session.last_question,
            top_k=5,
            export=None,
            stream=True,
        )

    def _cmd_why(self, argv: Sequence[str]) -> None:
        """Expand the sources cited by the last answer."""
        answer = self.session.last_answer
        if not answer:
            console.print(Text("No previous answer to inspect.",
                               style="grimore.muted"))
            return
        sources = answer.get("sources") or []
        if not sources:
            console.print(Text("Last answer cited no sources.",
                               style="grimore.muted"))
            return
        ui.section("Sources cited by the last answer")
        for s in sources:
            console.print(Text.assemble(
                ("  • ", "grimore.muted"),
                (f"[[{s}]]", "grimore.accent"),
            ))

    def _cmd_pin(self, argv: Sequence[str]) -> None:
        """Pin one or more @mentioned notes to every future ask."""
        if not argv:
            self._show_pins()
            return
        added: list[str] = []
        for raw in argv:
            token = raw.lstrip("@")
            att = self._resolve_mention(token)
            if att is None:
                console.print(Text(f"pin: no note matches @{token}",
                                   style="grimore.danger"))
                continue
            if any(p.path == att.path for p in self.session.pinned_notes):
                console.print(Text(f"pin: [[{att.title}]] already pinned",
                                   style="grimore.muted"))
                continue
            self.session.pinned_notes.append(att)
            added.append(att.title)
        if added:
            console.print(Text.assemble(
                ("Pinned: ", "grimore.success"),
                (", ".join(f"[[{t}]]" for t in added), "grimore.accent"),
            ))

    def _cmd_unpin(self, argv: Sequence[str]) -> None:
        """Drop one pin (or all pins when called with no argument)."""
        if not argv:
            n = len(self.session.pinned_notes)
            self.session.pinned_notes.clear()
            console.print(Text.assemble(
                ("Cleared ", "grimore.success"),
                (str(n), "grimore.primary"),
                (" pin(s).", "grimore.success"),
            ))
            return
        target = argv[0].lstrip("@").lower()
        before = len(self.session.pinned_notes)
        self.session.pinned_notes = [
            p for p in self.session.pinned_notes
            if target not in p.title.lower() and target not in str(p.path).lower()
        ]
        removed = before - len(self.session.pinned_notes)
        if removed:
            console.print(Text.assemble(
                (f"Removed {removed} pin(s) matching ", "grimore.success"),
                (f"@{target}", "grimore.accent"),
            ))
        else:
            console.print(Text(f"unpin: no pin matched @{target}",
                               style="grimore.muted"))

    def _show_pins(self) -> None:
        if not self.session.pinned_notes:
            console.print(Text("No notes currently pinned.",
                               style="grimore.muted"))
            ui.tip("Use [cyan]/pin @note[/] to attach a note to every ask.")
            return
        ui.section("Pinned notes")
        for p in self.session.pinned_notes:
            console.print(Text.assemble(
                ("  • ", "grimore.muted"),
                (f"[[{p.title}]]", "grimore.accent"),
            ))

    def _cmd_save(self, argv: Sequence[str]) -> None:
        """Export the session transcript as a markdown note inside the vault."""
        parser = _NonExitingArgParser(prog="save", add_help=True)
        parser.add_argument("path", nargs="?", default=None,
                            help="Output path relative to the vault root.")
        args = parser.parse_args(argv)
        if not self.session.question_log:
            console.print(Text("Nothing to save — no questions asked yet.",
                               style="grimore.muted"))
            return
        vault_root = self.session.vault_root.resolve()
        rel = args.path or f"_transcripts/{time.strftime('%Y%m%d-%H%M%S')}.md"
        try:
            target = SecurityGuard.resolve_within_vault(
                vault_root / rel, vault_root
            )
        except ValueError:
            console.print(Text(f"save: {rel!r} resolves outside the vault.",
                               style="grimore.danger"))
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        body = self._render_transcript_markdown()
        target.write_text(body, encoding="utf-8")
        console.print(Text.assemble(
            ("Transcript saved to ", "grimore.success"),
            (str(target.relative_to(vault_root)), "grimore.primary"),
            (".", "grimore.success"),
        ))

    def _render_transcript_markdown(self) -> str:
        """Turn the session's question_log + last_answer into a markdown
        note. Each prior question is emitted as a Q heading; the last
        answer (only the last — we don't keep prior answers in memory in
        v1) lands below the final question."""
        lines = [
            "---",
            f"title: \"Grimore Transcript {time.strftime('%Y-%m-%d %H:%M')}\"",
            f"date: {time.strftime('%Y-%m-%d')}",
            "type: oracle_transcript",
            "---",
            "",
            f"# Transcript — {self.session.vault_root.name}",
            "",
        ]
        for i, q in enumerate(self.session.question_log, start=1):
            lines.append(f"## Q{i}. {q}")
            lines.append("")
        if self.session.last_answer:
            ans = self.session.last_answer.get("answer", "")
            sources = self.session.last_answer.get("sources") or []
            lines.append("### Answer to the last question")
            lines.append("")
            lines.append(ans)
            lines.append("")
            if sources:
                lines.append("**Sources:**")
                for s in sources:
                    lines.append(f"- [[{s}]]")
                lines.append("")
        return "\n".join(lines)

    def _cmd_history(self, argv: Sequence[str]) -> None:
        """Show the last N questions from this session."""
        parser = _NonExitingArgParser(prog="history", add_help=True)
        parser.add_argument("n", nargs="?", type=int, default=10)
        args = parser.parse_args(argv)
        if not self.session.question_log:
            console.print(Text("No questions yet in this session.",
                               style="grimore.muted"))
            return
        items = self.session.question_log[-max(1, args.n):]
        ui.section(f"Last {len(items)} question(s)")
        # The first item's number is offset from the end of the full log.
        start = len(self.session.question_log) - len(items) + 1
        for offset, q in enumerate(items):
            console.print(Text.assemble(
                (f"  {start + offset:>3}. ", "grimore.muted"),
                (q, "grimore.primary"),
            ))

    # ── @-mention plumbing ─────────────────────────────────────────────

    def _extract_mentions(self, line: str) -> tuple[str, list[NoteAttachment]]:
        """Pull ``@token`` segments out of ``line``, resolve each to a
        vault note, and return ``(cleaned_line, attachments)``.

        Tokens that don't resolve to a real, vault-scoped note are left
        in the cleaned line as plain text (and a warning is printed) —
        the question still gets asked, but without that attachment.
        """
        attachments: list[NoteAttachment] = []
        seen: set[Path] = set()
        misses: list[str] = []

        def _sub(match: re.Match) -> str:
            token = match.group(1)
            att = self._resolve_mention(token)
            if att is None:
                misses.append(token)
                return match.group(0)  # leave the literal in place
            if att.path in seen:
                return ""  # silently dedupe within a single ask
            seen.add(att.path)
            attachments.append(att)
            return f" [[{att.title}]]"

        cleaned = _MENTION_RE.sub(_sub, line).strip()
        for token in misses:
            console.print(Text(f"@{token}: no note matched in the vault.",
                               style="grimore.muted"))
        return cleaned, attachments

    def _resolve_mention(self, token: str) -> Optional[NoteAttachment]:
        """Resolve ``token`` to a NoteAttachment, or None.

        Resolution order:
            1. exact path under the vault root (with or without .md);
            2. exact title match (case-insensitive);
            3. rapidfuzz best match above the configured threshold.

        Every successful resolution is re-validated through
        ``SecurityGuard.resolve_within_vault`` and capped at
        ``_MENTION_MAX_CHARS``.
        """
        vault_root = self.session.vault_root.resolve()

        # 1. Direct path attempt.
        for candidate in (vault_root / token, vault_root / f"{token}.md"):
            try:
                resolved = SecurityGuard.resolve_within_vault(candidate, vault_root)
            except ValueError:
                continue
            if resolved.is_file():
                return self._load_attachment(resolved, vault_root)

        # 2 + 3. Index-driven lookups (title exact / fuzzy).
        index = self._ensure_vault_index()
        if not index:
            return None
        token_lc = token.lower()
        for title, path in index:
            if title.lower() == token_lc:
                return self._load_attachment(path, vault_root)

        try:
            from rapidfuzz import process, fuzz
        except ImportError:  # pragma: no cover - rapidfuzz is required
            return None
        threshold = max(0, min(100, self.session.config.shell.fuzzy_threshold))
        titles = [t for t, _ in index]
        best = process.extractOne(token, titles, scorer=fuzz.WRatio,
                                  score_cutoff=threshold)
        if best is None:
            return None
        _, _, idx = best
        return self._load_attachment(index[idx][1], vault_root)

    def _ensure_vault_index(self) -> list[tuple[str, Path]]:
        """Lazily build ``[(title, path), …]`` over .md files in the vault.
        Cached on the instance until ``/refresh`` clears it."""
        if self._vault_index is not None:
            return self._vault_index
        vault_root = self.session.vault_root.resolve()
        ignored = self.session.config.vault.ignored_dirs
        index: list[tuple[str, Path]] = []
        if not vault_root.exists():
            self._vault_index = []
            return self._vault_index
        for md in vault_root.rglob("*.md"):
            if is_ignored_path(md, ignored):
                continue
            try:
                resolved = SecurityGuard.resolve_within_vault(md, vault_root)
            except ValueError:
                continue
            index.append((md.stem, resolved))
        self._vault_index = index
        return index

    def _load_attachment(self, path: Path, vault_root: Path) -> NoteAttachment:
        """Read up to ``_MENTION_MAX_CHARS`` from ``path`` and wrap as
        a ``NoteAttachment``. Always re-validates the path is inside the
        vault — belt-and-braces against a stale cached index."""
        resolved = SecurityGuard.resolve_within_vault(path, vault_root)
        body = resolved.read_text(encoding="utf-8", errors="replace")
        if len(body) > _MENTION_MAX_CHARS:
            body = body[:_MENTION_MAX_CHARS]
        return NoteAttachment(title=resolved.stem, path=resolved, content=body)

    # ── approval prompts ───────────────────────────────────────────────

    def _confirm(self, message: str, *, yes: bool = False) -> bool:
        """Show ``message`` and return True iff the user typed y/yes.

        ``yes=True`` (the ``--yes`` flag from a destructive command)
        bypasses the prompt entirely. Default behaviour is "answer N
        unless the user explicitly confirms", matching dry-run-first.
        """
        if yes:
            return True
        console.print(Text(message, style="grimore.warning"))
        try:
            reply = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            return False
        return reply in ("y", "yes")

    # ── help text ──────────────────────────────────────────────────────

    _help_text = {
        "ask": (
            "/ask <question> [-k N] [-e PATH] [--no-stream]\n"
            "  Consult the Oracle. Streams tokens by default; --no-stream\n"
            "  renders the answer in one shot. -e exports to a vault note.\n"
            "  Tip: just type your question — no /ask needed."
        ),
        "status": "/status\n  Vault, cognition, daemon dashboard.",
        "tags": "/tags [-n N]\n  Top-N tags by frequency (default 30).",
        "scan": (
            "/scan [-p PATH] [--dry-run|--no-dry-run] [--json] [--yes]\n"
            "  Tag and index notes. --no-dry-run prompts before writing."
        ),
        "connect": (
            "/connect [--dry-run|--no-dry-run] [-t T] [--yes]\n"
            "  Inject suggested-connection wikilinks."
        ),
        "prune": (
            "/prune [-p PATH] [--dry-run|--no-dry-run] [--yes]\n"
            "  Remove DB rows for vanished notes."
        ),
        "category": (
            "/category list | add <path> | rm <path> [-f] [--yes] | notes <path> [--flat]\n"
            "  Manage the hierarchical category tree."
        ),
        "chronicler": (
            "/chronicler list [--decay] | check <path> | verify <path>\n"
            "  Track which notes have likely gone stale."
        ),
        "mirror": (
            "/mirror | mirror scan [-k N] [--full] | show <id> | dismiss <id> | resolve <id>\n"
            "  The Black Mirror — surface contradictions across notes."
        ),
        "distill": (
            "/distill --tag <name> | --category <path> [-p N] [--dry-run]\n"
            "  Synthesize matching notes into a single _synthesis/ note."
        ),
        "refresh": "/refresh\n  Drop cached services and the @-mention index.",
        "models": (
            "/models | /models chat <name|index> | /models embed <name|index>\n"
            "  List Ollama models and switch the chat/embedding pick.\n"
            "  Updates the live session AND rewrites [cognition] in grimore.toml."
        ),
        "again": "/again\n  Re-run the last question with the same defaults.",
        "why": "/why\n  Re-print the sources cited by the last answer.",
        "pin": (
            "/pin @note [@note …]\n"
            "  Pin notes to every future ask in this session.\n"
            "  /pin alone lists current pins."
        ),
        "unpin": "/unpin [@note]\n  Drop one pin (or all when called bare).",
        "save": (
            "/save [path]\n"
            "  Export the session transcript as a markdown note in the vault.\n"
            "  Default path: _transcripts/<timestamp>.md"
        ),
        "history": "/history [N]\n  Show the last N questions (default 10).",
        "help": "/help [command]\n  Show this list, or details for one command.",
        "exit": "/exit | /quit\n  Leave the shell.",
        "quit": "/exit | /quit\n  Leave the shell.",
        "clear": "/clear\n  Clear the screen.",
    }
