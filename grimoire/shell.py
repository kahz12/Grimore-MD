"""
Interactive REPL — ``grimoire shell``.

The win is keeping :class:`~grimoire.session.Session` warm across calls so
consecutive ``ask`` invocations skip the cold-start of Embedder + LLMRouter.
``ask`` runs through :func:`grimoire.operations._do_ask` with the live
session; everything else delegates to the existing Typer command callbacks
to keep the CLI and the shell exercising the same code.

History is per-vault: questions never bleed between vaults.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
from pathlib import Path
from typing import Callable, Sequence

import typer
from rich.text import Text

from grimoire.operations import (
    _do_ask,
    _do_chronicler_check,
    _do_chronicler_list,
    _do_chronicler_verify,
    _do_mirror_dismiss,
    _do_mirror_list,
    _do_mirror_resolve,
    _do_mirror_scan,
    _do_mirror_show,
)
from grimoire.session import Session
from grimoire.utils import ui
from grimoire.utils.logger import get_logger

console = ui.console
logger = get_logger(__name__)


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
    """Per-vault history file under ``~/.cache/grimoire/``.

    The vault path is hashed (sha256 truncated to 16 hex chars) so the
    cache directory listing does not leak the absolute vault location,
    and so two different vaults do not share question history.

    Created at 0o600 with ``os.open`` to avoid umask-dependent perms
    (same hardening pattern as M-02).
    """
    vault_abs = str(session.vault_root.resolve())
    digest = hashlib.sha256(vault_abs.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "grimoire"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_dir.chmod(0o700)
    except OSError:
        pass
    path = cache_dir / f"shell_history.{digest}"
    if not path.exists():
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    return path


class GrimoireShell:
    """Dispatches one line at a time. The prompt-toolkit loop in
    :meth:`run` is just a thin driver around :meth:`dispatch`, which is
    what tests exercise.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._running = True
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
            "refresh": self._cmd_refresh,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "clear": self._cmd_clear,
        }

    # ── public API ─────────────────────────────────────────────────────

    def dispatch(self, line: str) -> None:
        """Parse one input line and run the matching handler.

        Tolerates unknown commands, parse errors and ``typer.Exit``
        without killing the loop. ``KeyboardInterrupt`` inside a handler
        cancels the command and returns to the prompt.
        """
        line = line.strip()
        if not line:
            return
        try:
            argv = shlex.split(line)
        except ValueError as e:
            console.print(Text(f"Parse error: {e}", style="grimoire.danger"))
            return
        cmd, *rest = argv
        handler = self.commands.get(cmd)
        if handler is None:
            console.print(Text.assemble(
                ("Unknown command: ", "grimoire.danger"),
                (cmd, "grimoire.primary"),
            ))
            ui.tip("Type [cyan]help[/] to see the list.")
            return
        try:
            handler(rest)
        except KeyboardInterrupt:
            console.print()
            console.print(Text("Interrupted.", style="grimoire.muted"))
        except _ShellArgError:
            return
        except typer.Exit:
            return
        except Exception as e:
            logger.exception("shell_command_failed", command=cmd)
            console.print(Text.assemble(
                ("Error: ", "grimoire.danger"),
                (str(e), "grimoire.muted"),
            ))

    def run(self) -> None:
        """Drive the REPL loop until the user types ``exit`` or hits Ctrl+D."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        history = FileHistory(str(_shell_history_path(self.session)))
        completer = WordCompleter(list(self.commands), ignore_case=True)
        ptk = PromptSession(history=history, completer=completer)

        self._print_banner()
        while self._running:
            try:
                line = ptk.prompt(self._prompt_text())
            except KeyboardInterrupt:
                continue  # Ctrl+C clears the buffer, doesn't exit
            except EOFError:
                break  # Ctrl+D exits
            self.dispatch(line)
        self.session.close()
        console.print(Text("Bye.", style="grimoire.muted"))

    # ── prompt + banner ────────────────────────────────────────────────

    def _prompt_text(self) -> str:
        vault_name = self.session.vault_root.name or "vault"
        suffix = " [dry-run]" if self.session.config.output.dry_run else ""
        return f"grimoire({vault_name}){suffix}> "

    def _print_banner(self) -> None:
        console.print(ui.render_banner())
        console.print(ui.info_panel(
            "Interactive shell. Services stay warm across commands.\n"
            "Type [cyan]help[/] for the list, [cyan]exit[/] or Ctrl+D to leave.",
            title="Shell ready",
        ))

    # ── built-in commands ──────────────────────────────────────────────

    def _cmd_help(self, argv: Sequence[str]) -> None:
        if argv:
            target = argv[0]
            doc = self._help_text.get(target)
            if doc:
                console.print(Text(doc, style="grimoire.muted"))
                return
            console.print(Text.assemble(
                ("No help for ", "grimoire.danger"),
                (target, "grimoire.primary"),
            ))
            return
        ui.section("Commands")
        for name in sorted(self.commands):
            short = self._help_text.get(name, "").splitlines()[0] if self._help_text.get(name) else ""
            console.print(Text.assemble(
                ("  ", ""),
                (name.ljust(10), "grimoire.primary"),
                ("  ", ""),
                (short, "grimoire.muted"),
            ))

    def _cmd_exit(self, argv: Sequence[str]) -> None:
        self._running = False

    def _cmd_clear(self, argv: Sequence[str]) -> None:
        console.clear()

    def _cmd_refresh(self, argv: Sequence[str]) -> None:
        self.session.refresh()
        console.print(Text("Session refreshed — services will rebuild on next use.",
                           style="grimoire.success"))

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
        _do_ask(
            self.session,
            question,
            top_k=args.top_k,
            export=args.export,
            stream=(args.export is None) and (not args.no_stream),
        )

    def _cmd_status(self, argv: Sequence[str]) -> None:
        from grimoire.cli import status
        status()

    def _cmd_tags(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="tags", add_help=True)
        parser.add_argument("-n", "--limit", type=int, default=30)
        args = parser.parse_args(argv)
        from grimoire.cli import tags
        tags(limit=args.limit)

    def _cmd_scan(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="scan", add_help=True)
        parser.add_argument("-p", "--vault-path", type=Path, default=None)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.add_argument("--json", dest="json_logs", action="store_true")
        parser.set_defaults(dry_run=None, json_logs=False)
        args = parser.parse_args(argv)
        from grimoire.cli import scan
        scan(vault_path=args.vault_path, dry_run=args.dry_run, json_logs=args.json_logs)

    def _cmd_connect(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="connect", add_help=True)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.add_argument("-t", "--threshold", type=float, default=None)
        parser.set_defaults(dry_run=None)
        args = parser.parse_args(argv)
        if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
            console.print(Text("--threshold must be in [0.0, 1.0]", style="grimoire.danger"))
            return
        from grimoire.cli import connect
        connect(dry_run=args.dry_run, threshold=args.threshold)

    def _cmd_prune(self, argv: Sequence[str]) -> None:
        parser = _NonExitingArgParser(prog="prune", add_help=True)
        parser.add_argument("-p", "--vault-path", type=Path, default=None)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
        parser.set_defaults(dry_run=True)
        args = parser.parse_args(argv)
        from grimoire.cli import prune
        prune(vault_path=args.vault_path, dry_run=args.dry_run)

    def _cmd_category(self, argv: Sequence[str]) -> None:
        if not argv:
            console.print(Text("category: missing subcommand (list · add · rm · notes)",
                               style="grimoire.danger"))
            return
        sub, *rest = argv
        from grimoire.cli import category_add, category_list, category_notes, category_rm
        if sub == "list":
            category_list(vault_path=None)
        elif sub == "add":
            if not rest:
                console.print(Text("category add: missing path", style="grimoire.danger"))
                return
            category_add(path=rest[0], vault_path=None)
        elif sub == "rm":
            parser = _NonExitingArgParser(prog="category rm", add_help=True)
            parser.add_argument("path")
            parser.add_argument("-f", "--force", action="store_true")
            args = parser.parse_args(rest)
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
                ("Unknown category subcommand: ", "grimoire.danger"),
                (sub, "grimoire.primary"),
            ))

    def _cmd_chronicler(self, argv: Sequence[str]) -> None:
        if not argv:
            console.print(Text("chronicler: missing subcommand (list · check · verify)",
                               style="grimoire.danger"))
            return
        sub, *rest = argv
        if sub == "list":
            parser = _NonExitingArgParser(prog="chronicler list", add_help=True)
            parser.add_argument("--decay", action="store_true")
            args = parser.parse_args(rest)
            _do_chronicler_list(self.session, decay=args.decay)
        elif sub == "check":
            if not rest:
                console.print(Text("chronicler check: missing path", style="grimoire.danger"))
                return
            _do_chronicler_check(self.session, rest[0])
        elif sub == "verify":
            if not rest:
                console.print(Text("chronicler verify: missing path", style="grimoire.danger"))
                return
            _do_chronicler_verify(self.session, rest[0])
        else:
            console.print(Text.assemble(
                ("Unknown chronicler subcommand: ", "grimoire.danger"),
                (sub, "grimoire.primary"),
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
                console.print(Text("mirror show: missing id", style="grimoire.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror show: id must be an integer, got {rest[0]!r}",
                                   style="grimoire.danger"))
                return
            _do_mirror_show(self.session, cid)
        elif sub == "dismiss":
            if not rest:
                console.print(Text("mirror dismiss: missing id", style="grimoire.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror dismiss: id must be an integer, got {rest[0]!r}",
                                   style="grimoire.danger"))
                return
            _do_mirror_dismiss(self.session, cid)
        elif sub == "resolve":
            if not rest:
                console.print(Text("mirror resolve: missing id", style="grimoire.danger"))
                return
            try:
                cid = int(rest[0])
            except ValueError:
                console.print(Text(f"mirror resolve: id must be an integer, got {rest[0]!r}",
                                   style="grimoire.danger"))
                return
            _do_mirror_resolve(self.session, cid)
        else:
            console.print(Text.assemble(
                ("Unknown mirror subcommand: ", "grimoire.danger"),
                (sub, "grimoire.primary"),
            ))

    # ── help text ──────────────────────────────────────────────────────

    _help_text = {
        "ask": (
            "ask <question> [-k N] [-e PATH] [--no-stream]\n"
            "  Consult the Oracle. Streams tokens by default; --no-stream\n"
            "  renders the answer in one shot. -e exports to a vault note."
        ),
        "status": "status\n  Vault, cognition, daemon dashboard.",
        "tags": "tags [-n N]\n  Top-N tags by frequency (default 30).",
        "scan": "scan [-p PATH] [--dry-run|--no-dry-run] [--json]\n  Tag and index notes.",
        "connect": "connect [--dry-run|--no-dry-run] [-t T]\n  Inject suggested-connection wikilinks.",
        "prune": "prune [-p PATH] [--dry-run|--no-dry-run]\n  Remove DB rows for vanished notes.",
        "category": (
            "category list | add <path> | rm <path> [-f] | notes <path> [--flat]\n"
            "  Manage the hierarchical category tree."
        ),
        "chronicler": (
            "chronicler list [--decay] | check <path> | verify <path>\n"
            "  Track which notes have likely gone stale."
        ),
        "mirror": (
            "mirror | mirror scan [-k N] [--full] | show <id> | dismiss <id> | resolve <id>\n"
            "  The Black Mirror — surface contradictions across notes."
        ),
        "refresh": "refresh\n  Drop cached services so the next call rebuilds them.",
        "help": "help [command]\n  Show this list, or details for one command.",
        "exit": "exit | quit\n  Leave the shell.",
        "quit": "exit | quit\n  Leave the shell.",
        "clear": "clear\n  Clear the screen.",
    }
