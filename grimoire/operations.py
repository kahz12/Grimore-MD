"""
Operation functions: command bodies extracted from ``cli.py`` so the
interactive shell and the Typer CLI share one implementation.

Each ``_do_*`` takes a :class:`grimoire.session.Session` plus the same
arguments as the corresponding ``--flag`` form of the CLI command. The
Typer command builds a one-shot Session, calls into here, then tears
down. The shell holds a long-lived Session and reuses it across calls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.text import Text

from grimoire.session import Session
from grimoire.utils import ui
from grimoire.utils.atomic import atomic_write
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

console = ui.console
logger = get_logger(__name__)


def _resolve_note_path(session: Session, raw: str) -> Optional[str]:
    """Resolve a user-supplied note path to the absolute string the DB uses.

    Tries ``raw`` as absolute, then relative-to-cwd, then relative-to-vault.
    Validates the result sits inside the vault to keep ``..`` traversal out.
    Returns None and prints a user-facing error on failure.
    """
    vault_root = session.vault_root.resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        cwd_try = (Path.cwd() / candidate).resolve()
        candidate = cwd_try if cwd_try.exists() else (vault_root / raw).resolve()

    try:
        SecurityGuard.resolve_within_vault(candidate, vault_root)
    except ValueError:
        console.print(ui.error_panel(
            f"[bold]{raw}[/] resolves outside the vault ({vault_root}).",
            title="Invalid path",
        ))
        return None
    if not candidate.exists():
        console.print(ui.error_panel(
            f"[bold]{raw}[/] does not exist.",
            title="Note not found",
        ))
        return None
    return str(candidate)


def _do_ask(
    session: Session,
    question: str,
    *,
    top_k: int = 5,
    export: Optional[Path] = None,
    stream: bool = False,
) -> None:
    """Body of ``grimoire ask``.

    When ``stream=True`` the answer is rendered token-by-token to the
    console via :meth:`Oracle.ask_stream`. When false (default — same
    behaviour as today's CLI), the full JSON answer is fetched and
    rendered in one shot. ``--export`` always uses the non-streaming
    path because it depends on the full answer to write the file.
    """
    config = session.config

    console.print()
    console.print(ui.info_panel(
        Text(question, style="bold white"),
        title="Question",
    ))

    if stream and export is None:
        # Render tokens as they arrive; collect the answer for the
        # final summary line below.
        ui.section("Oracle")
        answer_chunks: list[str] = []
        sources: list[str] = []
        for event in session.oracle.ask_stream(question, top_k=top_k):
            if event["type"] == "token":
                console.print(event["text"], end="", soft_wrap=True, markup=False)
                answer_chunks.append(event["text"])
            elif event["type"] == "done":
                sources = event["sources"]
        console.print()
        if not answer_chunks:
            console.print()
            console.print(ui.warn_panel(
                "The Oracle returned no tokens. Is Ollama running?",
                title="Silence",
            ))
        result = {"answer": "".join(answer_chunks), "sources": sources}
    else:
        with console.status(
            "[grimoire.mystic]The Oracle listens to the whispers...[/]",
            spinner="dots12",
        ):
            result = session.oracle.ask(question, top_k=top_k)
        console.print()
        console.print(ui.oracle_panel(result["answer"]))

    if result["sources"]:
        ui.section("Cited sources")
        for source in result["sources"]:
            console.print(Text.assemble(
                ("  • ", "grimoire.muted"),
                (f"[[{source}]]", "grimoire.accent"),
            ))
    else:
        ui.tip("The Oracle found no relevant notes. Have you run [cyan]grimoire scan[/]?")

    if export:
        vault_root = Path(config.vault.path).resolve()
        try:
            export_path = SecurityGuard.resolve_within_vault(
                vault_root / export, vault_root
            )
        except ValueError:
            console.print(ui.error_panel(
                f"[bold]--export[/] must point inside the vault ({vault_root}).",
                title="Invalid path",
            ))
            raise typer.BadParameter("--export must resolve to a path inside the vault")

        import yaml
        frontmatter_payload = {
            "title": f"Oracle: {question[:30]}...",
            "date": time.strftime("%Y-%m-%d"),
            "type": "oracle_response",
        }
        frontmatter_yaml = yaml.safe_dump(
            frontmatter_payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        body = (
            f"---\n{frontmatter_yaml}---\n\n"
            f"# 🔮 Question: {question}\n\n"
            f"{result['answer']}"
            f"\n\n## Sources\n"
            + "".join(f"- [[{source}]]\n" for source in result["sources"])
        )
        atomic_write(
            export_path,
            lambda fh: fh.write(body.encode("utf-8")),
            mode="wb",
        )

        console.print()
        console.print(ui.success_panel(
            f"Answer saved to [bold cyan]{export_path}[/].",
            title="Exported",
        ))


# ── Chronicler ───────────────────────────────────────────────────────────


def _do_chronicler_list(session: Session, *, decay: bool = False) -> None:
    """Body of ``grimoire chronicler list``.

    ``decay`` is informational — the LLM verdicts are always shown when
    present (since they have already been computed by ``check``). The
    flag exists so the CLI surface mirrors the documented option.
    """
    from grimoire.cognition.chronicler import Chronicler

    chronicler = Chronicler(session)
    stale = chronicler.list_stale()

    if not stale:
        console.print(ui.success_panel(
            "No notes past their freshness window. The vault is current.",
            title="Chronicler",
        ))
        return

    ui.section(f"Notes past their freshness window ({len(stale)})")
    vault_root = session.vault_root.resolve()
    for note in stale:
        try:
            display = Path(note.path).relative_to(vault_root)
        except ValueError:
            display = Path(note.path).name
        if note.likely_stale is True:
            decay_marker = ("  ⚠ likely stale", "grimoire.danger")
        elif note.likely_stale is False:
            decay_marker = ("  ✓ ok per LLM", "grimoire.success")
        else:
            decay_marker = ("", "")
        console.print(Text.assemble(
            ("  ◆ ", "grimoire.rune"),
            (note.title, "grimoire.primary"),
            ("  ", ""),
            (str(display), "grimoire.muted"),
            ("  ", ""),
            (f"{note.days_overdue}d overdue", "grimoire.warning"),
            decay_marker,
        ))
    console.print()
    ui.tip("Run [cyan]grimoire chronicler check <path>[/] for an LLM verdict on a single note.")


def _do_chronicler_check(session: Session, path: str) -> None:
    """Body of ``grimoire chronicler check``."""
    from grimoire.cognition.chronicler import Chronicler

    resolved = _resolve_note_path(session, path)
    if resolved is None:
        return
    chronicler = Chronicler(session)
    with console.status(
        "[grimoire.mystic]Asking the LLM whether this note has gone stale...[/]",
        spinner="dots12",
    ):
        result = chronicler.check_decay(resolved)

    if result is None:
        console.print(ui.warn_panel(
            "No verdict. The note may be exempt from Chronicler\n"
            "(its category has no freshness window) or the LLM call failed.",
            title="Chronicler",
        ))
        return

    likely = bool(result.get("likely_stale"))
    reasons = result.get("reasons") or []
    body = "\n".join(f"  • {r}" for r in reasons) or "(no reasons given)"
    if likely:
        console.print(ui.warn_panel(body, title="Chronicler — likely stale"))
    else:
        console.print(ui.success_panel(body, title="Chronicler — still current"))


def _do_chronicler_verify(session: Session, path: str) -> None:
    """Body of ``grimoire chronicler verify``."""
    from grimoire.cognition.chronicler import Chronicler

    resolved = _resolve_note_path(session, path)
    if resolved is None:
        return
    chronicler = Chronicler(session)
    if chronicler.verify(resolved):
        console.print(ui.success_panel(
            f"[bold]{resolved}[/] marked as verified.",
            title="Chronicler",
        ))
    else:
        console.print(ui.info_panel(
            f"No freshness row for [bold]{resolved}[/].\n"
            f"The note's category is exempt — nothing to update.",
            title="Chronicler",
        ))
