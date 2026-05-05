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


# ── Black Mirror ─────────────────────────────────────────────────────────


_SEVERITY_STYLE = {
    "high": "grimoire.danger",
    "medium": "grimoire.warning",
    "low": "grimoire.muted",
}


def _do_mirror_list(session: Session) -> None:
    """Body of ``grimoire mirror`` (default — list open contradictions)."""
    from grimoire.cognition.mirror import Mirror

    rows = Mirror(session).list_open()
    if not rows:
        total_seen = session.db.count_open_contradictions()  # 0 here
        if session.db.count_claims() == 0:
            console.print(ui.info_panel(
                "No claims indexed yet. Run [cyan]grimoire mirror scan[/] first.",
                title="Black Mirror",
            ))
        else:
            console.print(ui.success_panel(
                "No open contradictions. The vault is internally consistent.",
                title="Black Mirror",
            ))
        return

    ui.section(f"Open contradictions ({len(rows)})")
    vault_root = session.vault_root.resolve()
    for r in rows:
        try:
            display_a = Path(r.note_a).relative_to(vault_root)
        except ValueError:
            display_a = Path(r.note_a).name
        try:
            display_b = Path(r.note_b).relative_to(vault_root)
        except ValueError:
            display_b = Path(r.note_b).name
        style = _SEVERITY_STYLE.get(r.severity, "grimoire.muted")
        console.print(Text.assemble(
            ("  ◆ ", "grimoire.rune"),
            (f"#{r.id}", "grimoire.accent"),
            ("  ", ""),
            (r.severity.upper().ljust(6), style),
            ("  ", ""),
            (r.explanation, "grimoire.primary"),
        ))
        console.print(Text.assemble(
            ("       A: ", "grimoire.muted"),
            (str(display_a), "grimoire.muted"),
            ("  ", ""),
            (r.claim_a[:80], ""),
        ))
        console.print(Text.assemble(
            ("       B: ", "grimoire.muted"),
            (str(display_b), "grimoire.muted"),
            ("  ", ""),
            (r.claim_b[:80], ""),
        ))
    console.print()
    ui.tip("Run [cyan]grimoire mirror show <id>[/] for full detail, "
           "[cyan]dismiss <id>[/] for a false positive, "
           "[cyan]resolve <id>[/] when you've fixed one note.")


def _do_mirror_scan(session: Session, *, top_k: int = 5, full: bool = False) -> None:
    """Body of ``grimoire mirror scan``.

    Cost is dominated by LLM calls; we render a Rich progress bar so a
    cold scan over a large vault doesn't look frozen.
    """
    from grimoire.cognition.mirror import Mirror

    mirror = Mirror(session)
    mode = "full" if full else "incremental"
    ui.section(f"Mirror scan ({mode}, top_k={top_k})")

    with ui.progress_bar() as progress:
        extract_task = progress.add_task("Extracting claims", total=None)
        pair_task = progress.add_task("Checking pairs", total=None, visible=False)

        def on_progress(stage: str, current: int, total: int) -> None:
            if stage == "extract":
                if total and progress.tasks[extract_task].total != total:
                    progress.update(extract_task, total=total)
                progress.update(extract_task, completed=current)
            elif stage == "pairs":
                progress.update(pair_task, visible=True)
                if total and progress.tasks[pair_task].total != total:
                    progress.update(pair_task, total=total)
                progress.update(pair_task, completed=current)

        report = mirror.scan(top_k=top_k, full=full, progress=on_progress)

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Notes scanned",        Text(str(report.notes_scanned),        style="grimoire.accent")),
            ("Claims extracted",     Text(str(report.claims_extracted),     style="grimoire.success")),
            ("Pairs checked",        Text(str(report.pairs_checked),        style="grimoire.accent")),
            ("Contradictions found", Text(str(report.contradictions_found), style="grimoire.warning" if report.contradictions_found else "grimoire.muted")),
        ]),
        title="Mirror scan summary",
    ))
    if report.contradictions_found:
        ui.tip("Run [cyan]grimoire mirror[/] to review the new findings.")


def _do_mirror_show(session: Session, contradiction_id: int) -> None:
    from grimoire.cognition.mirror import Mirror

    detail = Mirror(session).show(contradiction_id)
    if detail is None:
        console.print(ui.error_panel(
            f"No contradiction with id [bold]{contradiction_id}[/].",
            title="Black Mirror",
        ))
        return
    style = _SEVERITY_STYLE.get(detail.severity, "grimoire.muted")
    ui.section(f"Contradiction #{detail.id}  ·  {detail.severity.upper()}  ·  {detail.status}")
    console.print(Text.assemble(
        ("  ", ""),
        (detail.explanation, style),
    ))
    console.print()
    console.print(Text.assemble(("  Claim A  ", "grimoire.accent"), (detail.note_a, "grimoire.muted")))
    console.print(Text.assemble(("    ", ""), (detail.claim_a, "grimoire.primary")))
    if detail.context_a:
        console.print(Text.assemble(("    ↳ ", "grimoire.muted"), (detail.context_a[:600], "grimoire.muted")))
    console.print()
    console.print(Text.assemble(("  Claim B  ", "grimoire.accent"), (detail.note_b, "grimoire.muted")))
    console.print(Text.assemble(("    ", ""), (detail.claim_b, "grimoire.primary")))
    if detail.context_b:
        console.print(Text.assemble(("    ↳ ", "grimoire.muted"), (detail.context_b[:600], "grimoire.muted")))


def _do_mirror_dismiss(session: Session, contradiction_id: int) -> None:
    from grimoire.cognition.mirror import Mirror

    if Mirror(session).dismiss(contradiction_id):
        console.print(ui.success_panel(
            f"Contradiction #{contradiction_id} dismissed. "
            f"It will not be re-flagged on future scans.",
            title="Black Mirror",
        ))
    else:
        console.print(ui.error_panel(
            f"No contradiction with id [bold]{contradiction_id}[/].",
            title="Black Mirror",
        ))


def _do_mirror_resolve(session: Session, contradiction_id: int) -> None:
    from grimoire.cognition.mirror import Mirror

    if Mirror(session).resolve(contradiction_id):
        console.print(ui.success_panel(
            f"Contradiction #{contradiction_id} marked as resolved.",
            title="Black Mirror",
        ))
    else:
        console.print(ui.error_panel(
            f"No contradiction with id [bold]{contradiction_id}[/].",
            title="Black Mirror",
        ))
