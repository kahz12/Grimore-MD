"""
Operation functions: command bodies extracted from ``cli.py`` so the
interactive shell and the Typer CLI share one implementation.

Each ``_do_*`` takes a :class:`grimore.session.Session` plus the same
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

from grimore.session import Session
from grimore.utils import ui
from grimore.utils.atomic import atomic_write
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

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
    extra_sources: Optional[list[tuple[str, str]]] = None,
) -> dict:
    """Body of ``grimore ask``.

    When ``stream=True`` the answer is rendered token-by-token to the
    console via :meth:`Oracle.ask_stream`. When false (default — same
    behaviour as today's CLI), the full JSON answer is fetched and
    rendered in one shot. ``--export`` always uses the non-streaming
    path because it depends on the full answer to write the file.

    ``extra_sources`` carries ``(title, body)`` pairs for user-explicit
    attachments (``@note`` mentions, ``/pin`` pins) that should ride
    alongside the question into the RAG context.

    Returns the result dict ``{"answer": str, "sources": list[str]}``
    so the shell can log it as ``session.last_answer``.
    """
    config = session.config

    console.print()
    console.print(ui.info_panel(
        Text(question, style="bold white"),
        title="Question",
    ))

    # Only forward ``extra_sources`` when present, so older mocks /
    # alternate Oracle implementations that don't take the kwarg keep
    # working unchanged.
    extra_kw = {"extra_sources": extra_sources} if extra_sources else {}

    if stream and export is None:
        # Render tokens as they arrive; collect the answer for the
        # final summary line below.
        ui.section("Oracle")
        answer_chunks: list[str] = []
        sources: list[str] = []
        stream_iter = session.oracle.ask_stream(
            question, top_k=top_k, **extra_kw
        )

        # Reasoning models (qwen3.5, deepseek-r1, …) often spend tens of
        # seconds in a "thinking" phase where Ollama streams chunks with
        # an empty ``response`` field. The router drops those, so the
        # shell would look frozen between submission and the first real
        # token. A spinner stays up until that first token arrives.
        first_event = None
        with console.status(
            "[grimore.mystic]Waiting on the Oracle…[/]",
            spinner="dots12",
        ):
            for event in stream_iter:
                first_event = event
                if event["type"] in ("token", "done"):
                    break

        if first_event is not None:
            if first_event["type"] == "token":
                console.print(
                    first_event["text"], end="", soft_wrap=True, markup=False
                )
                answer_chunks.append(first_event["text"])
            elif first_event["type"] == "done":
                sources = first_event["sources"]

        # Drain the rest of the stream without the spinner — tokens now
        # flow at their natural cadence.
        for event in stream_iter:
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
            "[grimore.mystic]The Oracle listens to the whispers...[/]",
            spinner="dots12",
        ):
            result = session.oracle.ask(question, top_k=top_k, **extra_kw)
        console.print()
        console.print(ui.oracle_panel(result["answer"]))

    if result["sources"]:
        ui.section("Cited sources")
        for source in result["sources"]:
            console.print(Text.assemble(
                ("  • ", "grimore.muted"),
                (f"[[{source}]]", "grimore.accent"),
            ))
    else:
        ui.tip("The Oracle found no relevant notes. Have you run [cyan]grimore scan[/]?")

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

    return result


# ── Chronicler ───────────────────────────────────────────────────────────


def _do_chronicler_list(session: Session, *, decay: bool = False) -> None:
    """Body of ``grimore chronicler list``.

    ``decay`` is informational — the LLM verdicts are always shown when
    present (since they have already been computed by ``check``). The
    flag exists so the CLI surface mirrors the documented option.
    """
    from grimore.cognition.chronicler import Chronicler

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
            decay_marker = ("  ⚠ likely stale", "grimore.danger")
        elif note.likely_stale is False:
            decay_marker = ("  ✓ ok per LLM", "grimore.success")
        else:
            decay_marker = ("", "")
        console.print(Text.assemble(
            ("  ◆ ", "grimore.rune"),
            (note.title, "grimore.primary"),
            ("  ", ""),
            (str(display), "grimore.muted"),
            ("  ", ""),
            (f"{note.days_overdue}d overdue", "grimore.warning"),
            decay_marker,
        ))
    console.print()
    ui.tip("Run [cyan]grimore chronicler check <path>[/] for an LLM verdict on a single note.")


def _do_chronicler_check(session: Session, path: str) -> None:
    """Body of ``grimore chronicler check``."""
    from grimore.cognition.chronicler import Chronicler

    resolved = _resolve_note_path(session, path)
    if resolved is None:
        return
    chronicler = Chronicler(session)
    with console.status(
        "[grimore.mystic]Asking the LLM whether this note has gone stale...[/]",
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
    """Body of ``grimore chronicler verify``."""
    from grimore.cognition.chronicler import Chronicler

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
    "high": "grimore.danger",
    "medium": "grimore.warning",
    "low": "grimore.muted",
}


def _do_mirror_list(session: Session) -> None:
    """Body of ``grimore mirror`` (default — list open contradictions)."""
    from grimore.cognition.mirror import Mirror

    rows = Mirror(session).list_open()
    if not rows:
        if session.db.count_claims() == 0:
            console.print(ui.info_panel(
                "No claims indexed yet. Run [cyan]grimore mirror scan[/] first.",
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
        style = _SEVERITY_STYLE.get(r.severity, "grimore.muted")
        console.print(Text.assemble(
            ("  ◆ ", "grimore.rune"),
            (f"#{r.id}", "grimore.accent"),
            ("  ", ""),
            (r.severity.upper().ljust(6), style),
            ("  ", ""),
            (r.explanation, "grimore.primary"),
        ))
        console.print(Text.assemble(
            ("       A: ", "grimore.muted"),
            (str(display_a), "grimore.muted"),
            ("  ", ""),
            (r.claim_a[:80], ""),
        ))
        console.print(Text.assemble(
            ("       B: ", "grimore.muted"),
            (str(display_b), "grimore.muted"),
            ("  ", ""),
            (r.claim_b[:80], ""),
        ))
    console.print()
    ui.tip("Run [cyan]grimore mirror show <id>[/] for full detail, "
           "[cyan]dismiss <id>[/] for a false positive, "
           "[cyan]resolve <id>[/] when you've fixed one note.")


def _do_mirror_scan(session: Session, *, top_k: int = 5, full: bool = False) -> None:
    """Body of ``grimore mirror scan``.

    Cost is dominated by LLM calls; we render a Rich progress bar so a
    cold scan over a large vault doesn't look frozen.
    """
    from grimore.cognition.mirror import Mirror

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
            ("Notes scanned",        Text(str(report.notes_scanned),        style="grimore.accent")),
            ("Claims extracted",     Text(str(report.claims_extracted),     style="grimore.success")),
            ("Pairs checked",        Text(str(report.pairs_checked),        style="grimore.accent")),
            ("Contradictions found", Text(str(report.contradictions_found), style="grimore.warning" if report.contradictions_found else "grimore.muted")),
        ]),
        title="Mirror scan summary",
    ))
    if report.contradictions_found:
        ui.tip("Run [cyan]grimore mirror[/] to review the new findings.")


def _do_mirror_show(session: Session, contradiction_id: int) -> None:
    from grimore.cognition.mirror import Mirror

    detail = Mirror(session).show(contradiction_id)
    if detail is None:
        console.print(ui.error_panel(
            f"No contradiction with id [bold]{contradiction_id}[/].",
            title="Black Mirror",
        ))
        return
    style = _SEVERITY_STYLE.get(detail.severity, "grimore.muted")
    ui.section(f"Contradiction #{detail.id}  ·  {detail.severity.upper()}  ·  {detail.status}")
    console.print(Text.assemble(
        ("  ", ""),
        (detail.explanation, style),
    ))
    console.print()
    console.print(Text.assemble(("  Claim A  ", "grimore.accent"), (detail.note_a, "grimore.muted")))
    console.print(Text.assemble(("    ", ""), (detail.claim_a, "grimore.primary")))
    if detail.context_a:
        console.print(Text.assemble(("    ↳ ", "grimore.muted"), (detail.context_a[:600], "grimore.muted")))
    console.print()
    console.print(Text.assemble(("  Claim B  ", "grimore.accent"), (detail.note_b, "grimore.muted")))
    console.print(Text.assemble(("    ", ""), (detail.claim_b, "grimore.primary")))
    if detail.context_b:
        console.print(Text.assemble(("    ↳ ", "grimore.muted"), (detail.context_b[:600], "grimore.muted")))


def _do_mirror_dismiss(session: Session, contradiction_id: int) -> None:
    from grimore.cognition.mirror import Mirror

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
    from grimore.cognition.mirror import Mirror

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


# ── Synthesizer ──────────────────────────────────────────────────────────


def _do_distill(
    session: Session,
    *,
    tag: Optional[str] = None,
    category: Optional[str] = None,
    passages_per_note: int = 3,
    dry_run: bool = False,
) -> None:
    """Body of ``grimore distill``.

    Exactly one of ``tag`` / ``category`` is expected. Renders a summary
    panel with the resolved sources and the output path.
    """
    from grimore.cognition.synthesizer import Synthesizer

    if not tag and not category:
        console.print(ui.error_panel(
            "Provide a selector: [cyan]--tag <name>[/] or [cyan]--category <path>[/].",
            title="Synthesizer",
        ))
        return
    if tag and category:
        console.print(ui.error_panel(
            "Pick one selector — not both [cyan]--tag[/] and [cyan]--category[/].",
            title="Synthesizer",
        ))
        return

    synth = Synthesizer(session)
    with console.status(
        "[grimore.mystic]The Synthesizer distills the chosen notes...[/]",
        spinner="dots12",
    ):
        try:
            report = synth.distill(
                tag=tag,
                category=category,
                passages_per_note=passages_per_note,
                dry_run=dry_run,
            )
        except ValueError as e:
            console.print(ui.error_panel(str(e), title="Synthesizer"))
            return

    if report.skipped_reason:
        console.print(ui.warn_panel(report.skipped_reason, title="Synthesizer"))
        return

    if report.output_path is None:
        console.print(ui.warn_panel(
            "No output produced. Re-run with [cyan]--no-dry-run[/] to write the file.",
            title="Synthesizer",
        ))
        return

    vault_root = session.vault_root.resolve()
    try:
        display = Path(report.output_path).relative_to(vault_root)
    except ValueError:
        display = Path(report.output_path).name
    rows = [
        ("Selector",          Text(report.selector,                     style="grimore.accent")),
        ("Notes considered",  Text(str(report.notes_considered),        style="grimore.accent")),
        ("Excluded (generated)",
                              Text(str(report.notes_excluded_generated), style="grimore.muted")),
        ("Notes used",        Text(str(report.notes_used),              style="grimore.success")),
        ("Output",            Text(str(display),                        style="grimore.primary")),
    ]
    panel = ui.success_panel(ui.kv_table(rows), title="Synthesizer")
    if dry_run:
        panel = ui.info_panel(ui.kv_table(rows), title="Synthesizer (dry run)")
    console.print()
    console.print(panel)
    if report.sources:
        ui.section("Sources")
        for src in report.sources:
            console.print(Text.assemble(
                ("  • ", "grimore.muted"),
                (f"[[{src}]]", "grimore.accent"),
            ))
