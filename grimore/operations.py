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
    history: Optional[list[dict]] = None,
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

    # Only forward optional kwargs when present, so older mocks / alternate
    # Oracle implementations that don't take them keep working unchanged.
    extra_kw: dict = {}
    if extra_sources:
        extra_kw["extra_sources"] = extra_sources
    if history:
        extra_kw["history"] = history

    if stream and export is None:
        # Render tokens as they arrive; collect the answer for the
        # final summary line below.
        ui.section("Oracle")
        answer_chunks: list[str] = []
        sources: list[str] = []
        dropped_citations = 0
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
                dropped_citations = first_event.get("dropped_citations", 0)

        # Drain the rest of the stream without the spinner — tokens now
        # flow at their natural cadence.
        for event in stream_iter:
            if event["type"] == "token":
                console.print(event["text"], end="", soft_wrap=True, markup=False)
                answer_chunks.append(event["text"])
            elif event["type"] == "done":
                sources = event["sources"]
                dropped_citations = event.get("dropped_citations", 0)
        console.print()
        if not answer_chunks:
            console.print()
            console.print(ui.warn_panel(
                "The Oracle returned no tokens. Is Ollama running?",
                title="Silence",
            ))
        result = {
            "answer": "".join(answer_chunks),
            "sources": sources,
            "dropped_citations": dropped_citations,
        }
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

    if result.get("dropped_citations"):
        n = result["dropped_citations"]
        console.print()
        console.print(ui.warn_panel(
            f"{n} citation(s) in the answer weren't among the retrieved "
            f"sources and may be ungrounded — treat them with caution.",
            title="Citations",
        ))

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
            raise typer.BadParameter(
                "--export must resolve to a path inside the vault"
            ) from None

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


# ── Daemon status ────────────────────────────────────────────────────────


def _do_daemon_status(
    config,
    *,
    pid_file: str,
    log_file: str,
    tail_n: int = 5,
) -> dict:
    """Body of ``grimore daemon status``: render a Rich panel summarising
    the live daemon and return a structured dict for tests / scripting.

    The dict shape is stable so a future ``--json`` flag can serialise it
    directly. ``running`` is the only field guaranteed present; the
    others are populated when the daemon is up.
    """
    from grimore.utils.event_log import tail_events
    from grimore.utils.system import _read_pid, is_running

    running = is_running(pid_file)
    pid = _read_pid(pid_file) if running else None
    events = tail_events(log_file, tail_n)

    out: dict = {
        "running": running,
        "pid": pid,
        "pid_file": pid_file,
        "log_file": log_file,
        "events": events,
        "debounce_seconds": getattr(config.daemon, "debounce_seconds", 45),
        "poll_fallback": getattr(config.daemon, "poll_fallback", False),
    }

    # Uptime: PID file mtime is set the moment we acquire the lock, so it's
    # a tight upper bound on the daemon's wall-clock age.
    uptime_s: Optional[float] = None
    if running:
        try:
            uptime_s = max(0.0, time.time() - Path(pid_file).stat().st_mtime)
            out["uptime_s"] = uptime_s
        except OSError:
            pass

    # Newest event timestamp + last-processed file for the "is anything
    # actually happening?" signal.
    last_event = events[-1] if events else None
    if last_event:
        out["last_event_ts"] = last_event.get("ts")
        out["last_event"] = last_event.get("event")
        if last_event.get("path"):
            out["last_path"] = last_event["path"]

    _render_daemon_status(out)
    return out


def _render_daemon_status(s: dict) -> None:
    """Pretty-print a daemon status dict as a Rich panel + recent-events table.

    Kept separate from data-gathering so tests can call ``_do_daemon_status``
    and assert against the dict without parsing Rich output.
    """
    from rich.table import Table
    from rich import box

    badge = ui.daemon_badge(s["running"])
    if s["running"]:
        uptime = _fmt_uptime(s.get("uptime_s"))
        body_lines = [
            f"PID:       [bold cyan]{s.get('pid', '?')}[/]",
            f"Uptime:    [bold]{uptime}[/]",
            f"Debounce:  [bold]{s['debounce_seconds']}s[/]"
            f"{'  (polling)' if s['poll_fallback'] else ''}",
            f"PID file:  [grimore.muted]{s['pid_file']}[/]",
            f"Log file:  [grimore.muted]{s['log_file']}[/]",
        ]
        if s.get("last_event_ts"):
            body_lines.append(
                f"Last event: [bold]{s['last_event']}[/] at "
                f"[grimore.muted]{s['last_event_ts']}[/]"
            )
        body = Text.from_markup("\n".join(body_lines))
        console.print(ui.info_panel(body, title=Text.assemble("Daemon  ", badge)))
    else:
        console.print(ui.warn_panel(
            Text.from_markup(
                f"No daemon running. Start one with [cyan]grimore daemon start[/].\n"
                f"PID file checked: [grimore.muted]{s['pid_file']}[/]"
            ),
            title=Text.assemble("Daemon  ", badge),
        ))

    events = s.get("events") or []
    if not events:
        return
    table = Table(box=box.SIMPLE, header_style="grimore.muted", pad_edge=False)
    table.add_column("Time", style="grimore.muted", no_wrap=True)
    table.add_column("Event", style="grimore.primary", no_wrap=True)
    table.add_column("Path / detail")
    for ev in events:
        detail_parts = []
        for key in ("path", "reason", "chunks", "tags", "error"):
            if key in ev:
                detail_parts.append(f"{key}={ev[key]}")
        table.add_row(
            ev.get("ts", "?"),
            ev.get("event", "?"),
            ", ".join(detail_parts) or "—",
        )
    console.print(table)


def _fmt_uptime(seconds: Optional[float]) -> str:
    """Compact human uptime: 3h 42m / 12m 9s / 47s. Returns '?' on None."""
    if seconds is None:
        return "?"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


# ── Embedding-model migration ────────────────────────────────────────────


def _disk_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (0 if undetectable)."""
    try:
        import shutil
        return int(shutil.disk_usage(str(path.parent if path.is_file() else path)).free)
    except OSError:
        return 0


def _do_migrate_embeddings(
    config,
    db,
    target_model: str,
    *,
    abort: bool = False,
    status_only: bool = False,
    write_config: bool = True,
    progress_factory=None,
) -> dict:
    """Body of ``grimore migrate-embeddings``.

    Three modes share one entry point:

    * ``status_only=True`` — show the current migration (if any) and
      exit. No work performed.
    * ``abort=True`` — drop the shadow table and mark the row aborted.
      Original ``embeddings`` is untouched throughout (the swap is the
      only destructive step).
    * Neither flag — start or resume a migration against ``target_model``,
      run the worker loop to completion, swap atomically, rebuild the
      vec table, and (when ``write_config``) persist the new model
      name to ``grimore.toml``.

    ``progress_factory`` is an optional zero-arg callable returning a
    Rich Progress instance for the worker loop's progress bar. Tests
    pass ``None`` (the default no-progress path) to keep output clean.

    Returns the final migration row dict.
    """
    from grimore.cognition.embedder import Embedder
    from grimore.utils.config import update_cognition_models

    if status_only:
        active = db.get_active_embedding_migration()
        if active is None:
            console.print(ui.info_panel(
                "No embedding migration in flight.",
                title="migrate-embeddings",
            ))
            return {"status": "idle"}
        console.print(ui.info_panel(
            f"Target: [bold]{active['target_model']}[/]\n"
            f"Progress: [bold]{active['done']}/{active['total']}[/]\n"
            f"Started: [grimore.muted]{active['started_at']}[/]",
            title=f"migration #{active['id']}",
        ))
        return active

    if abort:
        result = db.abort_embedding_migration()
        if result is None:
            console.print(ui.info_panel(
                "Nothing to abort — no migration in flight.",
                title="migrate-embeddings",
            ))
            return {"status": "idle"}
        console.print(ui.warn_panel(
            f"Migration #{result['id']} ({result['target_model']}) aborted. "
            f"Shadow table dropped; original embeddings untouched.",
            title="Aborted",
        ))
        return result

    # ── Active mode: start / resume / swap ─────────────────────────────
    active = db.get_active_embedding_migration()
    if active is not None and active["target_model"] != target_model:
        console.print(ui.error_panel(
            f"Another migration is in flight for "
            f"[bold]{active['target_model']}[/].\n"
            f"Abort it first with [cyan]grimore migrate-embeddings --abort[/].",
            title="Migration conflict",
        ))
        raise typer.Exit(code=1)

    # Disk-pressure preflight: a shadow table roughly doubles vector
    # storage. Refuse early rather than dying mid-migration.
    needed = max(db.embeddings_total_bytes(), 1) * 2
    free = _disk_free_bytes(Path(db.db_path))
    if free and free < needed:
        console.print(ui.error_panel(
            f"Not enough free disk to migrate safely.\n"
            f"Need ~[bold]{needed // (1024*1024)} MiB[/], "
            f"have [bold]{free // (1024*1024)} MiB[/].",
            title="Disk pressure",
        ))
        raise typer.Exit(code=1)

    row = db.begin_embedding_migration(target_model)
    if row["total"] == 0:
        console.print(ui.warn_panel(
            "No embeddings to migrate — vault is empty. "
            "Just edit [cyan]grimore.toml[/] to switch models.",
            title="migrate-embeddings",
        ))
        # Still mark the row complete so the bookkeeping is clean.
        if write_config:
            update_cognition_models(embedding_model=target_model)
        return row

    # Build a temporary Embedder targeting the new model.
    from dataclasses import replace
    migrating_cog = replace(config.cognition, model_embeddings_local=target_model)
    migrating_cfg = replace(config, cognition=migrating_cog)
    embedder = Embedder(migrating_cfg, cache=None)  # no cache reuse across models

    pending = db.iter_pending_migration_rows()
    if pending:
        ui.command_header(
            "migrate-embeddings",
            f"→ {target_model} · {row['done']}/{row['total']} done · "
            f"{len(pending)} remaining",
        )

    progress = progress_factory() if progress_factory else None
    task_id = None
    if progress is not None:
        progress.__enter__()
        task_id = progress.add_task("Re-embedding", total=len(pending))

    failures = 0
    try:
        for src_id, note_id, chunk_index, text, page, heading in pending:
            vector = embedder.embed(text or "")
            if vector is None:
                failures += 1
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                continue
            chunk_hash = Embedder.chunk_hash(text or "", target_model)
            db.append_migration_row(
                source_id=src_id,
                note_id=note_id,
                chunk_index=chunk_index,
                text_content=text,
                vector_blob=Embedder.serialize_vector(vector),
                page=page,
                heading=heading,
                chunk_hash=chunk_hash,
            )
            if progress is not None and task_id is not None:
                progress.advance(task_id)
    finally:
        if progress is not None:
            progress.__exit__(None, None, None)

    refreshed = db.get_active_embedding_migration()
    if refreshed is None or refreshed["done"] < refreshed["total"]:
        # Incomplete: surface and bail without swapping. Rerun resumes.
        done = refreshed["done"] if refreshed else row["done"]
        total = refreshed["total"] if refreshed else row["total"]
        console.print(ui.warn_panel(
            f"Migration stopped at [bold]{done}/{total}[/] "
            f"({failures} embed failure(s)). Run the same command again to "
            f"resume — completed rows persist in the shadow table.",
            title="Incomplete",
        ))
        return refreshed or row

    swapped = db.swap_embedding_migration()
    if write_config:
        update_cognition_models(embedding_model=target_model)
    console.print(ui.success_panel(
        f"Migrated [bold]{swapped['total']}[/] embeddings to "
        f"[bold cyan]{target_model}[/]. Original table replaced atomically.",
        title="migrate-embeddings",
    ))
    return swapped


# ── Eval ─────────────────────────────────────────────────────────────────


def _do_eval(
    session: Session,
    golden_path: Path,
    *,
    top_k: int = 5,
    judge: bool = True,
    export: Optional[Path] = None,
) -> dict:
    """Body of ``grimore eval``: load the YAML golden set, run every case
    through the Oracle, render the Rich summary, optionally dump JSON.

    Returns the report ``summary()`` dict so callers (tests, the daemon,
    a future CI integration) can assert on aggregate numbers without
    re-parsing the rendered output. Never raises on a per-case failure —
    the harness already converts those into zero-recall turns.
    """
    from grimore.cognition.eval import load_golden, run_eval, export_report

    if not golden_path.exists():
        console.print(ui.error_panel(
            f"Golden file not found: [bold]{golden_path}[/].\n"
            f"Ship one at [cyan]eval/grimore_golden.yaml[/] or pass [cyan]--golden[/].",
            title="Eval input missing",
        ))
        raise typer.Exit(code=1)

    cases = load_golden(golden_path)
    if not cases:
        console.print(ui.warn_panel(
            f"No cases in [bold]{golden_path}[/].",
            title="Empty golden",
        ))
        return {"top_k": top_k, "n": 0, "aggregate": {}, "turns": []}

    ui.command_header("eval", f"→ {golden_path} · top-k={top_k} · judge={'on' if judge else 'off'}")
    total_turns = sum(1 + len(c.follow_ups) for c in cases)
    console.print(ui.info_panel(
        f"Evaluating [bold]{len(cases)}[/] case(s) ({total_turns} turn(s)) "
        f"against [cyan]{session.vault_root}[/].",
        title="Eval",
    ))

    report = run_eval(session, cases, top_k=top_k, judge=judge)
    report.render(console)

    if export is not None:
        export_report(report, export)
        console.print(ui.success_panel(
            f"Report written to [bold cyan]{export}[/].",
            title="Exported",
        ))

    return report.summary()


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


# ── Knowledge-graph export ────────────────────────────────────────────────


def _do_graph_export(
    session: Session,
    *,
    output: Path,
    fmt: str,
    include_suggested: bool = True,
    suggested_top: int = 3,
    suggested_threshold: float = 0.7,
) -> dict:
    """Body of ``grimore graph export``.

    Builds the graph (nodes + edges) then routes through the format
    writer. Returns the ``graph.stats()`` dict so callers (tests, the
    shell) can assert on counts without re-reading the file.
    """
    from grimore.cognition.graph import build_graph, write_graph, SUPPORTED_FORMATS

    fmt_lc = (fmt or "").lower()
    if fmt_lc in ("canvas",):
        fmt_lc = "obsidian-canvas"
    if fmt_lc not in SUPPORTED_FORMATS:
        console.print(ui.error_panel(
            f"Unsupported format: [bold]{fmt}[/].\n"
            f"Choose one of: [cyan]{', '.join(SUPPORTED_FORMATS)}[/].",
            title="Graph export",
        ))
        raise typer.Exit(code=1)

    ui.command_header(
        "graph export",
        f"→ {output} · format={fmt_lc} · suggested={'on' if include_suggested else 'off'}",
    )
    graph = build_graph(
        session,
        include_suggested=include_suggested,
        suggested_top=suggested_top,
        suggested_threshold=suggested_threshold,
    )
    write_graph(graph, output, fmt_lc)

    stats = graph.stats()
    rows = [
        ("Nodes",  Text(str(stats["nodes"]), style="grimore.success")),
        ("Edges",  Text(str(stats["edges"]), style="grimore.success")),
    ]
    for kind in ("wikilink", "suggested", "contradicts"):
        rows.append((
            f"  {kind}",
            Text(str(stats["by_kind"].get(kind, 0)), style="grimore.accent"),
        ))
    rows.append(("Output", Text(str(output), style="grimore.primary")))
    console.print()
    console.print(ui.success_panel(ui.kv_table(rows), title="Graph exported"))
    return stats
