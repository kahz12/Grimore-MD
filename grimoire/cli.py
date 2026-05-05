"""
Command Line Interface (CLI) for Project Grimoire.
This module defines the user commands for scanning vaults, connecting notes,
consulting the Oracle (RAG), and managing the background daemon.
"""
import time
from pathlib import Path
from typing import Optional

import typer
from rich.text import Text

from grimoire.cognition.connector import Connector
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.oracle import Oracle
from grimoire.cognition.tagger import Tagger
from grimoire.ingest.parser import MarkdownParser
from grimoire.memory.db import Database
from grimoire.memory.maintenance import MaintenanceRunner
from grimoire.memory.taxonomy import load_taxonomy_from_vault, save_taxonomy_to_vault
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
from grimoire.output.frontmatter_writer import FrontmatterWriter
from grimoire.output.git_guard import GitGuard
from grimoire.output.link_injector import LinkInjector
from grimoire.session import Session
from grimoire.utils import ui
from grimoire.utils.atomic import atomic_write
from grimoire.utils.config import is_ignored_path, load_config
from grimoire.utils.logger import get_logger, setup_logger
from grimoire.utils.preflight import PreflightChecker, PreflightReport
from grimoire.utils.security import SecurityGuard


# Define the Typer application with metadata for the help screen
app = typer.Typer(
    help="🔮 [bold medium_purple3]Grimoire v2.0[/] — Automated Knowledge Engine",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="[italic grey50]Sense the vault · surface connections · wake the Oracle[/]",
)

console = ui.console
logger = get_logger(__name__)


def _mode_badge(is_dry_run: bool) -> Text:
    """Returns a visual badge indicating if the current execution is a dry run or live."""
    return ui.dry_run_badge() if is_dry_run else ui.live_mode_badge()


def _render_preflight_report(report: PreflightReport) -> None:
    """Print each preflight check with a ✓/⚠/✗ bullet and its fix suggestion."""
    for c in report.checks:
        if c.ok:
            bullet = ("  ✓ ", "grimoire.success")
        elif c.severity == "warning":
            bullet = ("  ⚠ ", "grimoire.warning")
        else:
            bullet = ("  ✗ ", "grimoire.danger")
        console.print(Text.assemble(
            bullet,
            (c.name, "grimoire.primary"),
            ("  ", ""),
            (c.message, "grimoire.muted"),
        ))
        if not c.ok and c.fix:
            for line in c.fix.splitlines():
                console.print(Text.assemble(
                    ("     ↳ ", "grimoire.accent"),
                    (line, "grimoire.muted"),
                ))


def _preflight_or_exit(config, *, check_git: bool | None = None) -> PreflightReport:
    """
    Run preflight and exit loudly on hard failure. On success, warnings are
    still printed (e.g. missing git with auto_commit on) so the user knows.
    """
    report = PreflightChecker(config).run(check_git=check_git)
    if not report.ok:
        console.print()
        console.print(ui.error_panel(
            "Preflight validation detected blocking issues.",
            title="Preflight",
        ))
        _render_preflight_report(report)
        console.print()
        ui.tip("Run [cyan]grimoire preflight[/] to re-check without launching the scan.")
        raise typer.Exit(code=1)
    if report.has_warnings:
        console.print()
        _render_preflight_report(report)
        console.print()
    return report


@app.command(rich_help_panel="Knowledge ops")
def scan(
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
    dry_run: bool = typer.Option(None, "--dry-run/--no-dry-run", help="Simulate changes without writing"),
    json_logs: bool = typer.Option(False, "--json", help="Emit logs in JSON format"),
):
    """
    📖 Scan the vault, tag new or changed notes and index their embeddings.
    
    This is the primary ingestion command. It identifies new or modified Markdown files,
    extracts metadata/content, generates tags and summaries via LLM, and creates
    vector embeddings for semantic search.
    """
    setup_logger(json_format=json_logs)
    config = load_config()

    actual_vault_path = vault_path or Path(config.vault.path)
    is_dry_run = dry_run if dry_run is not None else config.output.dry_run

    ui.command_header("scan", f"→ {actual_vault_path}")
    console.print(Text.assemble("  ", _mode_badge(is_dry_run)))

    if not actual_vault_path.exists():
        console.print(ui.error_panel(
            f"Vault not found at [bold]{actual_vault_path}[/]\n"
            f"Check the path or update [cyan]grimoire.toml[/].",
            title="Vault missing",
        ))
        raise typer.Exit(code=1)

    # Preflight: catch "Ollama is down" or "model not pulled" before the first
    # file hits the LLM. Uses the effective auto_commit setting for git check.
    _preflight_or_exit(config)

    # Initialize core services
    git_guard = GitGuard(str(actual_vault_path))
    if not git_guard.is_repo_ready():
        console.print(ui.warn_panel(
            "The directory is not a git repository. The scan will continue without a safety net.\n"
            "Initialise it with [cyan]git init[/] inside the vault to enable automatic snapshots.",
            title="Git not detected",
        ))

    db = Database(config.memory.db_path)
    parser = MarkdownParser()
    router = LLMRouter(config)
    vault_tax = load_taxonomy_from_vault(actual_vault_path)
    tagger = Tagger(config, router, vault_tax)
    writer = FrontmatterWriter()
    embedder = Embedder(config, cache=db)
    security = SecurityGuard(str(actual_vault_path))

    # Identify files to process (filtering out ignored directories)
    vault_root = actual_vault_path.resolve()
    files = []
    for f in actual_vault_path.glob("**/*.md"):
        if is_ignored_path(f, config.vault.ignored_dirs):
            continue
        try:
            SecurityGuard.resolve_within_vault(f, vault_root)
        except ValueError:
            logger.warning("path_escape_skipped", path=str(f))
            continue
        files.append(f)

    if not files:
        console.print(ui.info_panel(
            f"No [bold].md[/] notes found in [cyan]{actual_vault_path}[/].",
            title="Empty vault",
        ))
        return

    stats = {"unchanged": 0, "processed": 0, "skipped": 0, "errors": 0, "chunks": 0}
    ui.section(f"Scanning {len(files)} notes")

    with ui.progress_bar() as progress:
        task = progress.add_task("Scanning", total=len(files))
        for file in files:
            rel = file.relative_to(actual_vault_path)
            progress.update(task, description=f"[grimoire.muted]{rel}[/]")

            try:
                # Step 1: Parse Markdown and metadata
                note = parser.parse_file(file, vault_root=vault_root)
            except Exception as e:
                stats["errors"] += 1
                logger.error("parse_failed", path=str(file), error=str(e))
                progress.advance(task)
                continue

            # Respect privacy policy in frontmatter
            if note.metadata.get("privacy") == "never_process":
                stats["skipped"] += 1
                progress.advance(task)
                continue

            # Idempotency check: Skip if content hasn't changed
            if db.get_content_hash_by_path(str(file)) == note.content_hash:
                stats["unchanged"] += 1
                progress.advance(task)
                continue

            # Security: Scan for PII/Secrets
            sensitive_findings = security.scan_for_sensitive_data(note.content)
            if sensitive_findings:
                logger.warning("sensitive_data_detected", path=str(file), types=sensitive_findings)

            # Sanitize content before sending to LLM
            clean_content = security.sanitize_prompt(note.content)
            
            # Step 2: Cognition (Tagging and Summarization)
            cognition_data = tagger.tag_note(clean_content)

            if is_dry_run:
                stats["processed"] += 1
                logger.info(
                    "dry_run_preview",
                    path=str(file),
                    tags=cognition_data["tags"],
                    summary=cognition_data["summary"][:80],
                )
                progress.advance(task)
                continue

            try:
                # Step 3: Safety snapshot before writing
                if config.output.auto_commit and git_guard.is_repo_ready():
                    git_guard.commit_pre_change(str(file))

                # Step 4: Write metadata back to note file
                metadata_updates = {
                    "tags": cognition_data["tags"],
                    "summary": cognition_data["summary"],
                    "category": cognition_data.get("category", ""),
                    "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                writer.write_metadata(file, metadata_updates, dry_run=False)

                # Step 5: Update persistence layer (DB)
                note_id = db.upsert_note(str(file), note.title, note.content_hash)
                db.update_last_tagged(str(file))
                if note_id is not None:
                    db.upsert_tags(note_id, cognition_data["tags"])
                    db.set_note_category(note_id, cognition_data.get("category") or None)

                # Step 6: Vectorize content for semantic search
                embedded = embedder.embed_chunks(clean_content)
                if embedded and note_id is not None:
                    db.delete_note_embeddings(note_id)
                    for idx, (chunk_text, vector) in enumerate(embedded):
                        db.store_embedding(
                            note_id,
                            idx,
                            chunk_text[:500],
                            embedder.serialize_vector(vector),
                        )
                    stats["chunks"] += len(embedded)

                stats["processed"] += 1
            except Exception as e:
                stats["errors"] += 1
                logger.error("processing_failed", path=str(file), error=str(e))

            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────────────────
    summary_rows = [
        ("Processed",  Text(str(stats["processed"]), style="grimoire.success")),
        ("Unchanged",  Text(str(stats["unchanged"]), style="grimoire.muted")),
        ("Skipped",    Text(str(stats["skipped"]),   style="grimoire.warning")),
        ("Errors",     Text(str(stats["errors"]),    style="grimoire.danger" if stats["errors"] else "grimoire.muted")),
        ("Chunks",     Text(str(stats["chunks"]),    style="grimoire.accent")),
    ]
    console.print()
    console.print(ui.success_panel(ui.kv_table(summary_rows), title="Scan Summary"))

    if is_dry_run and stats["processed"] > 0:
        ui.tip("This was a dry run. Run [cyan]grimoire scan --no-dry-run[/] to apply changes.")


def _validate_threshold(value: Optional[float]) -> Optional[float]:
    """
    --threshold is a cosine-similarity floor. Cosine ranges in [-1, 1] but
    a value below 0 would propose every pair (spam) and above 1 would
    reject every pair (no-op). Constrain to the meaningful sub-range so
    a typo doesn't silently flood the vault with wikilinks.
    """
    if value is None:
        return None
    if not (0.0 <= value <= 1.0):
        raise typer.BadParameter("--threshold must be in [0.0, 1.0]")
    return value


@app.command(rich_help_panel="Knowledge ops")
def connect(
    dry_run: bool = typer.Option(None, "--dry-run/--no-dry-run", help="Simulate link injection"),
    threshold: float = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Min. cosine similarity to propose a link, in [0.0, 1.0] (default: cognition.connect_threshold).",
        callback=_validate_threshold,
    ),
):
    """
    🕸️  Discover semantic connections between notes and inject [[wikilinks]].

    This command uses vector embeddings and cosine similarity to find related notes.
    It then injects a 'Suggested Connections' section into the Markdown files.
    """
    setup_logger()
    config = load_config()
    is_dry_run = dry_run if dry_run is not None else config.output.dry_run
    effective_threshold = threshold if threshold is not None else config.cognition.connect_threshold

    ui.command_header("connect", "Weaving threads between ideas")
    console.print(Text.assemble("  ", _mode_badge(is_dry_run)))

    db = Database(config.memory.db_path)
    embedder = Embedder(config, cache=db)
    connector = Connector(db, embedder)
    injector = LinkInjector()

    all_embeddings = db.get_all_embeddings()
    if not all_embeddings:
        console.print(ui.warn_panel(
            "No embeddings yet. Run [cyan]grimoire scan --no-dry-run[/] first.",
            title="No vector memory",
        ))
        return

    ui.section(f"Searching connections between {len(all_embeddings)} fragments")

    processed_notes: set[int] = set()
    total_links = 0
    unique_sources = 0

    with ui.progress_bar() as progress:
        task = progress.add_task("Connecting", total=len(all_embeddings))
        for note_id, _text, vector_blob in all_embeddings:
            progress.advance(task)
            if note_id in processed_notes:
                continue
            processed_notes.add(note_id)

            location = db.get_note_location(note_id)
            if not location:
                logger.warning("orphan_embedding", note_id=note_id)
                continue
            path, title = location

            # Find similar notes using cosine similarity on embeddings.
            # dedupe_by_note guarantees we see distinct notes (not 12 chunks
            # of the same note) so the 3-candidate budget is always fillable.
            vector = embedder.deserialize_vector(vector_blob)
            similar = connector.find_similar_notes(
                vector, top_k=12, exclude_note_id=note_id, dedupe_by_note=True,
            )

            candidates = [s for s in similar if s["score"] > effective_threshold][:3]

            if not candidates:
                continue

            connections_to_inject = []
            bullet_lines = []
            for c in candidates:
                c_title = db.get_note_title(c["note_id"])
                if not c_title:
                    continue
                bullet_lines.append(f"  ↳ [cyan]{c_title}[/]  [grimoire.muted](score {c['score']:.2f})[/]")
                connections_to_inject.append({"title": c_title, "reason": "High semantic similarity."})

            if connections_to_inject:
                unique_sources += 1
                total_links += len(connections_to_inject)
                progress.console.print(Text.assemble(
                    ("◆ ", "grimoire.rune"),
                    (str(title), "grimoire.primary"),
                ))
                for line in bullet_lines:
                    progress.console.print(line)
                
                # Inject the discovered links back into the file
                injector.inject_links(Path(path), connections_to_inject, dry_run=is_dry_run)

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Connected notes", Text(str(unique_sources), style="grimoire.success")),
            ("Suggested links", Text(str(total_links), style="grimoire.accent")),
        ]),
        title="Connection summary",
    ))


@app.command(rich_help_panel="Knowledge ops")
def ask(
    question: str = typer.Argument(..., help="The question to ask the Oracle"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Context fragments to retrieve"),
    export: Path = typer.Option(None, "--export", "-e", help="Save the answer as a markdown note"),
):
    """
    🔮 Consult the Grimoire Oracle about your vault's knowledge.
    
    This is a Retrieval-Augmented Generation (RAG) system. It searches for relevant
    note fragments, provides them as context to the LLM, and generates an answer
    with citations to your own notes.
    """
    setup_logger()
    config = load_config()

    ui.command_header("ask", "Consulting the Oracle")

    # ask doesn't write to the vault, so don't nag about missing git.
    _preflight_or_exit(config, check_git=False)

    session = Session(config)
    _do_ask(session, question, top_k=top_k, export=export)


@app.command(rich_help_panel="Daemon")
def daemon(
    action: str = typer.Argument("run", help="Action: run · start · stop · status"),
    json_logs: bool = typer.Option(False, "--json", help="Emit logs in JSON format"),
):
    """
    🧿 Manage the Grimoire daemon (foreground / background / status).
    
    The daemon watches for file changes in real-time and automatically 
    processes them after a debounce period.
    """
    from grimoire.utils.system import is_running, start_daemon_background, stop_daemon

    pid_file = "grimoire.pid"
    log_file = "grimoire.log"

    ui.command_header("daemon", action)

    if action == "run":
        # Run daemon in foreground
        setup_logger(json_format=json_logs)
        config = load_config()
        from grimoire.daemon import GrimoireDaemon

        _preflight_or_exit(config)

        console.print(ui.info_panel(
            "The daemon will run in the foreground. [bold]Ctrl-C[/] to stop.",
            title="Foreground mode",
        ))
        instance = GrimoireDaemon(config, pid_file=pid_file)
        instance.start()
    elif action == "start":
        # Start daemon in background
        if is_running(pid_file):
            console.print(ui.warn_panel(
                f"A daemon is already running. PID file: [cyan]{pid_file}[/].",
                title="Already running",
            ))
            return
        # Validate BEFORE forking — otherwise the user would only find the
        # failure by tailing the log file.
        _preflight_or_exit(load_config())
        start_daemon_background(pid_file, log_file)
        console.print(ui.success_panel(
            f"Daemon running in the background. Logs → [cyan]{log_file}[/].",
            title="Started",
        ))
    elif action == "stop":
        # Stop background daemon
        if not is_running(pid_file):
            console.print(ui.info_panel("No daemon running.", title="Nothing to stop"))
            return
        stop_daemon(pid_file)
        console.print(ui.success_panel("Daemon stopped.", title="Stop"))
    elif action == "status":
        # Check if daemon is active
        active = is_running(pid_file)
        console.print(Text.assemble("  ", ui.daemon_badge(active)))
    else:
        console.print(ui.error_panel(
            f"Unknown action: [bold]{action}[/]\nUse [cyan]run · start · stop · status[/].",
            title="Invalid command",
        ))
        raise typer.Exit(code=2)


@app.command(rich_help_panel="Knowledge ops")
def tags(
    limit: int = typer.Option(30, "--limit", "-n", help="How many tags to show (by frequency)"),
):
    """🏷️  List the most-used tags and how many notes each one labels."""
    setup_logger()
    config = load_config()
    db = Database(config.memory.db_path)

    ui.command_header("tags", f"top {limit}")

    rows = db.get_tag_frequency(limit=limit)
    if not rows:
        console.print(ui.info_panel(
            "No tags recorded yet. Run [cyan]grimoire scan --no-dry-run[/] first.",
            title="No tags",
        ))
        return

    console.print()
    console.print(ui.tag_frequency_table(rows))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(rows)} tags", "grimoire.accent"),
        ("  ·  ", "grimoire.muted"),
        (f"{db.get_tag_count()} unique in use", "grimoire.muted"),
    ))


@app.command(rich_help_panel="System")
def prune(
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="List only, don't delete"),
):
    """
    🧹 Remove DB entries for notes that no longer exist on disk.
    
    Ensures the database remains synchronized with the actual file system
    by removing orphan records and cleaning up unused tags.
    """
    setup_logger()
    config = load_config()
    actual_vault_path = vault_path or Path(config.vault.path)

    ui.command_header("prune", f"→ {actual_vault_path}")
    console.print(Text.assemble("  ", _mode_badge(dry_run)))

    if not actual_vault_path.exists():
        console.print(ui.error_panel(
            f"Vault not found at [bold]{actual_vault_path}[/]",
            title="Vault missing",
        ))
        raise typer.Exit(code=1)

    db = Database(config.memory.db_path)

    # Scan disk for existing files to identify orphans in DB
    existing_paths = {
        str(f) for f in actual_vault_path.glob("**/*.md")
        if not is_ignored_path(f, config.vault.ignored_dirs)
    }

    stale = db.find_stale_notes(existing_paths)
    if not stale:
        console.print(ui.success_panel(
            "No orphaned notes. The database is synchronised with the vault.",
            title="All clean",
        ))
        return

    ui.section(f"Orphaned notes detected ({len(stale)})")
    for _, path in stale[:50]:
        try:
            display = Path(path).relative_to(actual_vault_path)
        except ValueError:
            display = Path(path).name
        console.print(Text.assemble(
            ("  ✗ ", "grimoire.danger"),
            (str(display), "grimoire.muted"),
        ))
    if len(stale) > 50:
        console.print(Text(f"  … and {len(stale) - 50} more", style="grimoire.muted"))

    if dry_run:
        console.print()
        ui.tip("Dry run complete. Run [cyan]grimoire prune --no-dry-run[/] to delete.")
        return

    # Delete records from DB
    removed = db.prune_missing_notes(existing_paths)
    purged = db.purge_unused_tags()

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Notes deleted", Text(str(removed), style="grimoire.success")),
            ("Tags purged",   Text(str(purged),  style="grimoire.accent")),
        ]),
        title="Prune complete",
    ))


@app.command(rich_help_panel="System")
def status():
    """
    🧭 Full dashboard: vault, cognition, daemon.
    
    Displays a high-level overview of the system state, including
    indexing progress, model configurations, and daemon status.
    """
    from grimoire.utils.system import is_running

    config = load_config()
    db = Database(config.memory.db_path)

    # Gather metrics from database
    stats = db.get_dashboard_stats()
    total_notes = stats["total_notes"]
    tagged_notes = stats["tagged_notes"]
    total_embeddings = stats["total_embeddings"]
    cached_embeddings = stats["cached_embeddings"]
    categorised_notes = stats["categorised_notes"]
    unique_tags = db.get_tag_count()
    category_rows = db.get_category_frequency()

    console.print()
    console.print(ui.render_banner())

    ui.section("Vault")
    console.print(ui.kv_table([
        ("Path",         Text(config.vault.path, style="grimoire.accent")),
        ("Notes",        ui.coverage_bar(tagged_notes, total_notes)),
        ("Chunks",       Text(str(total_embeddings), style="grimoire.accent")),
        ("Cache",        Text(f"{cached_embeddings} vectors", style="grimoire.accent")),
        ("Mode",         _mode_badge(config.output.dry_run)),
        ("Auto-commit",  Text("yes" if config.output.auto_commit else "no", style="grimoire.accent")),
    ]))

    ui.section("Cognition")
    console.print(ui.kv_table([
        ("LLM",          Text(config.cognition.model_llm_local,        style="grimoire.accent")),
        ("Embeddings",   Text(config.cognition.model_embeddings_local, style="grimoire.accent")),
        ("Unique tags",  Text(str(unique_tags),                        style="grimoire.accent")),
        ("Categories",   Text(f"{len(category_rows)} active · {categorised_notes} notes",
                              style="grimoire.accent")),
        ("Remote",       Text("allowed" if config.cognition.allow_remote else "local-first",
                              style="grimoire.warning" if config.cognition.allow_remote else "grimoire.success")),
    ]))

    ui.section("Daemon")
    active = is_running("grimoire.pid")
    console.print(ui.kv_table([
        ("Estado", ui.daemon_badge(active)),
        ("PID file", Text("grimoire.pid", style="grimoire.muted")),
    ]))

    # ── Hints ───────────────────────────────────────────────────────────────
    if total_notes == 0:
        console.print()
        ui.tip("The vault is empty. Add [bold].md[/] notes and run [cyan]grimoire scan[/].")
    elif total_embeddings == 0:
        console.print()
        ui.tip("There are notes but none indexed. Try [cyan]grimoire scan --no-dry-run[/].")
    elif config.output.dry_run:
        console.print()
        ui.tip("You are in dry-run mode. Set [cyan]dry_run = false[/] in [cyan]grimoire.toml[/] to write.")


category_app = typer.Typer(
    help="🗂️  Manage the hierarchical category tree (History · Science · …).",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(category_app, name="category", rich_help_panel="Knowledge ops")


def _resolve_vault_path(vault_path: Path | None) -> Path:
    config = load_config()
    path = vault_path or Path(config.vault.path)
    if not path.exists():
        console.print(ui.error_panel(
            f"Vault not found at [bold]{path}[/]",
            title="Vault missing",
        ))
        raise typer.Exit(code=1)
    return path


@category_app.command("list")
def category_list(
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
):
    """📚 Show the full category tree with per-node note counts."""
    setup_logger()
    actual_vault_path = _resolve_vault_path(vault_path)
    vault_tax = load_taxonomy_from_vault(actual_vault_path)
    config = load_config()
    db = Database(config.memory.db_path)

    ui.command_header("category list", str(actual_vault_path))

    tree = vault_tax.categories
    if tree.is_empty():
        console.print(ui.info_panel(
            "No categories configured yet.\n"
            "Add the first one with [cyan]grimoire category add <path>[/].",
            title="Empty tree",
        ))
        return

    console.print()

    def render(node: str, depth: int) -> None:
        for child in tree.children(node):
            name = child.rsplit("/", 1)[-1]
            count = db.count_notes_under_category(child)
            indent = "  " * depth
            bullet = "◆" if depth == 0 else "↳"
            style = "grimoire.primary" if depth == 0 else "grimoire.accent"
            console.print(Text.assemble(
                (f"  {indent}{bullet} ", "grimoire.rune"),
                (name, style),
                ("  ", ""),
                (f"({count} notes)" if count else "(empty)", "grimoire.muted"),
            ))
            render(child, depth + 1)

    render("", 0)
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(tree.paths())} total categories", "grimoire.accent"),
    ))


@category_app.command("add")
def category_add(
    path: str = typer.Argument(..., help="Hierarchical path, e.g. 'Science/Physics/Quantum'"),
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
):
    """➕ Create a category (and any missing ancestors)."""
    setup_logger()
    actual_vault_path = _resolve_vault_path(vault_path)
    vault_tax = load_taxonomy_from_vault(actual_vault_path)

    ui.command_header("category add", path)

    try:
        created = vault_tax.categories.add(path)
    except ValueError as e:
        console.print(ui.error_panel(str(e), title="Invalid path"))
        raise typer.Exit(code=2)

    if not created:
        console.print(ui.info_panel(
            f"[bold]{path}[/] already exists in the tree.",
            title="No changes",
        ))
        return

    save_taxonomy_to_vault(actual_vault_path, vault_tax)
    console.print(ui.success_panel(
        f"Category [bold]{path}[] added and persisted in [cyan]taxonomy.yml[/].",
        title="Created",
    ))


@category_app.command("rm")
def category_rm(
    path: str = typer.Argument(..., help="Category to remove (includes children)"),
    force: bool = typer.Option(False, "--force", "-f", help="Delete even if notes are assigned"),
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
):
    """➖ Remove a category and its entire subtree."""
    setup_logger()
    actual_vault_path = _resolve_vault_path(vault_path)
    vault_tax = load_taxonomy_from_vault(actual_vault_path)
    config = load_config()
    db = Database(config.memory.db_path)

    ui.command_header("category rm", path)

    canonical = vault_tax.categories.resolve(path) or path
    notes_affected = db.count_notes_under_category(canonical)
    if notes_affected and not force:
        console.print(ui.warn_panel(
            f"[bold]{canonical}[/] has {notes_affected} notes assigned.\n"
            f"Use [cyan]--force[/] to delete it anyway "
            f"(the field will remain empty in the DB but not in the files).",
            title="Category in use",
        ))
        raise typer.Exit(code=1)

    removed = vault_tax.categories.remove(canonical)
    if not removed:
        console.print(ui.info_panel(
            f"[bold]{path}[/] does not exist in the tree.",
            title="Nothing to delete",
        ))
        return

    save_taxonomy_to_vault(actual_vault_path, vault_tax)
    console.print(ui.success_panel(
        f"Category [bold]{canonical}[/] removed from the tree.\n"
        f"{notes_affected} notes remain without category in the DB.",
        title="Deleted",
    ))



@category_app.command("notes")
def category_notes(
    path: str = typer.Argument(..., help="Category to inspect"),
    recursive: bool = typer.Option(True, "--recursive/--flat", help="Include descendants"),
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
):
    """📄 List notes filed under a category."""
    setup_logger()
    actual_vault_path = _resolve_vault_path(vault_path)
    vault_tax = load_taxonomy_from_vault(actual_vault_path)
    config = load_config()
    db = Database(config.memory.db_path)

    ui.command_header("category notes", path)

    canonical = vault_tax.categories.resolve(path)
    if canonical is None:
        console.print(ui.warn_panel(
            f"[bold]{path}[/] does not exist in the tree. Use [cyan]grimoire category list[/] to see available ones.",
            title="Unknown category",
        ))
        raise typer.Exit(code=1)

    rows = db.get_notes_by_category(canonical, recursive=recursive)
    if not rows:
        console.print(ui.info_panel(
            f"No notes under [bold]{canonical}[/].",
            title="Empty",
        ))
        return

    console.print()
    for _nid, npath, title in rows:
        try:
            display = Path(npath).relative_to(actual_vault_path)
        except ValueError:
            display = Path(npath).name
        console.print(Text.assemble(
            ("  • ", "grimoire.rune"),
            (title or str(display), "grimoire.primary"),
            ("  ", ""),
            (str(display), "grimoire.muted"),
        ))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(rows)} notes", "grimoire.accent"),
    ))


@app.command(rich_help_panel="System")
def preflight(
    check_git: bool = typer.Option(
        None,
        "--check-git/--no-check-git",
        help="Override whether to require a git repo (defaults to output.auto_commit).",
    ),
):
    """🛫 Validate config, Ollama connectivity and vault access before running the pipeline."""
    setup_logger()
    config = load_config()

    ui.command_header("preflight", config.vault.path)
    report = PreflightChecker(config).run(check_git=check_git)

    console.print()
    _render_preflight_report(report)
    console.print()

    if report.errors:
        console.print(ui.error_panel(
            f"{len(report.errors)} blocking error(s). Fix them before running `scan` or `ask`.",
            title="Preflight",
        ))
        raise typer.Exit(code=1)
    if report.warnings:
        console.print(ui.warn_panel(
            f"{len(report.warnings)} non-blocking warning(s).",
            title="Preflight",
        ))
        return
    console.print(ui.success_panel(
        "All checks passed. The pipeline is ready.",
        title="Preflight",
    ))


maintenance_app = typer.Typer(
    help="🧹 Database housekeeping: VACUUM, WAL checkpoint, tag purge.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(maintenance_app, name="maintenance", rich_help_panel="Operations")


def _fmt_bytes(n: int) -> str:
    """Compact human-readable size — KB/MB are plenty for a note DB."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


@maintenance_app.command("run")
def maintenance_run(
    skip_vacuum: bool = typer.Option(False, "--skip-vacuum", help="Don't rewrite the DB file"),
    skip_purge: bool = typer.Option(False, "--skip-purge", help="Don't drop orphan tags"),
    skip_checkpoint: bool = typer.Option(False, "--skip-checkpoint", help="Don't fold the -wal sidecar"),
):
    """🧹 Run the periodic housekeeping pipeline on demand."""
    setup_logger()
    config = load_config()
    db = Database(config.memory.db_path)

    # Clone the configured defaults, then apply the CLI overrides.
    from dataclasses import replace
    mcfg = replace(
        config.maintenance,
        vacuum=config.maintenance.vacuum and not skip_vacuum,
        purge_tags=config.maintenance.purge_tags and not skip_purge,
        wal_checkpoint=config.maintenance.wal_checkpoint and not skip_checkpoint,
    )

    ui.command_header("maintenance run", config.memory.db_path)
    report = MaintenanceRunner(db, mcfg).run(reason="manual")

    console.print()
    console.print(Text.assemble(
        ("  ◆ ", "grimoire.rune"),
        ("Tags purged", "grimoire.primary"),
        ("  ", ""),
        (str(report.tags_purged), "grimoire.accent"),
    ))
    if report.checkpoint:
        console.print(Text.assemble(
            ("  ◆ ", "grimoire.rune"),
            ("WAL checkpoint", "grimoire.primary"),
            ("  ", ""),
            (
                f"{report.checkpoint.get('checkpointed_frames', 0)}/"
                f"{report.checkpoint.get('log_frames', 0)} frames",
                "grimoire.accent",
            ),
        ))
    if report.vacuum:
        reclaimed = report.vacuum.get("reclaimed_bytes", 0)
        console.print(Text.assemble(
            ("  ◆ ", "grimoire.rune"),
            ("VACUUM", "grimoire.primary"),
            ("  ", ""),
            (f"{_fmt_bytes(reclaimed)} freed", "grimoire.accent"),
            ("  ", ""),
            (
                f"({_fmt_bytes(report.vacuum.get('before_bytes', 0))} → "
                f"{_fmt_bytes(report.vacuum.get('after_bytes', 0))})",
                "grimoire.muted",
            ),
        ))
    if report.skipped:
        console.print(Text.assemble(
            ("  ↳ ", "grimoire.muted"),
            (f"Skipped: {', '.join(report.skipped)}", "grimoire.muted"),
        ))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{report.duration_s:.2f}s", "grimoire.accent"),
    ))


chronicler_app = typer.Typer(
    help="📜 Track which notes have likely gone stale.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(chronicler_app, name="chronicler", rich_help_panel="Knowledge ops")


@chronicler_app.command("list")
def chronicler_list_cmd(
    decay: bool = typer.Option(False, "--decay/--no-decay", help="Annotate with cached LLM decay verdicts."),
):
    """📋 Show notes past their freshness window."""
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("chronicler list", config.vault.path)
    try:
        _do_chronicler_list(session, decay=decay)
    finally:
        session.close()


@chronicler_app.command("check")
def chronicler_check_cmd(
    path: str = typer.Argument(..., help="Note path (absolute, or relative to cwd / vault)"),
):
    """🔬 Run the LLM decay check on a single note."""
    setup_logger()
    config = load_config()
    _preflight_or_exit(config, check_git=False)
    session = Session(config)
    ui.command_header("chronicler check", path)
    try:
        _do_chronicler_check(session, path)
    finally:
        session.close()


@chronicler_app.command("verify")
def chronicler_verify_cmd(
    path: str = typer.Argument(..., help="Note path (absolute, or relative to cwd / vault)"),
):
    """✅ Mark a note as freshly verified."""
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("chronicler verify", path)
    try:
        _do_chronicler_verify(session, path)
    finally:
        session.close()


mirror_app = typer.Typer(
    help="🪞 The Black Mirror — surface contradictions across notes.",
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(mirror_app, name="mirror", rich_help_panel="Knowledge ops")


@mirror_app.callback(invoke_without_command=True)
def mirror_default(ctx: typer.Context):
    """🪞 List open contradictions when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("mirror", config.vault.path)
    try:
        _do_mirror_list(session)
    finally:
        session.close()


@mirror_app.command("scan")
def mirror_scan_cmd(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Neighbors per claim during pair-search."),
    full: bool = typer.Option(False, "--full", help="Re-extract every note (cold rebuild)."),
):
    """🔭 Extract claims and check pairs for contradictions."""
    setup_logger()
    config = load_config()
    _preflight_or_exit(config, check_git=False)
    session = Session(config)
    ui.command_header("mirror scan", config.vault.path)
    try:
        _do_mirror_scan(session, top_k=top_k, full=full)
    finally:
        session.close()


@mirror_app.command("show")
def mirror_show_cmd(
    contradiction_id: int = typer.Argument(..., help="Contradiction id (from `grimoire mirror`)."),
):
    """🔍 Show a contradiction in full, with surrounding context."""
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("mirror show", str(contradiction_id))
    try:
        _do_mirror_show(session, contradiction_id)
    finally:
        session.close()


@mirror_app.command("dismiss")
def mirror_dismiss_cmd(
    contradiction_id: int = typer.Argument(..., help="Contradiction id."),
):
    """🚫 Mark a contradiction as a false positive (won't be re-flagged)."""
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("mirror dismiss", str(contradiction_id))
    try:
        _do_mirror_dismiss(session, contradiction_id)
    finally:
        session.close()


@mirror_app.command("resolve")
def mirror_resolve_cmd(
    contradiction_id: int = typer.Argument(..., help="Contradiction id."),
):
    """✅ Mark a contradiction as resolved (you fixed one of the notes)."""
    setup_logger()
    config = load_config()
    session = Session(config)
    ui.command_header("mirror resolve", str(contradiction_id))
    try:
        _do_mirror_resolve(session, contradiction_id)
    finally:
        session.close()


@app.command(rich_help_panel="System")
def shell():
    """
    🌀 Open the interactive Grimoire shell.

    Keeps the database, embedder and LLM router warm across commands so
    consecutive [bold]ask[/]s skip the cold-start cost. Type [cyan]help[/]
    inside the shell, or Ctrl+D to leave.
    """
    setup_logger()
    config = load_config()
    ui.command_header("shell", config.vault.path)
    _preflight_or_exit(config, check_git=False)

    from grimoire.shell import GrimoireShell
    session = Session(config)
    try:
        GrimoireShell(session).run()
    finally:
        session.close()


if __name__ == "__main__":
    app()
