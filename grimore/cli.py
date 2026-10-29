"""
Command Line Interface (CLI) for Project Grimore.
This module defines the user commands for scanning vaults, connecting notes,
consulting the Oracle (RAG), and managing the background daemon.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.text import Text

from grimore.cognition.chunker import build_candidate_chunks
from grimore.cognition.connector import Connector
from grimore.cognition.embedder import Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.reembed import reembed_note
from grimore.cognition.privacy import ingest_services
from grimore.cognition.tagger import Tagger, tag_document
from grimore.ingest.extraction_profile import (
    index_profile, markdown_layout_stale, pdf_extraction_profile, pdf_retry_pending,
)
from grimore.ingest.pdf_reconcile import reconcile_unchanged_document
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics, iter_vault_documents
from grimore.memory.classification_overrides import apply_classification_override
from grimore.memory.db import Database
from grimore.temporal import derive_document_dates
from grimore.utils.hashing import sha256_file
from grimore.utils.index_progress import IndexProgressReporter
from grimore.memory.maintenance import MaintenanceRunner
from grimore.memory.taxonomy import load_taxonomy_from_vault, save_taxonomy_to_vault
from grimore.operations import (
    _do_ask,
    _do_chronicler_check,
    _do_chronicler_list,
    _do_chronicler_verify,
    _do_daemon_status,
    _do_dedupe,
    _do_distill,
    _do_eval,
    _do_graph_export,
    _do_migrate_embeddings,
    _do_mirror_dismiss,
    _do_mirror_list,
    _do_mirror_resolve,
    _do_mirror_scan,
    _do_mirror_show,
)
from grimore.output.frontmatter_writer import FrontmatterWriter
from grimore.output.change_journal import ChangeJournal
from grimore import __version__
from grimore.output.git_guard import GitGuard
from grimore.output.link_injector import LinkInjector
from grimore.session import Session
from grimore.utils import ui
from grimore.utils import matrix_cache
from grimore.utils.config import load_config, set_active_profile
from grimore.utils.logger import get_logger, setup_logger
from grimore.utils.preflight import PreflightChecker, PreflightReport
from grimore.utils.security import SecurityGuard


# Define the Typer application with metadata for the help screen
app = typer.Typer(
    help=f"🔮 [bold medium_purple3]Grimore v{__version__}[/] — Automated Knowledge Engine",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="[italic grey50]Sense the vault · surface connections · wake the Oracle[/]",
)

console = ui.console
logger = get_logger(__name__)


def _version_callback(value: bool) -> None:
    """Eager ``--version`` handler: print the installed version and exit."""
    if value:
        console.print(f"grimore {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the Grimore version and exit.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-P",
        help="Activate a \\[profiles.<name>] block from grimore.toml. "
             "Beats the GRIMORE_PROFILE environment variable.",
    ),
) -> None:
    """🔮 Grimore — Automated Knowledge Engine."""
    if profile:
        set_active_profile(profile)


def _mode_badge(is_dry_run: bool) -> Text:
    """Returns a visual badge indicating if the current execution is a dry run or live."""
    return ui.dry_run_badge() if is_dry_run else ui.live_mode_badge()


def _render_preflight_report(report: PreflightReport) -> None:
    """Print each preflight check with a ✓/⚠/✗ bullet and remediation guidance."""
    for c in report.checks:
        if c.ok:
            bullet = ("  ✓ ", "grimore.success")
        elif c.severity == "warning":
            bullet = ("  ⚠ ", "grimore.warning")
        else:
            bullet = ("  ✗ ", "grimore.danger")
        console.print(Text.assemble(
            bullet,
            (c.name, "grimore.primary"),
            ("  ", ""),
            (c.message, "grimore.muted"),
        ))
        if not c.ok and c.fix:
            for line in c.fix.splitlines():
                console.print(Text.assemble(
                    ("     ↳ ", "grimore.accent"),
                    (line, "grimore.muted"),
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
        ui.tip("Run [cyan]grimore preflight[/] to re-check without launching the scan.")
        raise typer.Exit(code=1)
    if report.has_warnings:
        console.print()
        _render_preflight_report(report)
        console.print()
    return report


def _preflight_for_recovery_ui(config) -> PreflightReport:
    """Report startup problems without hiding the UI that can explain them."""
    report = PreflightChecker(config).run(check_git=False)
    if not report.ok:
        console.print()
        console.print(ui.warn_panel(
            "Grimore is starting in recovery mode. Questions and indexing may "
            "remain unavailable until the checks below are resolved.",
            title="Preflight",
        ))
        _render_preflight_report(report)
        console.print()
    elif report.has_warnings:
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
    📖 Scan the vault, tag new or changed documents and index their embeddings.

    This is the primary ingestion command. It identifies new or modified files
    in every configured format (Markdown, PDF, EPUB, DOCX, ODT, RTF, HTML, TXT),
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
            f"Check the path or update [cyan]grimore.toml[/].",
            title="Vault missing",
        ))
        raise typer.Exit(code=1)

    # Preflight: catch "Ollama is down" or "model not pulled" before the first
    # file hits the LLM. Uses the effective auto_commit setting for git check.
    _preflight_or_exit(config)

    # Build the services one scan shares across every file. The embedder gets
    # the database as its cache so unchanged text is never embedded twice.
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

    # Identify files to process (filtering out ignored dirs and the
    # sidecar tree where Grimore stores its own non-MD metadata).
    vault_root = actual_vault_path.resolve()
    files = iter_vault_documents(
        actual_vault_path,
        config.vault.formats,
        config.vault.ignored_dirs,
        sidecar_dir=config.vault.sidecar_dir,
        sniff_magic=config.ingest.sniff_magic,
    )
    index_progress = IndexProgressReporter(vault_root, "scan")
    change_journal = None
    change_set_id = None
    if files and not is_dry_run:
        try:
            change_journal = ChangeJournal(vault_root)
            change_set_id = change_journal.start(
                "scan", details={"source": "cli", "file_count": len(files)},
            )
        except OSError as exc:
            # Snapshot history is an additional safety layer. Keep the
            # established scan behaviour when the private cache is unavailable
            # and make the missing protection visible in structured logs.
            logger.warning("change_journal_unavailable", error=str(exc))
            change_journal = None
    index_progress.start(
        total=len(files), dry_run=is_dry_run, change_set_id=change_set_id,
    )

    if not files:
        formats_label = ", ".join(config.vault.formats) or "md"
        console.print(ui.info_panel(
            f"No documents matching [bold]{formats_label}[/] found in "
            f"[cyan]{actual_vault_path}[/].",
            title="Empty vault",
        ))
        index_progress.finish()
        return

    stats = {
        "unchanged": 0, "processed": 0, "refreshed": 0, "partial": 0,
        "skipped": 0, "errors": 0, "chunks": 0,
    }
    current_pdf_profile = pdf_extraction_profile(config)
    ui.section(f"Scanning {len(files)} notes")

    with ui.progress_bar() as progress:
        task = progress.add_task("Scanning", total=len(files))
        for file in files:
            rel = file.relative_to(actual_vault_path)
            rel_label = str(rel)
            index_progress.begin_item(rel_label)
            progress.update(task, description=f"[grimore.muted]{rel}[/]")

            # Size+mtime is the O(1) fast path. Only read the whole
            # file when that cheap fingerprint changed or is not yet known.
            try:
                disk_stat = file.stat()
                stored_state = db.get_ingest_state(str(file))
                profile_changed = bool(
                    stored_state
                    and (stored_state[3] == "pdf" or file.suffix.lower() == ".pdf")
                    and stored_state[4] != current_pdf_profile
                )
                retry_pending = bool(
                    stored_state and pdf_retry_pending(stored_state[4], config)
                )
                temporal_backfill = db.note_timeline_needs_backfill(str(file))
                # An unchanged Markdown note indexed under an older chunk
                # layout is re-chunked below, without re-tagging.
                rechunk = markdown_layout_stale(stored_state)
                if not temporal_backfill and not profile_changed and not rechunk and stored_state is not None and stored_state[1:3] == (
                    disk_stat.st_size, disk_stat.st_mtime_ns,
                ):
                    stats["unchanged"] += 1
                    index_progress.record("skipped", rel_label, reason="unchanged_stat")
                    progress.advance(task)
                    continue
                file_hash = sha256_file(file)
            except OSError as e:
                stats["errors"] += 1
                logger.error("file_hash_failed", path=str(file), error=str(e))
                index_progress.record("failed", rel_label, reason="file_hash", error=str(e))
                progress.advance(task)
                continue
            stored_hash = stored_state[0] if stored_state else db.get_file_hash(str(file))
            if stored_hash == file_hash and not temporal_backfill and not profile_changed and not rechunk:
                if not is_dry_run:
                    db.update_file_state(
                        str(file), file_hash, disk_stat.st_size, disk_stat.st_mtime_ns,
                    )
                stats["unchanged"] += 1
                index_progress.record("skipped", rel_label, reason="unchanged_bytes")
                progress.advance(task)
                continue

            try:
                # Parse the document into a format-neutral ParsedNote.
                note = parser.parse_file(file, vault_root=vault_root, config=config)
            except Exception as e:
                stats["errors"] += 1
                logger.error("parse_failed", path=str(file), error=str(e))
                index_progress.record("failed", rel_label, reason="parse", error=str(e))
                progress.advance(task)
                continue
            document_dates = derive_document_dates(
                note.metadata, note.format, mtime_ns=disk_stat.st_mtime_ns,
            )

            # A note can opt out of processing with ``privacy: never_process``
            # in its frontmatter. It is skipped before any content reaches
            # the scanner, the LLM or the embedder.
            if note.metadata.get("privacy") == "never_process":
                stats["skipped"] += 1
                index_progress.record("skipped", rel_label, reason="privacy")
                progress.advance(task)
                continue

            # Identical extracted text does not imply identical PDF page
            # anchors. Refresh the sidecar and reconcile the indexed chunks
            # before publishing the new file fingerprint. A failed write or
            # embedding stays retryable on the next scan.
            if ((not profile_changed or retry_pending) and
                    db.get_content_hash_by_path(str(file)) == note.content_hash):
                extraction = extraction_diagnostics(note)
                refresh_chunks = note.format == "pdf" or (rechunk and note.format == "md")
                refresh_reason = (
                    "pdf_same_text_refresh" if note.format == "pdf" else "markdown_rechunk"
                )
                try:
                    refreshed = (
                        writer.refresh_extraction_metadata(
                            note, extraction,
                            vault_root=vault_root,
                            sidecar_dir=config.vault.sidecar_dir,
                            write_sidecars=config.vault.write_sidecars,
                            dry_run=is_dry_run,
                            change_journal=change_journal,
                        ) if extraction is not None else None
                    )
                    embedded_chunks = 0
                    if refresh_chunks and not is_dry_run:
                        sensitive_findings = security.scan_for_sensitive_data(note.content)
                        with ingest_services(
                            config, tagger, embedder,
                            sensitive=bool(sensitive_findings),
                        ) as (_note_tagger, note_embedder):
                            result = reconcile_unchanged_document(
                                db, note_embedder, note, config,
                                clean_content=security.sanitize_prompt(note.content),
                                file_hash=file_hash,
                                size_bytes=disk_stat.st_size,
                                mtime_ns=disk_stat.st_mtime_ns,
                                document_dates=document_dates,
                            )
                        embedded_chunks = result.embedded
                except Exception as exc:
                    stats["errors"] += 1
                    logger.error(f"{refresh_reason}_failed", path=str(file), error=str(exc))
                    index_progress.record(
                        "failed", rel_label, reason=refresh_reason, error=str(exc),
                    )
                    progress.advance(task)
                    continue
                if not refresh_chunks and not is_dry_run:
                    db.update_file_state(
                        str(file), file_hash, disk_stat.st_size, disk_stat.st_mtime_ns,
                    )
                    db.record_document_dates(str(file), **document_dates)
                if refresh_chunks:
                    stats["refreshed"] += 1
                    stats["chunks"] += embedded_chunks
                    if extraction and extraction["status"] == "partial":
                        stats["partial"] += 1
                    if is_dry_run and refreshed:
                        try:
                            refreshed["target"] = str(
                                Path(refreshed["target"]).resolve().relative_to(vault_root)
                            )
                        except (OSError, ValueError):
                            refreshed["target"] = Path(refreshed["target"]).name
                    index_progress.record(
                        "previewed" if is_dry_run else "processed", rel_label,
                        reason=refresh_reason, chunks=embedded_chunks,
                        changes=refreshed if is_dry_run else None,
                    )
                else:
                    stats["unchanged"] += 1
                    index_progress.record("skipped", rel_label, reason="unchanged_content")
                progress.advance(task)
                continue

            # Flag PII and secrets (API keys, tokens...). A hit is logged and
            # makes ingest_services hand this note only loopback clients, so
            # sensitive text is never sent to a remote backend.
            sensitive_findings = security.scan_for_sensitive_data(note.content)
            if sensitive_findings:
                logger.warning("sensitive_data_detected", path=str(file), types=sensitive_findings)

            try:
                with ingest_services(
                    config, tagger, embedder, sensitive=bool(sensitive_findings),
                ) as (note_tagger, note_embedder):
                    # Sanitize content before sending to LLM
                    clean_content = security.sanitize_prompt(note.content)

                    # Generate cognitive metadata: tags, summary, category, and links.
                    cognition_data = tag_document(note_tagger, note, clean_content)
                    cognition_data = apply_classification_override(
                        db, str(file), cognition_data,
                    )

                    metadata_updates = {
                        "tags": cognition_data["tags"],
                        "summary": cognition_data["summary"],
                        "category": cognition_data.get("category", ""),
                        "last_tagged": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
                        ),
                    }
                    extraction = extraction_diagnostics(note)
                    if extraction is not None:
                        metadata_updates["extraction"] = extraction
                    if "section_analysis" in cognition_data:
                        metadata_updates["section_analysis"] = cognition_data["section_analysis"]
                    elif "section_analysis" in note.metadata:
                        metadata_updates["section_analysis"] = None

                    if is_dry_run:
                        changes = writer.preview_metadata(
                            note,
                            metadata_updates,
                            vault_root=vault_root,
                            sidecar_dir=config.vault.sidecar_dir,
                            write_sidecars=config.vault.write_sidecars,
                        )
                        if changes:
                            try:
                                changes["target"] = str(
                                    Path(changes["target"]).resolve().relative_to(vault_root)
                                )
                            except (OSError, ValueError):
                                changes["target"] = Path(changes["target"]).name
                        stats["processed"] += 1
                        if extraction and extraction["status"] == "partial":
                            stats["partial"] += 1
                        logger.info(
                            "dry_run_preview",
                            path=str(file),
                            tags=cognition_data["tags"],
                            summary=cognition_data["summary"][:80],
                        )
                        index_progress.record(
                            "previewed", rel_label, changes=changes,
                            reason=("partial_extraction" if extraction and
                                    extraction["status"] == "partial" else None),
                        )
                        progress.advance(task)
                        continue

                    try:
                        embedded_chunks = 0
                        # Take the safety snapshot immediately before writing.
                        if config.output.auto_commit and git_guard.is_repo_ready():
                            git_guard.commit_pre_change(str(file))

                        # Write metadata to the source or its managed sidecar.
                        # For .md the writer mutates the source file in place;
                        # for everything else it materialises (or refreshes) a
                        # sidecar .md under <vault>/<sidecar_dir>/.
                        target = writer.write_metadata(
                            note,
                            metadata_updates,
                            vault_root=vault_root,
                            sidecar_dir=config.vault.sidecar_dir,
                            write_sidecars=config.vault.write_sidecars,
                            dry_run=False,
                            change_journal=change_journal,
                        )

                        # Markdown metadata is written into the source itself. Persist
                        # the post-write fingerprint so the next scan can take the O(1)
                        # path immediately instead of hashing once more.
                        latest_stat = file.stat()
                        if (latest_stat.st_size, latest_stat.st_mtime_ns) != (
                            disk_stat.st_size, disk_stat.st_mtime_ns,
                        ):
                            file_hash = sha256_file(file)
                        disk_stat = latest_stat

                        # Persist the indexed note and its metadata.
                        sidecar_target_str: Optional[str] = None
                        if note.format != "md" and target is not None:
                            sidecar_target_str = str(target)
                        note_id = db.upsert_note(
                            str(file), note.title, "",
                            format=note.format,
                            file_hash="",
                            sidecar_path=sidecar_target_str,
                            size_bytes=-1,
                            mtime_ns=-1,
                        )
                        db.update_last_tagged(str(file))
                        if note_id is not None:
                            db.upsert_tags(note_id, cognition_data["tags"])
                            db.set_note_category(note_id, cognition_data.get("category") or None)

                        # Vectorize content for semantic search.
                        # Section-aware path when the adapter provided structural
                        # anchors (PDF pages, EPUB / DOCX / HTML headings); falls
                        # back to body-only chunking for MD / TXT. The chunker
                        # runs first (cheap, deterministic); ``reembed_note`` then
                        # diffs the resulting chunks against stored hashes so
                        # untouched chunks skip the Ollama round-trip.
                        if note_id is not None:
                            candidate_chunks = build_candidate_chunks(
                                note, clean_content, note_embedder, config,
                            )
                            result = reembed_note(
                                db, note_embedder, note_id, candidate_chunks,
                                text_truncation=getattr(
                                    config.cognition, "chunk_store_chars", 500
                                ),
                            )
                            stats["chunks"] += result.embedded
                            embedded_chunks = result.embedded
                            if result.stored != len(candidate_chunks):
                                raise RuntimeError("embedding incomplete; file remains pending for retry")
                            db.complete_note_indexing(
                                note_id, note.content_hash, file_hash,
                                disk_stat.st_size, disk_stat.st_mtime_ns,
                                extraction_profile=index_profile(
                                    config, note.format, extraction,
                                ),
                                extraction=extraction,
                                **document_dates,
                            )

                        stats["processed"] += 1
                        if extraction and extraction["status"] == "partial":
                            stats["partial"] += 1
                        index_progress.record(
                            "processed", rel_label, chunks=embedded_chunks,
                            reason=("partial_extraction" if extraction and
                                    extraction["status"] == "partial" else None),
                        )
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error("processing_failed", path=str(file), error=str(e))
                        index_progress.record(
                            "failed", rel_label, reason="processing", error=str(e),
                        )

                    progress.advance(task)
            except Exception as exc:
                stats["errors"] += 1
                logger.error("processing_failed", path=str(file), error=str(exc))
                index_progress.record(
                    "failed", rel_label, reason="processing", error=str(exc),
                )
                progress.advance(task)

    index_progress.finish()
    if change_journal is not None:
        change_journal.finish(
            "completed_with_errors" if stats["errors"] else "completed",
        )

    # ── Summary ────────────────────────────────────────────────────────────
    summary_rows = [
        ("Processed",  Text(str(stats["processed"]), style="grimore.success")),
        ("PDFs refreshed", Text(str(stats["refreshed"]), style="grimore.accent")),
        ("Partial extraction", Text(str(stats["partial"]), style="grimore.warning")),
        ("Unchanged",  Text(str(stats["unchanged"]), style="grimore.muted")),
        ("Skipped",    Text(str(stats["skipped"]),   style="grimore.warning")),
        ("Errors",     Text(str(stats["errors"]),    style="grimore.danger" if stats["errors"] else "grimore.muted")),
        ("Chunks",     Text(str(stats["chunks"]),    style="grimore.accent")),
    ]
    console.print()
    console.print(ui.success_panel(ui.kv_table(summary_rows), title="Scan Summary"))

    if is_dry_run and (stats["processed"] > 0 or stats["refreshed"] > 0):
        ui.tip("This was a dry run. Run [cyan]grimore scan --no-dry-run[/] to apply changes.")


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
        help="Min. cosine similarity to propose a link, in \\[0.0, 1.0] (default: cognition.connect_threshold).",
        callback=_validate_threshold,
    ),
):
    """
    🕸️  Discover semantic connections between notes and inject \\[\\[wikilinks]].

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
    connector = Connector(
        db, embedder,
        vector_backend=config.cognition.vector_backend,
        matrix_cache_enabled=bool(getattr(
            config.cognition, "vec_matrix_cache", True)),
    )
    injector = LinkInjector()

    # One row per note -- the note's first chunk, which stands in for the
    # note as a whole when comparing notes against each other.
    note_vectors = db.get_first_chunk_vectors()
    if not note_vectors:
        console.print(ui.warn_panel(
            "No embeddings yet. Run [cyan]grimore scan --no-dry-run[/] first.",
            title="No vector memory",
        ))
        return

    change_journal = None
    if not is_dry_run:
        try:
            change_journal = ChangeJournal(Path(config.vault.path))
            change_journal.start("connect", details={"source": "cli"})
        except OSError as exc:
            logger.warning("change_journal_unavailable", error=str(exc))
            change_journal = None

    ui.section(f"Searching connections between {len(note_vectors)} notes")

    # The whole all-against-all sweep as one blocked matmul, instead of one
    # full-matrix multiply per note. Results are identical to calling
    # find_similar_notes in a loop; only the number of BLAS calls changes.
    note_ids = [nid for nid, _blob in note_vectors]
    # Straight from the stored bytes into one float32 matrix. Deserializing
    # each vector into a Python list first costs roughly 8x the memory (a list
    # slot plus a boxed float per element) and every one of them stays alive
    # for the whole sweep, which on a large vault is hundreds of megabytes.
    query_matrix = Embedder.buffer_to_matrix(
        b"".join(blob for _nid, blob in note_vectors),
        len(note_vectors),
        len(note_vectors[0][1]) // 4,
    )
    similar_by_note = connector.find_similar_notes_batch(
        query_matrix if query_matrix is not None
        else [embedder.deserialize_vector(blob) for _nid, blob in note_vectors],
        top_k=12, exclude_note_ids=note_ids, dedupe_by_note=True,
        with_text=False,
    )
    # Titles for every candidate above the threshold, in one query rather
    # than one per suggested link.
    wanted_titles = {
        s["note_id"]
        for hits in similar_by_note
        for s in hits if s["score"] > effective_threshold
    }
    titles = db.get_note_titles(wanted_titles)
    # path / title / format / sidecar for every note, fetched in one query
    # rather than two lookups per note.
    targets = db.get_note_targets(note_ids)

    total_links = 0
    unique_sources = 0

    with ui.progress_bar() as progress:
        task = progress.add_task("Connecting", total=len(note_vectors))
        for note_id, similar in zip(note_ids, similar_by_note, strict=True):
            progress.advance(task)

            target = targets.get(note_id)
            if not target:
                logger.warning("orphan_embedding", note_id=note_id)
                continue
            # Format-aware writeback target: .md hits the source file,
            # everything else lands in its sidecar. Rows whose ``format``
            # column is NULL come back as "md" (notes indexed before the
            # column existed are all Markdown).
            path, title, note_format, sidecar = target
            note_sidecar = Path(sidecar) if sidecar else None

            candidates = [s for s in similar if s["score"] > effective_threshold][:3]

            if not candidates:
                continue

            connections_to_inject = []
            bullet_lines = []
            for c in candidates:
                c_title = titles.get(c["note_id"])
                if not c_title:
                    continue
                bullet_lines.append(f"  ↳ [cyan]{c_title}[/]  [grimore.muted](score {c['score']:.2f})[/]")
                connections_to_inject.append({"title": c_title, "reason": "High semantic similarity."})

            if connections_to_inject:
                unique_sources += 1
                total_links += len(connections_to_inject)
                progress.console.print(Text.assemble(
                    ("◆ ", "grimore.rune"),
                    (str(title), "grimore.primary"),
                ))
                for line in bullet_lines:
                    progress.console.print(line)

                # Inject the discovered links into the right markdown
                # target: source for .md, sidecar for everything else.
                injector.inject_for(
                    source_path=Path(path),
                    format=note_format,
                    connections=connections_to_inject,
                    sidecar_path=note_sidecar,
                    vault_root=Path(config.vault.path),
                    sidecar_dir=config.vault.sidecar_dir,
                    dry_run=is_dry_run,
                    change_journal=change_journal,
                )

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Connected notes", Text(str(unique_sources), style="grimore.success")),
            ("Suggested links", Text(str(total_links), style="grimore.accent")),
        ]),
        title="Connection summary",
    ))
    if change_journal is not None:
        change_journal.finish()


@app.command(rich_help_panel="Knowledge ops")
def ask(
    question: str = typer.Argument(..., help="The question to ask the Oracle"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Context fragments to retrieve"),
    export: Path = typer.Option(None, "--export", "-e", help="Save the answer as a markdown note"),
    category: str = typer.Option(
        None, "--category", "-c",
        help="Only search notes in this category (descendants included).",
    ),
    tag: list[str] = typer.Option(
        None, "--tag", "-t",
        help="Only search notes carrying this tag. Repeat to require several.",
    ),
    fmt: list[str] = typer.Option(
        None, "--format", "-f",
        help="Only search notes of this format (md, pdf, …). Repeat to allow several.",
    ),
):
    """
    🔮 Consult the Grimore Oracle about your vault's knowledge.

    This is a Retrieval-Augmented Generation (RAG) system. It searches for relevant
    note fragments, provides them as context to the LLM, and generates an answer
    with citations to your own notes.

    The filters narrow which notes are searched at all, so the Oracle can only
    cite from what survives them. They combine with AND, including repeated
    --tag: "--tag a --tag b" means notes carrying both.
    """
    setup_logger()
    config = load_config()

    ui.command_header("ask", "Consulting the Oracle")

    # ask doesn't write to the vault, so don't nag about missing git.
    _preflight_or_exit(config, check_git=False)

    session = Session(config)
    filter_note_ids = session.db.resolve_note_filter(
        category=category, tags=tag, formats=fmt,
    )
    if filter_note_ids is not None and not filter_note_ids:
        # An empty result is not the same as no filter: searching the whole
        # vault here would silently answer from notes the user excluded.
        console.print(ui.warn_panel(
            "No notes match those filters, so there is nothing to search.",
            title="Empty filter",
        ))
        return
    if filter_note_ids is not None:
        console.print(Text.assemble(
            ("  ◆ ", "grimore.rune"),
            (f"Searching {len(filter_note_ids)} filtered note(s)", "grimore.muted"),
        ))
    _do_ask(session, question, top_k=top_k, export=export,
            filter_note_ids=filter_note_ids)


@app.command(rich_help_panel="Knowledge ops")
def eval(
    golden: Path = typer.Option(
        Path("eval/grimore_golden.yaml"), "--golden", "-g",
        help="Path to the golden YAML (see eval/grimore_golden.yaml for the schema).",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Context fragments fed to the answer per ask."),
    retrieval_k: int = typer.Option(
        10, "--retrieval-k",
        help="Depth of the ranked pool scored for Hit@k / MRR, independent of --top-k.",
    ),
    retrieval_only: bool = typer.Option(
        False, "--retrieval-only",
        help="Score ranking only — skip answer generation. Fast, deterministic, CI-friendly.",
    ),
    baseline: bool = typer.Option(
        False, "--baseline",
        help="Run twice — hybrid (RRF) vs dense-only — and show the per-metric delta.",
    ),
    judge: bool = typer.Option(
        True, "--judge/--no-judge",
        help="Use the local LLM to rate answer relevance. Off when Ollama is busy or offline.",
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", help="Write the full report as JSON to this path.",
    ),
    history: Optional[Path] = typer.Option(
        None, "--history",
        help="Append a one-line run record to this JSONL ledger for trend tracking.",
    ),
    compare: Optional[Path] = typer.Option(
        None, "--compare",
        help="Diff against a previous --export run; exits non-zero on any regression.",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint",
        help="Save each finished case to this JSONL file and resume from it on "
             "rerun; cases whose generation failed are asked again.",
    ),
    json_logs: bool = typer.Option(False, "--json", help="Emit logs in JSON format."),
):
    """
    📊 Evaluate retrieval + answer quality against a golden YAML.

    Reports Hit@1/Hit@3, MRR, recall@k, citation validity and coverage,
    keyword recall,
    an LLM-as-judge relevance score, and p50/p95 latency. Read-only (no vault
    writes). Skip the judge with --no-judge for fast offline runs, or use
    --retrieval-only to score ranking alone (no answer LLM) on every change.
    Add --baseline to quantify the hybrid-fusion (RRF) uplift over dense-only.
    Track trends with --history <ledger.jsonl> and gate CI with
    --compare <previous.json> (non-zero exit on regression). Long runs can
    survive interruptions with --checkpoint <run.jsonl>.
    """
    setup_logger(json_format=json_logs)
    config = load_config()

    # eval is read-only over the vault; skip the git-clean nag.
    _preflight_or_exit(config, check_git=False)

    session = Session(config)
    _do_eval(
        session, golden, top_k=top_k, retrieval_k=retrieval_k,
        retrieval_only=retrieval_only, baseline=baseline, judge=judge,
        export=export, history=history, compare=compare, checkpoint=checkpoint,
    )


@app.command(rich_help_panel="Daemon")
def daemon(
    action: str = typer.Argument("run", help="Action: run · start · stop · status"),
    json_logs: bool = typer.Option(False, "--json", help="Emit logs in JSON format"),
):
    """
    🧿 Manage the Grimore daemon (foreground / background / status).

    The daemon watches for file changes in real-time and automatically
    processes them after a debounce period.
    """
    from grimore.utils.system import is_running, start_daemon_background, stop_daemon
    from grimore.utils.paths import daemon_lock_path, daemon_log_path

    # Both paths flow through platformdirs so the CLI, the shell history file
    # and the daemon's own bookkeeping all agree on where state lives.
    pid_file = str(daemon_lock_path())
    log_file = str(daemon_log_path())

    ui.command_header("daemon", action)

    if action == "run":
        # Foreground: this process becomes the daemon and blocks in
        # instance.start() until Ctrl-C.
        setup_logger(json_format=json_logs)
        config = load_config()
        from grimore.daemon import GrimoreDaemon

        _preflight_or_exit(config)

        console.print(ui.info_panel(
            "The daemon will run in the foreground. [bold]Ctrl-C[/] to stop.",
            title="Foreground mode",
        ))
        # Pass the resolved cache path explicitly so foreground and background
        # forms always lock the same file (otherwise the daemon's default path
        # could drift if platformdirs evaluates differently in some env).
        instance = GrimoreDaemon(config, pid_file=pid_file)
        instance.start()
    elif action == "start":
        # Background: spawn a detached `python -m grimore daemon` process and
        # return. The daemon writes its own PID file, which `stop` and
        # `status` use to find it again.
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
        # Ask the process named in the PID file to exit; stop_daemon
        # escalates to a hard kill if it does not stop in time.
        if not is_running(pid_file):
            console.print(ui.info_panel("No daemon running.", title="Nothing to stop"))
            return
        stop_daemon(pid_file)
        console.print(ui.success_panel("Daemon stopped.", title="Stop"))
    elif action == "status":
        # Rich panel: PID, uptime, debounce config, last few events from
        # daemon.log. Falls back to the bare "not running" message when
        # there's no live process.
        _do_daemon_status(load_config(), pid_file=pid_file, log_file=log_file)
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
            "No tags recorded yet. Run [cyan]grimore scan --no-dry-run[/] first.",
            title="No tags",
        ))
        return

    console.print()
    console.print(ui.tag_frequency_table(rows))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(rows)} tags", "grimore.accent"),
        ("  ·  ", "grimore.muted"),
        (f"{db.get_tag_count()} unique in use", "grimore.muted"),
    ))


@app.command(rich_help_panel="Knowledge ops")
def dedupe(
    threshold: float = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Min. cosine similarity to report a near-duplicate pair, in "
             "[0.0, 1.0] (default: cognition.dedupe_threshold).",
        callback=_validate_threshold,
    ),
    limit: int = typer.Option(
        30, "--limit", "-n",
        help="Max near-duplicate pairs to report (best scores first).",
    ),
    export: Path = typer.Option(
        None, "--export", "-e",
        help="Write the full report as JSON to this path.",
    ),
):
    """
    👯 Find duplicate notes — exact copies and semantic near-duplicates.

    Report-only: nothing in the vault or the index is modified. Exact
    duplicates share a content hash; near-duplicates are note pairs whose
    mean chunk vectors exceed the cosine threshold. Deterministic and
    LLM-free, so it works while Ollama is busy or offline.
    """
    setup_logger()
    config = load_config()
    session = Session(config)
    try:
        _do_dedupe(session, threshold=threshold, limit=limit, export=export)
    finally:
        session.close()


@app.command("migrate-embeddings", rich_help_panel="System")
def migrate_embeddings(
    target_model: Optional[str] = typer.Argument(
        None,
        help="The new embedding model name (e.g. 'qwen3-embedding:0.6b').",
    ),
    abort: bool = typer.Option(
        False, "--abort",
        help="Drop the shadow table and mark any in-flight migration aborted.",
    ),
    status: bool = typer.Option(
        False, "--status",
        help="Show the current migration's progress and exit.",
    ),
    write_config: bool = typer.Option(
        True, "--write-config/--no-write-config",
        help="On success, update \\[cognition].model_embeddings_local in grimore.toml.",
    ),
):
    """
    🔄 Hot-swap the embedding model without dropping the index.

    Re-embeds every chunk under the new model into a shadow table while
    the live ``embeddings`` keeps serving queries. When complete, swaps
    atomically in a single transaction; on interruption, rerun the same
    command to resume from the last completed row.
    """
    setup_logger()
    config = load_config()

    if status or abort:
        db = Database(config.memory.db_path)
        _do_migrate_embeddings(
            config, db,
            target_model=target_model or "",
            abort=abort,
            status_only=status,
            write_config=write_config,
        )
        return

    if not target_model:
        console.print(ui.error_panel(
            "Missing target model. Pass it as an argument, e.g.\n"
            "  [cyan]grimore migrate-embeddings qwen3-embedding:0.6b[/]\n"
            "Or check progress: [cyan]grimore migrate-embeddings --status[/].",
            title="migrate-embeddings",
        ))
        raise typer.Exit(code=1)

    _preflight_or_exit(config, check_git=False)
    db = Database(config.memory.db_path)
    _do_migrate_embeddings(
        config, db,
        target_model=target_model,
        abort=False,
        status_only=False,
        write_config=write_config,
        progress_factory=ui.progress_bar,
    )


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

    # Scan disk for existing files to identify orphans in DB.
    # Honours config.vault.formats so non-Markdown docs aren't pruned
    # just because we enumerate with the wrong extension.
    existing_paths = {
        str(f) for f in iter_vault_documents(
            actual_vault_path,
            config.vault.formats,
            config.vault.ignored_dirs,
            sidecar_dir=config.vault.sidecar_dir,
            sniff_magic=config.ingest.sniff_magic,
        )
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
            ("  ✗ ", "grimore.danger"),
            (str(display), "grimore.muted"),
        ))
    if len(stale) > 50:
        console.print(Text(f"  … and {len(stale) - 50} more", style="grimore.muted"))

    if dry_run:
        console.print()
        ui.tip("Dry run complete. Run [cyan]grimore prune --no-dry-run[/] to delete.")
        return

    # Drop the index rows (and their chunks/embeddings) for notes missing on
    # disk, then the tags that no remaining note references.
    removed = db.prune_missing_notes(existing_paths)
    purged = db.purge_unused_tags()

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Notes deleted", Text(str(removed), style="grimore.success")),
            ("Tags purged",   Text(str(purged),  style="grimore.accent")),
        ]),
        title="Prune complete",
    ))


def _render_status_dashboard(config) -> None:
    """Render the status dashboard against a given config.

    Split from the Typer entrypoint so callers with a live in-memory
    config (e.g. the interactive shell after a ``models chat`` swap)
    can pass it through rather than going back to disk via
    ``load_config()``.
    """
    from grimore.utils.paths import daemon_lock_path
    from grimore.utils.connectivity import cognition_connectivity
    from grimore.utils.system import is_running

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

    from grimore import __version__

    ui.section("Vault")
    profile_name = getattr(config, "active_profile", None)
    rows = [
        ("Version",      Text(__version__, style="grimore.muted")),
    ]
    if profile_name:
        rows.append(("Profile", Text(profile_name, style="grimore.accent")))
    rows.extend([
        ("Path",         Text(config.vault.path, style="grimore.accent")),
        ("Notes",        ui.coverage_bar(tagged_notes, total_notes)),
        ("Chunks",       Text(str(total_embeddings), style="grimore.accent")),
        ("Cache",        Text(f"{cached_embeddings} vectors", style="grimore.accent")),
        ("Mode",         _mode_badge(config.output.dry_run)),
        ("Auto-commit",  Text("yes" if config.output.auto_commit else "no", style="grimore.accent")),
    ])
    console.print(ui.kv_table(rows))

    ui.section("Cognition")
    backend_name = getattr(config.cognition, "llm_backend", "ollama") or "ollama"
    connectivity = cognition_connectivity(config)
    console.print(ui.kv_table([
        ("Backend",      Text(backend_name,                            style="grimore.accent")),
        ("LLM",          Text(config.cognition.model_llm_local,        style="grimore.accent")),
        ("Embeddings",   Text(config.cognition.model_embeddings_local, style="grimore.accent")),
        ("Unique tags",  Text(str(unique_tags),                        style="grimore.accent")),
        ("Categories",   Text(f"{len(category_rows)} active · {categorised_notes} notes",
                              style="grimore.accent")),
        ("Execution",    Text(connectivity["mode"],
                              style="grimore.warning" if connectivity["mode"] != "local" else "grimore.success")),
        ("Chat route",   Text(connectivity["chat"]["mode"], style="grimore.accent")),
        ("Embed route",  Text(connectivity["embeddings"]["mode"], style="grimore.accent")),
        ("Remote gate",  Text("allowed" if connectivity["remote_allowed"] else "blocked",
                              style="grimore.warning" if connectivity["remote_allowed"] else "grimore.success")),
    ]))

    ui.section("Daemon")
    pid_file = str(daemon_lock_path())
    active = is_running(pid_file)
    console.print(ui.kv_table([
        ("Status", ui.daemon_badge(active)),
        ("PID file", Text(pid_file, style="grimore.muted")),
    ]))

    # ── Hints ───────────────────────────────────────────────────────────────
    if total_notes == 0:
        console.print()
        formats_label = ", ".join(config.vault.formats) or "md"
        ui.tip(f"The vault is empty. Add documents ([bold]{formats_label}[/]) and run [cyan]grimore scan[/].")
    elif total_embeddings == 0:
        console.print()
        ui.tip("There are documents but none indexed. Try [cyan]grimore scan --no-dry-run[/].")
    elif config.output.dry_run:
        console.print()
        ui.tip("You are in dry-run mode. Set [cyan]dry_run = false[/] in [cyan]grimore.toml[/] to write.")


@app.command(rich_help_panel="System")
def status():
    """
    🧭 Full dashboard: vault, cognition, daemon.

    Displays a high-level overview of the system state, including
    indexing progress, model configurations, and daemon status.
    """
    _render_status_dashboard(load_config())


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
            "Add the first one with [cyan]grimore category add <path>[/].",
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
            style = "grimore.primary" if depth == 0 else "grimore.accent"
            console.print(Text.assemble(
                (f"  {indent}{bullet} ", "grimore.rune"),
                (name, style),
                ("  ", ""),
                (f"({count} notes)" if count else "(empty)", "grimore.muted"),
            ))
            render(child, depth + 1)

    render("", 0)
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(tree.paths())} total categories", "grimore.accent"),
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
        raise typer.Exit(code=2) from None

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
            f"[bold]{path}[/] does not exist in the tree. Use [cyan]grimore category list[/] to see available ones.",
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
            ("  • ", "grimore.rune"),
            (title or str(display), "grimore.primary"),
            ("  ", ""),
            (str(display), "grimore.muted"),
        ))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(rows)} notes", "grimore.accent"),
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


pilot_app = typer.Typer(
    help="🧭 Prepare and validate a privacy-preserving individual-user pilot.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(pilot_app, name="pilot", rich_help_panel="Operations")


@pilot_app.command("init")
def pilot_init(
    output: Path = typer.Argument(..., help="New YAML file for the pilot record."),
    participants: int = typer.Option(
        5, "--participants", "-n", help="Planned cohort size (5–10).",
    ),
    study_id: Optional[str] = typer.Option(
        None, "--study-id", help="Pseudonymous study identifier.",
    ),
) -> None:
    """Create a private two-week pilot worksheet with three tasks per person."""
    from grimore.pilot import write_pilot_study

    ui.command_header("pilot init", str(output))
    try:
        target = write_pilot_study(
            output, participant_count=participants, study_id=study_id,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        console.print(ui.error_panel(str(exc), title="Pilot record not created"))
        raise typer.Exit(code=1) from exc
    console.print(ui.success_panel(
        f"Created {target} with {participants} pseudonymous participant records.\n"
        "Complete consent, three real task goals, success criteria, and the "
        "usual method for every participant; then run `grimore pilot check`.",
        title="Pilot worksheet ready",
    ))


@pilot_app.command("check")
def pilot_check(
    study_file: Path = typer.Argument(..., help="Pilot YAML to validate."),
) -> None:
    """Check cohort size, consent, privacy rules, and the three-task contract."""
    from grimore.pilot import check_pilot_study, is_private_file, load_pilot_study

    ui.command_header("pilot check", str(study_file))
    try:
        result = check_pilot_study(load_pilot_study(study_file))
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(ui.error_panel(str(exc), title="Invalid pilot record"))
        raise typer.Exit(code=1) from exc
    console.print(ui.kv_table([
        ("Participants", result.participant_count),
        ("Ready", result.ready_participants),
        ("Agreed tasks", result.task_count),
        ("Journey stages observed", result.observed_stages),
        ("Full journeys observed", result.observed_journeys),
        ("Full journeys completed", result.completed_journeys),
        ("Private file", "yes" if is_private_file(study_file) else "check permissions"),
    ]))
    for warning in result.warnings:
        console.print(Text.assemble(("  ⚠ ", "grimore.warning"), warning))
    if result.errors:
        details = Text("\n".join(f"• {error}" for error in result.errors))
        console.print(ui.error_panel(details, title="Pilot is not ready"))
        raise typer.Exit(code=1)
    console.print(ui.success_panel(
        "Every participant has recorded consent and exactly three agreed tasks "
        "with success criteria and a usual-method baseline.",
        title="Pilot intake ready",
    ))


@pilot_app.command("observe")
def pilot_observe(
    study_file: Path = typer.Argument(..., help="Pilot YAML to update."),
    participant_id: str = typer.Argument(..., help="Opaque participant code, e.g. P01."),
    stage: str = typer.Argument(
        ...,
        help="configuration | first_answer | source_check | document_review",
    ),
    outcome: str = typer.Option(
        ..., "--outcome", help="Observed outcome: completed | blocked.",
    ),
    assistance: str = typer.Option(
        ...,
        "--assistance",
        help="Facilitator help: none | prompt | intervention.",
    ),
    observed_at: Optional[str] = typer.Option(
        None, "--at", help="ISO-8601 timestamp; defaults to current UTC time.",
    ),
    replace: bool = typer.Option(
        False, "--replace", help="Correct a previously observed stage.",
    ),
) -> None:
    """Record one privacy-safe stage of the observed end-to-end journey."""
    from grimore.pilot import record_pilot_observation

    ui.command_header("pilot observe", f"{participant_id} · {stage}")
    try:
        backup = record_pilot_observation(
            study_file,
            participant_id=participant_id,
            stage=stage,
            outcome=outcome,
            assistance=assistance,
            observed_at=observed_at,
            replace=replace,
        )
    except (FileNotFoundError, LookupError, OSError, TypeError, ValueError) as exc:
        console.print(ui.error_panel(str(exc), title="Observation not recorded"))
        raise typer.Exit(code=1) from exc
    console.print(ui.success_panel(
        f"Recorded {stage} for {participant_id}.\n"
        f"Previous bytes are recoverable from {backup}.",
        title="Pilot observation saved",
    ))


@pilot_app.command("restore-backup")
def pilot_restore_backup(
    study_file: Path = typer.Argument(..., help="Pilot YAML whose backup should be restored."),
) -> None:
    """Restore the last pre-change copy; the current file becomes the backup."""
    from grimore.pilot import restore_pilot_backup

    ui.command_header("pilot restore-backup", str(study_file))
    try:
        restored = restore_pilot_backup(study_file)
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(ui.error_panel(str(exc), title="Pilot backup not restored"))
        raise typer.Exit(code=1) from exc
    console.print(ui.success_panel(
        f"Restored {restored}. The replaced bytes remain in its .bak file.",
        title="Pilot backup restored",
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

    # The matrix cache survives VACUUM (ids are preserved), so housekeeping
    # leaves it alone -- unless the feature is off, in which case the file is
    # dead weight the size of every vector in the vault.
    if not bool(getattr(config.cognition, "vec_matrix_cache", True)):
        matrix_cache.clear(config.memory.db_path)

    console.print()
    console.print(Text.assemble(
        ("  ◆ ", "grimore.rune"),
        ("Tags purged", "grimore.primary"),
        ("  ", ""),
        (str(report.tags_purged), "grimore.accent"),
    ))
    if report.checkpoint:
        console.print(Text.assemble(
            ("  ◆ ", "grimore.rune"),
            ("WAL checkpoint", "grimore.primary"),
            ("  ", ""),
            (
                f"{report.checkpoint.get('checkpointed_frames', 0)}/"
                f"{report.checkpoint.get('log_frames', 0)} frames",
                "grimore.accent",
            ),
        ))
    if report.vacuum:
        reclaimed = report.vacuum.get("reclaimed_bytes", 0)
        console.print(Text.assemble(
            ("  ◆ ", "grimore.rune"),
            ("VACUUM", "grimore.primary"),
            ("  ", ""),
            (f"{_fmt_bytes(reclaimed)} freed", "grimore.accent"),
            ("  ", ""),
            (
                f"({_fmt_bytes(report.vacuum.get('before_bytes', 0))} → "
                f"{_fmt_bytes(report.vacuum.get('after_bytes', 0))})",
                "grimore.muted",
            ),
        ))
    if report.skipped:
        console.print(Text.assemble(
            ("  ↳ ", "grimore.muted"),
            (f"Skipped: {', '.join(report.skipped)}", "grimore.muted"),
        ))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{report.duration_s:.2f}s", "grimore.accent"),
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
    contradiction_id: int = typer.Argument(..., help="Contradiction id (from `grimore mirror`)."),
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


graph_app = typer.Typer(
    help="🕸️  Export the vault's note graph (wikilinks + suggestions + contradictions).",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph", rich_help_panel="Knowledge ops")


@graph_app.command("export")
def graph_export_cmd(
    output: Path = typer.Argument(..., help="Where to write the export."),
    fmt: str = typer.Option(
        "json", "--format", "-f",
        help="Output format: json | dot | obsidian-canvas.",
    ),
    suggested: bool = typer.Option(
        True, "--suggested/--no-suggested",
        help="Include Connector-suggested edges (slower; requires embeddings).",
    ),
    suggested_top: int = typer.Option(
        3, "--suggested-top",
        help="Top-K semantic neighbours per note when --suggested is on.",
    ),
    suggested_threshold: float = typer.Option(
        0.7, "--suggested-threshold",
        help="Drop suggested edges with cosine below this score.",
    ),
):
    """🕸️ Write the vault's note graph in the chosen format."""
    setup_logger()
    config = load_config()
    session = Session(config)
    try:
        _do_graph_export(
            session,
            output=output,
            fmt=fmt,
            include_suggested=suggested,
            suggested_top=suggested_top,
            suggested_threshold=suggested_threshold,
        )
    finally:
        session.close()


@app.command(rich_help_panel="Knowledge ops")
def distill(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Distill notes carrying this tag."),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Distill notes under this category (recursive)."),
    passages: int = typer.Option(3, "--passages", "-p", help="Top-K passages per source note."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build the synthesis but don't write the file."),
):
    """
    🧪 Distill notes that share a tag or category into a single reference note.

    The output lands in [cyan]<vault>/_synthesis/[/] with
    [cyan]grimore_generated: true[/] in frontmatter so subsequent
    distills don't re-include their own outputs.
    """
    setup_logger()
    config = load_config()
    _preflight_or_exit(config, check_git=False)
    session = Session(config)
    selector_label = f"--tag {tag}" if tag else (f"--category {category}" if category else "")
    ui.command_header("distill", selector_label or config.vault.path)
    try:
        _do_distill(
            session,
            tag=tag,
            category=category,
            passages_per_note=passages,
            dry_run=dry_run,
        )
    finally:
        session.close()


@app.command(rich_help_panel="System")
def shell():
    """
    🌀 Open the interactive Grimore shell.

    Keeps the database, embedder and LLM router warm across commands so
    consecutive [bold]ask[/]s skip the cold-start cost. Type [cyan]help[/]
    inside the shell, or Ctrl+D to leave.
    """
    setup_logger()
    config = load_config()
    ui.command_header("shell", config.vault.path)
    _preflight_or_exit(config, check_git=False)

    from grimore.shell import GrimoreShell
    session = Session(config)
    try:
        GrimoreShell(session).run()
    finally:
        session.close()


@app.command(rich_help_panel="Integrations")
def mcp(
    json_logs: bool = typer.Option(
        True, "--json/--no-json",
        help="Emit logs as JSON (default). MCP clients spawn the process "
             "and capture stderr, so JSON logs are easier to grep later.",
    ),
):
    """
    🔌 Run the Grimore MCP (Model Context Protocol) server over stdio.

    Exposes read-only tools — [bold]grimore_ask[/], [bold]grimore_search[/],
    [bold]grimore_get_note[/], [bold]grimore_get_source[/],
    [bold]grimore_review_inbox[/], [bold]grimore_library_info[/],
    [bold]grimore_connect[/], and [bold]grimore_list_categories[/] — to MCP-aware
    clients like Claude Desktop, Cursor, and Zed. The vault stays read-only; scan and migrate
    operations remain on the CLI.

    Typical Claude Desktop setup (in [cyan]claude_desktop_config.json[/]):

        \b
        {
          "mcpServers": {
            "grimore": {
              "command": "grimore",
              "args": ["mcp"]
            }
          }
        }
    """
    setup_logger(json_format=json_logs, stream=sys.stderr)
    config = load_config()
    # Metadata/source/review queries need neither chat nor embeddings, so
    # model services are not probed at startup: an editor first checks vault
    # identity and remote-query consent. Only the local library is validated.
    vault_root = Path(config.vault.path)
    if not vault_root.is_dir() or not os.access(vault_root, os.R_OK | os.X_OK):
        sys.stderr.write("MCP: the configured vault is not an accessible directory. Check grimore.toml.\n")
        raise typer.Exit(code=1)

    from grimore.mcp_server import run_stdio
    session = Session(config)
    try:
        run_stdio(session=session)
    finally:
        session.close()


@app.command(rich_help_panel="Integrations")
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Bind address. Default is loopback only. Use a non-loopback "
             "address with --allow-lan to expose on a LAN.",
    ),
    port: int = typer.Option(8000, "--port", "-p"),
    allow_lan: bool = typer.Option(
        False, "--allow-lan",
        help="Required when --host is non-loopback. Acknowledges the vault "
             "will be reachable from other machines on the network.",
    ),
    api_token: Optional[str] = typer.Option(
        None, "--api-token",
        help="Bearer token required on every non-loopback request (all "
             "methods) and on protected writes from every client. Read from "
             "env GRIMORE_API_TOKEN if unset. Required "
             "for any non-loopback bind. Prefer the env var: a token on "
             "the command line is visible to other local users via ps.",
        envvar="GRIMORE_API_TOKEN",
    ),
    strict_token: bool = typer.Option(
        False, "--strict-token",
        help="Require the bearer token from loopback clients too (needs "
             "--api-token). Recommended on Android/Termux, where any app "
             "on the device can reach localhost ports. Enter the token in "
             "the bundled UI before using API-backed controls.",
    ),
    cors_origin: Optional[str] = typer.Option(
        None, "--cors-origin",
        help="Allow this origin via CORS. No wildcards. Off by default.",
    ),
    json_logs: bool = typer.Option(False, "--json", help="Emit logs in JSON format."),
):
    """
    🌐 Run the Grimore HTTP API and minimal web UI.

    Exposes the MCP read surface plus token-protected, scoped setup and
    indexing operations, with a single-page UI at [cyan]/[/]. Loopback by
    default — explicitly pass
    [cyan]--allow-lan[/] together with a non-loopback [cyan]--host[/] to
    expose on a LAN, in which case [cyan]--api-token[/] becomes
    mandatory.
    """
    setup_logger(json_format=json_logs)
    config = load_config()
    _preflight_for_recovery_ui(config)

    try:
        from grimore.api.app import build_app
    except ImportError:
        console.print(ui.error_panel(
            "Starlette / uvicorn aren't installed.\n"
            "Install the extra: [cyan]pip install 'grimore[serve]'[/].",
            title="serve unavailable",
        ))
        raise typer.Exit(code=1) from None
    try:
        import uvicorn  # noqa: F401  (validate before starting the panel)
    except ImportError:
        console.print(ui.error_panel(
            "uvicorn isn't installed.\n"
            "Install the extra: [cyan]pip install 'grimore[serve]'[/].",
            title="serve unavailable",
        ))
        raise typer.Exit(code=1) from None

    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not allow_lan:
        console.print(ui.error_panel(
            f"[bold]--host {host}[/] is non-loopback; pass [cyan]--allow-lan[/] "
            f"to confirm and provide [cyan]--api-token[/] for auth.",
            title="LAN bind requires --allow-lan",
        ))
        raise typer.Exit(code=1)
    if not is_loopback and not api_token:
        console.print(ui.error_panel(
            "Non-loopback binds require [cyan]--api-token[/] (or "
            "[cyan]GRIMORE_API_TOKEN[/] env).",
            title="Missing API token",
        ))
        raise typer.Exit(code=1)
    if strict_token and not api_token:
        console.print(ui.error_panel(
            "[cyan]--strict-token[/] requires [cyan]--api-token[/] (or "
            "[cyan]GRIMORE_API_TOKEN[/] env) — without a token there is "
            "nothing to enforce.",
            title="Missing API token",
        ))
        raise typer.Exit(code=1)
    if api_token and "--api-token" in sys.argv:
        # The value itself never gets printed — only the advice. A token on
        # the command line is readable by any local user via `ps` / /proc.
        console.print(ui.warn_panel(
            "The API token was passed on the command line, which other "
            "local users can read via [cyan]ps[/]. Prefer the env var: "
            "[cyan]GRIMORE_API_TOKEN=... grimore serve[/].",
            title="Token visible in process list",
        ))

    session = Session(config)
    app_ = build_app(
        session, api_token=api_token, cors_origin=cors_origin,
        strict_token=strict_token,
    )
    if api_token:
        auth_line = "Token auth ON (strict — loopback included)" if strict_token \
            else "Token auth ON"
    else:
        auth_line = "No token — loopback only."
    console.print(ui.success_panel(
        f"Grimore API listening on [bold cyan]http://{host}:{port}[/].\n{auth_line}",
        title="serve",
    ))

    try:
        # proxy_headers=False so the ASGI scope's client address is always
        # the raw transport peer, never rewritten from an X-Forwarded-For
        # header. The token middleware's loopback exemption depends on that
        # peer being trustworthy — Grimore is served directly, not behind a
        # reverse proxy, so honouring forwarding headers would only let a
        # remote caller spoof a loopback origin.
        uvicorn.run(
            app_, host=host, port=port, log_config=None, proxy_headers=False,
        )
    finally:
        session.close()


if __name__ == "__main__":
    app()
