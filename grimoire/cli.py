"""
Command Line Interface (CLI) for Project Grimoire.
This module defines the user commands for scanning vaults, connecting notes,
consulting the Oracle (RAG), and managing the background daemon.
"""
import time
from pathlib import Path

import typer
from rich.text import Text

from grimoire.cognition.connector import Connector
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.oracle import Oracle
from grimoire.cognition.tagger import Tagger
from grimoire.ingest.parser import MarkdownParser
from grimoire.memory.db import Database
from grimoire.memory.taxonomy import load_taxonomy_from_vault
from grimoire.cognition.chunker import Chunker
from grimoire.output.frontmatter_writer import FrontmatterWriter
from grimoire.output.git_guard import GitGuard
from grimoire.output.link_injector import LinkInjector
from grimoire.utils import ui
from grimoire.utils.config import load_config
from grimoire.utils.logger import get_logger, setup_logger
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
            f"Comprueba la ruta o ajusta [cyan]grimoire.toml[/].",
            title="Vault missing",
        ))
        raise typer.Exit(code=1)

    # Initialize core services
    git_guard = GitGuard(str(actual_vault_path))
    if not git_guard.is_repo_ready():
        console.print(ui.warn_panel(
            "El directorio no es un repositorio git. El escaneo continuará sin red de seguridad.\n"
            "Inicialízalo con [cyan]git init[/] dentro del vault para activar snapshots automáticos.",
            title="Git no detectado",
        ))

    db = Database(config.memory.db_path)
    parser = MarkdownParser()
    router = LLMRouter(config)
    taxonomy = load_taxonomy_from_vault(actual_vault_path)
    tagger = Tagger(config, router, taxonomy)
    writer = FrontmatterWriter()
    embedder = Embedder(config, cache=db)
    security = SecurityGuard(str(actual_vault_path))

    # Identify files to process (filtering out ignored directories)
    files = [
        f for f in actual_vault_path.glob("**/*.md")
        if not any(part in str(f) for part in config.vault.ignored_dirs)
    ]

    if not files:
        console.print(ui.info_panel(
            f"No se encontraron notas [bold].md[/] en [cyan]{actual_vault_path}[/].",
            title="Vault vacío",
        ))
        return

    stats = {"unchanged": 0, "processed": 0, "skipped": 0, "errors": 0, "chunks": 0}
    ui.section(f"Analizando {len(files)} notas")

    with ui.progress_bar() as progress:
        task = progress.add_task("Escaneando", total=len(files))
        for file in files:
            rel = file.relative_to(actual_vault_path)
            progress.update(task, description=f"[grimoire.muted]{rel}[/]")

            try:
                # Step 1: Parse Markdown and metadata
                note = parser.parse_file(file)
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
            existing = db.get_note_by_path(str(file))
            if existing and existing[3] == note.content_hash:
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
                    "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                writer.write_metadata(file, metadata_updates, dry_run=False)

                # Step 5: Update persistence layer (DB)
                note_id = db.upsert_note(str(file), note.title, note.content_hash)
                db.update_last_tagged(str(file))
                if note_id is not None:
                    db.upsert_tags(note_id, cognition_data["tags"])

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
        ("Procesadas",  Text(str(stats["processed"]), style="grimoire.success")),
        ("Sin cambios", Text(str(stats["unchanged"]), style="grimoire.muted")),
        ("Omitidas",    Text(str(stats["skipped"]),   style="grimoire.warning")),
        ("Errores",     Text(str(stats["errors"]),    style="grimoire.danger" if stats["errors"] else "grimoire.muted")),
        ("Chunks",      Text(str(stats["chunks"]),    style="grimoire.accent")),
    ]
    console.print()
    console.print(ui.success_panel(ui.kv_table(summary_rows), title="Resumen del escaneo"))

    if is_dry_run and stats["processed"] > 0:
        ui.tip("Fue un ensayo. Ejecuta [cyan]grimoire scan --no-dry-run[/] para aplicar cambios.")


@app.command(rich_help_panel="Knowledge ops")
def connect(
    dry_run: bool = typer.Option(None, "--dry-run/--no-dry-run", help="Simulate link injection"),
):
    """
    🕸️  Discover semantic connections between notes and inject [[wikilinks]].
    
    This command uses vector embeddings and cosine similarity to find related notes.
    It then injects a 'Suggested Connections' section into the Markdown files.
    """
    setup_logger()
    config = load_config()
    is_dry_run = dry_run if dry_run is not None else config.output.dry_run

    ui.command_header("connect", "Tejiendo hilos entre ideas")
    console.print(Text.assemble("  ", _mode_badge(is_dry_run)))

    db = Database(config.memory.db_path)
    embedder = Embedder(config, cache=db)
    connector = Connector(db, embedder)
    injector = LinkInjector()

    all_embeddings = db.get_all_embeddings()
    if not all_embeddings:
        console.print(ui.warn_panel(
            "No hay embeddings aún. Ejecuta [cyan]grimoire scan --no-dry-run[/] primero.",
            title="Sin memoria vectorial",
        ))
        return

    ui.section(f"Buscando conexiones entre {len(all_embeddings)} fragmentos")

    processed_notes: set[int] = set()
    total_links = 0
    unique_sources = 0

    with ui.progress_bar() as progress:
        task = progress.add_task("Conectando", total=len(all_embeddings))
        for note_id, _text, vector_blob in all_embeddings:
            progress.advance(task)
            if note_id in processed_notes:
                continue
            processed_notes.add(note_id)

            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT path, title FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
            if not row:
                logger.warning("orphan_embedding", note_id=note_id)
                continue
            path, title = row

            # Find similar notes using cosine similarity on embeddings
            vector = embedder.deserialize_vector(vector_blob)
            similar = connector.find_similar_notes(vector, top_k=12, exclude_note_id=note_id)

            seen_ids: set[int] = set()
            candidates = []
            for s in similar:
                # Threshold for similarity and avoiding duplicates/self
                if s["score"] <= 0.7 or s["note_id"] in seen_ids:
                    continue
                seen_ids.add(s["note_id"])
                candidates.append(s)
                if len(candidates) >= 3:
                    break

            if not candidates:
                continue

            connections_to_inject = []
            bullet_lines = []
            for c in candidates:
                with db._get_connection() as conn:
                    c_row = conn.execute(
                        "SELECT title FROM notes WHERE id = ?", (c["note_id"],)
                    ).fetchone()
                if not c_row:
                    continue
                c_title = c_row[0]
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
            ("Notas conectadas", Text(str(unique_sources), style="grimoire.success")),
            ("Enlaces sugeridos", Text(str(total_links), style="grimoire.accent")),
        ]),
        title="Resumen de conexiones",
    ))


@app.command(rich_help_panel="Knowledge ops")
def ask(
    question: str = typer.Argument(..., help="La pregunta al Oráculo"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Fragmentos de contexto a recuperar"),
    export: Path = typer.Option(None, "--export", "-e", help="Guardar la respuesta como nota markdown"),
):
    """
    🔮 Consult the Grimoire Oracle about your vault's knowledge.
    
    This is a Retrieval-Augmented Generation (RAG) system. It searches for relevant
    note fragments, provides them as context to the LLM, and generates an answer
    with citations to your own notes.
    """
    setup_logger()
    config = load_config()

    ui.command_header("ask", "Consultando al Oráculo")

    db = Database(config.memory.db_path)
    router = LLMRouter(config)
    embedder = Embedder(config, cache=db)
    oracle = Oracle(config, db, router, embedder)

    console.print()
    console.print(ui.info_panel(
        Text(question, style="bold white"),
        title="Pregunta",
    ))

    with console.status("[grimoire.mystic]El Oráculo escucha los susurros...[/]", spinner="dots12"):
        # Perform the RAG query
        result = oracle.ask(question, top_k=top_k)

    console.print()
    console.print(ui.oracle_panel(result["answer"]))

    if result["sources"]:
        ui.section("Fuentes citadas")
        for source in result["sources"]:
            console.print(Text.assemble(
                ("  • ", "grimoire.muted"),
                (f"[[{source}]]", "grimoire.accent"),
            ))
    else:
        ui.tip("El Oráculo no ha encontrado notas relevantes. ¿Has ejecutado [cyan]grimoire scan[/]?")

    if export:
        # Export the Oracle's response as a new Markdown file in the vault
        vault_root = Path(config.vault.path).resolve()
        export_path = (vault_root / export).resolve()
        try:
            export_path.relative_to(vault_root)
        except ValueError:
            console.print(ui.error_panel(
                f"[bold]--export[/] debe apuntar dentro del vault ({vault_root}).",
                title="Ruta inválida",
            ))
            raise typer.BadParameter("--export must resolve to a path inside the vault")

        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f'---\ntitle: "Oracle: {question[:30]}..."\n')
            f.write(f"date: {time.strftime('%Y-%m-%d')}\ntype: oracle_response\n---\n\n")
            f.write(f"# 🔮 Question: {question}\n\n")
            f.write(result["answer"])
            f.write("\n\n## Sources\n")
            for source in result["sources"]:
                f.write(f"- [[{source}]]\n")

        console.print()
        console.print(ui.success_panel(
            f"Respuesta guardada en [bold cyan]{export_path}[/].",
            title="Exportado",
        ))


@app.command(rich_help_panel="Daemon")
def daemon(
    action: str = typer.Argument("run", help="Acción: run · start · stop · status"),
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

        console.print(ui.info_panel(
            "El daemon correrá en primer plano. [bold]Ctrl-C[/] para detener.",
            title="Foreground mode",
        ))
        instance = GrimoireDaemon(config)
        instance.start()
    elif action == "start":
        # Start daemon in background
        if is_running(pid_file):
            console.print(ui.warn_panel(
                f"Ya hay un daemon activo. PID file: [cyan]{pid_file}[/].",
                title="Ya está corriendo",
            ))
            return
        start_daemon_background(pid_file, log_file)
        console.print(ui.success_panel(
            f"Daemon en segundo plano. Logs → [cyan]{log_file}[/].",
            title="Arrancado",
        ))
    elif action == "stop":
        # Stop background daemon
        if not is_running(pid_file):
            console.print(ui.info_panel("No hay daemon corriendo.", title="Nada que detener"))
            return
        stop_daemon(pid_file)
        console.print(ui.success_panel("Daemon detenido.", title="Stop"))
    elif action == "status":
        # Check if daemon is active
        active = is_running(pid_file)
        console.print(Text.assemble("  ", ui.daemon_badge(active)))
    else:
        console.print(ui.error_panel(
            f"Acción desconocida: [bold]{action}[/]\nUsa [cyan]run · start · stop · status[/].",
            title="Comando inválido",
        ))
        raise typer.Exit(code=2)


@app.command(rich_help_panel="Knowledge ops")
def tags(
    limit: int = typer.Option(30, "--limit", "-n", help="Cuántos tags mostrar (por frecuencia)"),
):
    """🏷️  List the most-used tags and how many notes each one labels."""
    setup_logger()
    config = load_config()
    db = Database(config.memory.db_path)

    ui.command_header("tags", f"top {limit}")

    rows = db.get_tag_frequency(limit=limit)
    if not rows:
        console.print(ui.info_panel(
            "Todavía no hay tags registrados. Ejecuta [cyan]grimoire scan --no-dry-run[/] primero.",
            title="Sin tags",
        ))
        return

    console.print()
    console.print(ui.tag_frequency_table(rows))
    console.print()
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(rows)} tags", "grimoire.accent"),
        ("  ·  ", "grimoire.muted"),
        (f"{db.get_tag_count()} únicos en uso", "grimoire.muted"),
    ))


@app.command(rich_help_panel="System")
def prune(
    vault_path: Path = typer.Option(None, "--vault-path", "-p", help="Path to the vault"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Solo listar, no borrar"),
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
        if not any(part in str(f) for part in config.vault.ignored_dirs)
    }

    stale = db.find_stale_notes(existing_paths)
    if not stale:
        console.print(ui.success_panel(
            "Ninguna nota huérfana. La base de datos está sincronizada con el vault.",
            title="Todo limpio",
        ))
        return

    ui.section(f"Notas huérfanas detectadas ({len(stale)})")
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
        console.print(Text(f"  … y {len(stale) - 50} más", style="grimoire.muted"))

    if dry_run:
        console.print()
        ui.tip("Ensayo completado. Ejecuta [cyan]grimoire prune --no-dry-run[/] para borrar.")
        return

    # Delete records from DB
    removed = db.prune_missing_notes(existing_paths)
    purged = db.purge_unused_tags()

    console.print()
    console.print(ui.success_panel(
        ui.kv_table([
            ("Notas borradas", Text(str(removed), style="grimoire.success")),
            ("Tags purgados", Text(str(purged), style="grimoire.accent")),
        ]),
        title="Prune completado",
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
    with db._get_connection() as conn:
        total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        tagged_notes = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE last_tagged IS NOT NULL"
        ).fetchone()[0]
        total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        cached_embeddings = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
    unique_tags = db.get_tag_count()

    console.print()
    console.print(ui.render_banner())

    ui.section("Vault")
    console.print(ui.kv_table([
        ("Ruta",          Text(config.vault.path, style="grimoire.accent")),
        ("Notas",         ui.coverage_bar(tagged_notes, total_notes)),
        ("Chunks",        Text(str(total_embeddings), style="grimoire.accent")),
        ("Cache",         Text(f"{cached_embeddings} vectores", style="grimoire.accent")),
        ("Modo",          _mode_badge(config.output.dry_run)),
        ("Auto-commit",   Text("sí" if config.output.auto_commit else "no", style="grimoire.accent")),
    ]))

    ui.section("Cognición")
    console.print(ui.kv_table([
        ("LLM",         Text(config.cognition.model_llm_local,        style="grimoire.accent")),
        ("Embeddings",  Text(config.cognition.model_embeddings_local, style="grimoire.accent")),
        ("Tags únicos", Text(str(unique_tags),                        style="grimoire.accent")),
        ("Remoto",      Text("permitido" if config.cognition.allow_remote else "local-first",
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
        ui.tip("El vault está vacío. Añade notas [bold].md[/] y lanza [cyan]grimoire scan[/].")
    elif total_embeddings == 0:
        console.print()
        ui.tip("Hay notas pero ninguna indexada. Prueba [cyan]grimoire scan --no-dry-run[/].")
    elif config.output.dry_run:
        console.print()
        ui.tip("Estás en modo ensayo. Ajusta [cyan]dry_run = false[/] en [cyan]grimoire.toml[/] para escribir.")


if __name__ == "__main__":
    app()
