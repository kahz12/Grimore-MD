import typer
from pathlib import Path
from rich.console import Console
from grimoire.utils.config import load_config
from grimoire.utils.logger import setup_logger, get_logger
from grimoire.output.git_guard import GitGuard
from grimoire.memory.db import Database
from grimoire.ingest.parser import MarkdownParser
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.tagger import Tagger
from grimoire.memory.taxonomy import Taxonomy
from grimoire.output.frontmatter_writer import FrontmatterWriter
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.connector import Connector
from grimoire.output.link_injector import LinkInjector
from grimoire.cognition.oracle import Oracle
import time

app = typer.Typer(help="Grimoire v2.0 - Automated Knowledge Engine")
console = Console()
logger = get_logger(__name__)

@app.command()
def scan(
    vault_path: Path = typer.Option(None, help="Path to the vault"),
    dry_run: bool = typer.Option(None, "--dry-run/--no-dry-run", help="Simulate changes without writing"),
    json_logs: bool = typer.Option(False, "--json", help="Output logs in JSON format")
):
    """
    Scan the vault and identify files to process.
    """
    setup_logger(json_format=json_logs)
    config = load_config()
    
    actual_vault_path = vault_path or Path(config.vault.path)
    # If dry_run is None (not passed), use config. If passed, use CLI value.
    is_dry_run = dry_run if dry_run is not None else config.output.dry_run
    
    logger.info("scan_start", path=str(actual_vault_path), dry_run=is_dry_run)
    
    if not actual_vault_path.exists():
        logger.error("vault_not_found", path=str(actual_vault_path))
        raise typer.Exit(code=1)

    git_guard = GitGuard(str(actual_vault_path))
    if not git_guard.is_repo_ready():
        logger.warning("git_not_ready", message="Operations will continue without git safety")

    db = Database(config.memory.db_path)
    parser = MarkdownParser()
    router = LLMRouter(config)
    taxonomy = Taxonomy()
    tagger = Tagger(config, router, taxonomy)
    writer = FrontmatterWriter()
    embedder = Embedder(config)

    files = list(actual_vault_path.glob("**/*.md"))
    logger.info("scan_complete", files_found=len(files))
    
    for file in files:
        if any(part in str(file) for part in config.vault.ignored_dirs):
            continue
        
        note = parser.parse_file(file)
        existing = db.get_note_by_path(str(file))
        
        if existing and existing[3] == note.content_hash:
            console.print(f"[dim]✓ {file.relative_to(actual_vault_path)} (unchanged)[/dim]")
        else:
            console.print(f"[yellow]⚡ {file.relative_to(actual_vault_path)} (processing...)[/yellow]")
            
            # Cognition: Tagging & Summary
            cognition_data = tagger.tag_note(note.content)
            
            if not is_dry_run:
                # Update Frontmatter
                metadata_updates = {
                    "tags": cognition_data["tags"],
                    "summary": cognition_data["summary"],
                    "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                writer.write_metadata(file, metadata_updates, dry_run=False)

                note_id = db.upsert_note(str(file), note.title, note.content_hash)
                db.update_last_tagged(str(file))

                vector = embedder.embed(note.content)
                if vector and note_id is not None:
                    db.delete_note_embeddings(note_id)
                    db.store_embedding(note_id, 0, note.content[:500], embedder.serialize_vector(vector))
                else:
                    logger.warning("embedding_skipped", path=str(file))

                logger.info("note_processed", path=str(file))
            else:
                console.print(f"  [blue]Tags:[/blue] {cognition_data['tags']}")
                console.print(f"  [blue]Summary:[/blue] {cognition_data['summary']}")

@app.command()
def connect(
    dry_run: bool = typer.Option(None, "--dry-run/--no-dry-run", help="Simulate link injection"),
):
    """
    Discover semantic connections between notes and inject links.
    """
    setup_logger()
    config = load_config()
    is_dry_run = dry_run if dry_run is not None else config.output.dry_run
    
    db = Database(config.memory.db_path)
    embedder = Embedder(config)
    connector = Connector(db, embedder)
    injector = LinkInjector()
    
    all_embeddings = db.get_all_embeddings()
    logger.info("connection_discovery_start", total_chunks=len(all_embeddings))
    
    # Simple algorithm: for each note, find top similar notes
    processed_notes = set()
    for note_id, text, vector_blob in all_embeddings:
        if note_id in processed_notes: continue
        processed_notes.add(note_id)
        
        # Get path for this note_id
        with db._get_connection() as conn:
            path, title = conn.execute("SELECT path, title FROM notes WHERE id = ?", (note_id,)).fetchone()
        
        vector = embedder.deserialize_vector(vector_blob)
        similar = connector.find_similar_notes(vector, top_k=3, exclude_note_id=note_id)
        
        # Filter by threshold
        candidates = [s for s in similar if s["score"] > 0.7]
        
        if candidates:
            console.print(f"[bold]Connections for {title}:[/bold]")
            connections_to_inject = []
            for c in candidates:
                with db._get_connection() as conn:
                    c_title = conn.execute("SELECT title FROM notes WHERE id = ?", (c['note_id'],)).fetchone()[0]
                console.print(f"  - {c_title} (score: {c['score']:.2f})")
                connections_to_inject.append({"title": c_title, "reason": "High semantic similarity."})
            
            injector.inject_links(Path(path), connections_to_inject, dry_run=is_dry_run)

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the Oracle"),
    top_k: int = typer.Option(5, help="Number of context chunks to retrieve"),
    export: Path = typer.Option(None, "--export", help="Save the answer to a new markdown file")
):
    """
    Consult the Grimoire Oracle about your vault's knowledge.
    """
    setup_logger()
    config = load_config()
    
    db = Database(config.memory.db_path)
    router = LLMRouter(config)
    embedder = Embedder(config)
    oracle = Oracle(config, db, router, embedder)
    
    with console.status("[bold green]The Oracle is consulting the whispers..."):
        result = oracle.ask(question, top_k=top_k)
    
    console.print("\n[bold purple]🔮 Oracle Response:[/bold purple]")
    console.print(result["answer"])
    
    if result["sources"]:
        console.print("\n[bold blue]📚 Sources:[/bold blue]")
        for source in result["sources"]:
            console.print(f"  - [[{source}]]")

    if export:
        export_path = Path(config.vault.path) / export
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"---\ntitle: \"Oracle: {question[:30]}...\"\ndate: {time.strftime('%Y-%m-%d')}\ntype: oracle_response\n---\n\n")
            f.write(f"# 🔮 Question: {question}\n\n")
            f.write(result["answer"])
            f.write("\n\n## Sources\n")
            for source in result["sources"]:
                f.write(f"- [[{source}]]\n")
        console.print(f"\n[green]✓ Response exported to {export_path}[/green]")

@app.command()
def daemon(
    action: str = typer.Argument("run", help="Action: run (foreground), start (background), stop, status"),
    json_logs: bool = typer.Option(False, "--json", help="Output logs in JSON format")
):
    """
    Manage the Grimoire daemon.
    """
    from grimoire.utils.system import start_daemon_background, stop_daemon, is_running
    
    pid_file = "grimoire.pid"
    log_file = "grimoire.log"

    if action == "run":
        setup_logger(json_format=json_logs)
        config = load_config()
        from grimoire.daemon import GrimoireDaemon
        daemon = GrimoireDaemon(config)
        daemon.start()
    elif action == "start":
        start_daemon_background(pid_file, log_file)
    elif action == "stop":
        stop_daemon(pid_file)
    elif action == "status":
        if is_running(pid_file):
            console.print("[green]Daemon is running.[/green]")
        else:
            console.print("[red]Daemon is stopped.[/red]")

@app.command()
def status():
    """
    Show Grimoire status, configuration and vault metrics.
    """
    config = load_config()
    db = Database(config.memory.db_path)
    
    with db._get_connection() as conn:
        total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        tagged_notes = conn.execute("SELECT COUNT(*) FROM notes WHERE last_tagged IS NOT NULL").fetchone()[0]
        total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    console.print("[bold blue]Grimoire v2.0 Dashboard[/bold blue]")
    console.print(f"Vault Path:  {config.vault.path}")
    console.print(f"Notes:       {total_notes} total ({tagged_notes} tagged)")
    console.print(f"Embeddings:  {total_embeddings} chunks indexed")
    console.print(f"Dry Run:     {config.output.dry_run}")
    console.print(f"Local LLM:   {config.cognition.model_llm_local}")
    
    from grimoire.utils.system import is_running
    if is_running("grimoire.pid"):
        console.print("Daemon:      [green]ACTIVE[/green]")
    else:
        console.print("Daemon:      [red]INACTIVE[/red]")

if __name__ == "__main__":
    app()
