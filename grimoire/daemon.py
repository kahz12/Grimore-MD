import time
from pathlib import Path
from grimoire.utils.config import Config, load_config
from grimoire.utils.logger import setup_logger, get_logger
from grimoire.ingest.observer import VaultObserver
from grimoire.ingest.parser import MarkdownParser
from grimoire.memory.db import Database
from grimoire.output.git_guard import GitGuard
from grimoire.output.frontmatter_writer import FrontmatterWriter
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.tagger import Tagger
from grimoire.memory.taxonomy import load_taxonomy_from_vault
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.connector import Connector
from grimoire.output.link_injector import LinkInjector
from grimoire.utils.notifications import Notifier
from grimoire.utils.security import SecurityGuard
from grimoire.utils.backup import BackupManager

logger = get_logger(__name__)

class GrimoireDaemon:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.memory.db_path)
        self.parser = MarkdownParser()
        self.git_guard = GitGuard(config.vault.path)
        self.writer = FrontmatterWriter()
        self.injector = LinkInjector()
        self.notifier = Notifier()
        self.security = SecurityGuard(config.vault.path)
        self.backup = BackupManager(config.memory.db_path)
        
        # Cognition setup
        self.router = LLMRouter(config)
        self.taxonomy = load_taxonomy_from_vault(Path(config.vault.path))
        self.tagger = Tagger(config, self.router, self.taxonomy)
        self.embedder = Embedder(config, cache=self.db)
        self.connector = Connector(self.db, self.embedder)
        
        self.vault_root = Path(config.vault.path).resolve()
        self.observer = None
        self.processed_count = 0
        self.last_batch_time = time.time()
        self.last_backup_time = time.time()

    def _log_path(self, file_path: Path) -> str:
        try:
            return str(Path(file_path).resolve().relative_to(self.vault_root))
        except (ValueError, OSError):
            return Path(file_path).name

    def start(self):
        logger.info("daemon_starting", vault=self.config.vault.path)
        self.observer = VaultObserver(
            vault_path=self.config.vault.path,
            callback=self.process_file,
            ignored_dirs=self.config.vault.ignored_dirs
        )
        self.observer.start()
        
        try:
            while True:
                current_time = time.time()
                # Batch notification
                if self.processed_count > 0 and (current_time - self.last_batch_time > 300):
                    self.notifier.notify_batch_processed(self.processed_count)
                    self.processed_count = 0
                    self.last_batch_time = current_time
                
                # Daily backup check (every 24h)
                if current_time - self.last_backup_time > 86400:
                    self.backup.create_backup()
                    self.last_backup_time = current_time
                
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("daemon_stopping")
        if self.observer:
            self.observer.stop()

    def process_file(self, file_path: Path):
        rel = self._log_path(file_path)
        logger.info("processing_file", path=rel)

        try:
            # 1. Parse file
            note = self.parser.parse_file(file_path)

            # 2. Security & Policy Check
            privacy = note.metadata.get("privacy", "local")
            if privacy == "never_process":
                logger.info("policy_skip", path=rel, reason="privacy: never_process")
                return

            # Scan for sensitive data — if found, force local-only processing.
            sensitive_findings = self.security.scan_for_sensitive_data(note.content)
            force_local = bool(sensitive_findings)
            if force_local:
                logger.warning("sensitive_data_detected", path=rel, types=sensitive_findings)
                if self.config.cognition.allow_remote:
                    logger.warning("remote_disabled_for_note", path=rel,
                                   reason="sensitive_data_detected")

            # 3. Check hash idempotency
            existing_note = self.db.get_note_by_path(str(file_path))
            if existing_note and existing_note[3] == note.content_hash:
                logger.info("file_unchanged", path=rel)
                return

            # 4. Git Guard: commit current state
            if self.config.output.auto_commit and not self.config.output.dry_run:
                self.git_guard.commit_pre_change(str(file_path))

            # 5. Cognition: Tagging & Summary
            # Sanitize content before sending to LLM
            clean_content = self.security.sanitize_prompt(note.content)
            cognition_data = self.tagger.tag_note(clean_content)
            
            # 6. Output: Update Frontmatter
            metadata_updates = {
                "tags": cognition_data["tags"],
                "summary": cognition_data["summary"],
                "last_tagged": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            self.writer.write_metadata(file_path, metadata_updates, dry_run=self.config.output.dry_run)

            # 7. Embeddings (chunked)
            embedded = self.embedder.embed_chunks(clean_content)

            # 8. Update Database & Embeddings
            note_id = self.db.upsert_note(str(file_path), note.title, note.content_hash)
            self.db.update_last_tagged(str(file_path))
            if note_id is not None:
                self.db.upsert_tags(note_id, cognition_data["tags"])

            if embedded:
                self.db.delete_note_embeddings(note_id)
                for idx, (chunk_text, vector) in enumerate(embedded):
                    self.db.store_embedding(
                        note_id,
                        idx,
                        chunk_text[:500],
                        self.embedder.serialize_vector(vector),
                    )
                logger.info("file_embedded", path=rel, chunks=len(embedded))

            self.processed_count += 1
            self.last_batch_time = time.time()
            logger.info("file_processed_complete", path=rel)

        except Exception as e:
            logger.error("processing_failed", path=rel, error=str(e))
            self.notifier.notify("Grimorio: Error Crítico", f"Error al procesar {file_path.name}")
