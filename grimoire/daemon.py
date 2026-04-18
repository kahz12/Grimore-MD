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
from grimoire.memory.taxonomy import Taxonomy
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
        self.taxonomy = Taxonomy()
        self.tagger = Tagger(config, self.router, self.taxonomy)
        self.embedder = Embedder(config)
        self.connector = Connector(self.db, self.embedder)
        
        self.observer = None
        self.processed_count = 0
        self.last_batch_time = time.time()
        self.last_backup_time = time.time()

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
        logger.info("processing_file", path=str(file_path))
        
        try:
            # 1. Parse file
            note = self.parser.parse_file(file_path)
            
            # 2. Security & Policy Check
            privacy = note.metadata.get("privacy", "local")
            if privacy == "never_process":
                logger.info("policy_skip", path=str(file_path), reason="privacy: never_process")
                return

            # Scan for sensitive data
            sensitive_findings = self.security.scan_for_sensitive_data(note.content)
            if sensitive_findings:
                logger.warning("sensitive_data_detected", path=str(file_path), types=sensitive_findings)
                # If sensitive data is found, we FORCE local processing regardless of global config
                # and skip remote APIs if they were enabled.
            
            # 3. Check hash idempotency
            existing_note = self.db.get_note_by_path(str(file_path))
            if existing_note and existing_note[3] == note.content_hash:
                logger.info("file_unchanged", path=str(file_path))
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

            # 7. Embeddings
            vector = self.embedder.embed(clean_content)
            
            # 8. Update Database & Embeddings
            note_id = self.db.upsert_note(str(file_path), note.title, note.content_hash)
            self.db.update_last_tagged(str(file_path))
            
            if vector:
                self.db.delete_note_embeddings(note_id)
                vector_blob = self.embedder.serialize_vector(vector)
                self.db.store_embedding(note_id, 0, note.content[:500], vector_blob)
            
            self.processed_count += 1
            self.last_batch_time = time.time()
            logger.info("file_processed_complete", path=str(file_path))
            
        except Exception as e:
            logger.error("processing_failed", path=str(file_path), error=str(e))
            self.notifier.notify("Grimorio: Error Crítico", f"Error al procesar {file_path.name}")
