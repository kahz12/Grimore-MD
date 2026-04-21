"""
Frontmatter Writer.
This module updates Markdown YAML frontmatter with tags and summaries.
It uses atomic writes to prevent file corruption.
"""
import frontmatter
from pathlib import Path
from typing import Any
from grimoire.utils.atomic import atomic_write
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class FrontmatterWriter:
    """
    Responsible for writing cognitive metadata back to the original Markdown files.
    """
    def write_metadata(self, file_path: Path, metadata_updates: dict[str, Any], dry_run: bool = True):
        """
        Updates the frontmatter of a Markdown file with new metadata (tags, summary, etc.).
        Uses atomic_write to ensure that partial writes don't happen if the process is interrupted.
        """
        if dry_run:
            logger.info("dry_run_metadata_update", path=str(file_path), updates=metadata_updates)
            return

        try:
            # Load existing frontmatter and content
            post = frontmatter.load(file_path)
            # Merge new metadata into existing metadata
            post.metadata.update(metadata_updates)

            # Perform an atomic write to the file
            atomic_write(file_path, lambda fh: frontmatter.dump(post, fh), mode="wb")

            logger.info("metadata_updated", path=str(file_path))
        except Exception as e:
            logger.error("metadata_update_failed", path=str(file_path), error=str(e))
