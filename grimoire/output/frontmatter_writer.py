import frontmatter
from pathlib import Path
from typing import Any
from grimoire.utils.atomic import atomic_write
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class FrontmatterWriter:
    def write_metadata(self, file_path: Path, metadata_updates: dict[str, Any], dry_run: bool = True):
        """
        Updates the frontmatter of a markdown file with new metadata.
        """
        if dry_run:
            logger.info("dry_run_metadata_update", path=str(file_path), updates=metadata_updates)
            return

        try:
            post = frontmatter.load(file_path)
            post.metadata.update(metadata_updates)

            atomic_write(file_path, lambda fh: frontmatter.dump(post, fh), mode="wb")

            logger.info("metadata_updated", path=str(file_path))
        except Exception as e:
            logger.error("metadata_update_failed", path=str(file_path), error=str(e))
