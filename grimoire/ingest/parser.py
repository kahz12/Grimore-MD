"""
Markdown parsing logic for Project Grimoire.
Uses the python-frontmatter library to separate YAML metadata from note content.
"""
import frontmatter
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
from grimoire.utils.hashing import calculate_content_hash
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Hard cap on parsed file size. Markdown notes above this threshold are
# almost certainly binary blobs, dumps, or accidents — reading them into
# memory would open a DoS vector via a shared or networked vault.
MAX_NOTE_BYTES = 2_000_000

@dataclass
class ParsedNote:
    """Represents a fully parsed Markdown note with its metadata and unique content hash."""
    path: Path
    title: str
    metadata: dict[str, Any]
    content: str
    content_hash: str

class MarkdownParser:
    """
    Responsible for reading Markdown files and extracting structured data.
    """
    def parse_file(self, file_path: Path) -> ParsedNote:
        """
        Parses a Markdown file into a ParsedNote object.
        Extracts title from frontmatter, first H1, or filename as fallback.
        Calculates a SHA-256 hash of the content for change detection.
        """
        try:
            size = file_path.stat().st_size
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e
        if size > MAX_NOTE_BYTES:
            logger.warning(
                "note_too_large",
                path=str(file_path),
                size=size,
                max=MAX_NOTE_BYTES,
            )
            raise ValueError(
                f"note exceeds {MAX_NOTE_BYTES} bytes: {file_path} ({size} bytes)"
            )

        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        content = post.content
        metadata = post.metadata
        
        # Determine title: from metadata or first H1 or filename
        title = metadata.get('title')
        if not title:
            h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1)
            else:
                title = file_path.stem

        return ParsedNote(
            path=file_path,
            title=title,
            metadata=metadata,
            content=content,
            content_hash=calculate_content_hash(content)
        )
