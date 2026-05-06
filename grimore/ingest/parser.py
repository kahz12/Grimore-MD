"""
Markdown parsing logic for Project Grimore.
Uses the python-frontmatter library to separate YAML metadata from note content.
"""
import frontmatter
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any, Union
from grimore.utils.hashing import calculate_content_hash
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

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

    Vault-scope contract
    --------------------
    Callers MUST ensure ``file_path`` resolves inside the vault before
    calling :py:meth:`parse_file`. The two in-tree consumers (``cli.scan``
    and ``daemon.process_file``) do this via ``SecurityGuard.resolve_within_vault``.
    For defence-in-depth, callers can also pass ``vault_root`` to
    :py:meth:`parse_file` and the parser will re-validate internally —
    use this whenever the parser is invoked from a new code path so a
    forgotten outer check cannot turn into a path-traversal bug.
    """
    def parse_file(
        self,
        file_path: Path,
        *,
        vault_root: Optional[Union[str, Path]] = None,
    ) -> ParsedNote:
        """
        Parses a Markdown file into a ParsedNote object.
        Extracts title from frontmatter, first H1, or filename as fallback.
        Calculates a SHA-256 hash of the content for change detection.

        If ``vault_root`` is provided, the resolved ``file_path`` is required
        to live under it (symlinks followed, ``..`` rejected). Raises
        :class:`ValueError` if the file escapes the vault.
        """
        if vault_root is not None:
            # Defence-in-depth: even if the caller already filtered, re-check
            # here so the parser is safe to drop into a new call site.
            SecurityGuard.resolve_within_vault(file_path, vault_root)

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
