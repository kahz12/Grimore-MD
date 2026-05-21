"""
Content Hashing for Idempotency.
This module provides a stable hash calculation for Markdown content,
allowing the system to detect if a file has truly changed before processing.
"""
import hashlib
import re
from pathlib import Path

# Regex to normalize whitespace and newlines for stable hashing
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{2,}")

# Stream files in 64 KiB chunks. Large enough to amortise syscall cost on
# a 100 MB PDF, small enough to keep peak RSS flat for the daemon.
_FILE_HASH_CHUNK = 64 * 1024


def calculate_content_hash(text: str) -> str:
    """
    Calculates a SHA-256 hash of the cleaned text.
    Whitespace and newlines are normalized so that trivial formatting changes
    (like adding an extra space) do not trigger a full re-process of the note.
    """
    # Collapse multiple spaces into one
    normalized = _WHITESPACE_RUN.sub(" ", text)
    # Collapse multiple blank lines into exactly two newlines
    normalized = _NEWLINE_RUN.sub("\n\n", normalized)
    # Strip trailing whitespace from every line and strip the whole string
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """
    Streamed SHA-256 of the raw file bytes — the cheap "did anything change"
    key that gates the expensive text-extraction + LLM path for non-MD
    formats (PDFs, ePubs, DOCX). Used as the first tier of the two-tier
    change-detection scheme; see the multi-format blueprint §6.4.

    Returns the hex digest. Raises OSError if the file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_FILE_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
