"""
Content Hashing for Idempotency.
This module provides a stable hash calculation for Markdown content,
allowing the system to detect if a file has truly changed before processing.
"""
import hashlib
import re

# Regex to normalize whitespace and newlines for stable hashing
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{2,}")


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
