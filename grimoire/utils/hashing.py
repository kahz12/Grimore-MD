import hashlib
import re

_WHITESPACE_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{2,}")


def calculate_content_hash(text: str) -> str:
    """
    Calculates a hash of the clean body text.
    Whitespace runs are collapsed (so trivial reformatting is idempotent),
    but token boundaries and line breaks are preserved so that distinct
    bodies do not collide (e.g. "ab cd" != "abcd").
    """
    normalized = _WHITESPACE_RUN.sub(" ", text)
    normalized = _NEWLINE_RUN.sub("\n\n", normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
