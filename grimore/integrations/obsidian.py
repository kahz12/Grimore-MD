"""Generate explicit Obsidian open targets, never execute URI actions."""
from pathlib import PurePosixPath
from urllib.parse import quote, urlencode


def open_target(*, vault: str, path: str | None, format: str = "md",
                heading: str | None = None, source_state: str = "unverified") -> dict:
    """Use only a caller-validated, vault-relative Markdown path.

    The vault name/id is explicit: a folder basename is not necessarily the
    Obsidian vault name. Opening always shows the live file, not our index.
    URI anchors do not disambiguate repeated headings or locate chunk offsets.
    """
    base = {"client": "obsidian", "uri": None, "scope": None}
    if source_state in {"missing", "unavailable"} or not path:
        return {**base, "reason": "source_unavailable"}
    if format != "md":
        return {**base, "reason": "unsupported_format"}
    parts = PurePosixPath(path)
    if (parts.is_absolute() or ".." in parts.parts or "\\" in path
            or "#" in path or any(ord(ch) < 32 for ch in path)):
        return {**base, "reason": "unsupported_note_path"}
    destination = path
    scope = "document"
    limitations = ["opens_current_file_not_indexed_snapshot", "chunk_offset_not_supported"]
    if (source_state == "current" and isinstance(heading, str) and heading.strip()
            and not any(ch in heading for ch in "#^\n\r")):
        destination += "#" + heading
        scope = "heading"
        limitations.append("repeated_headings_not_disambiguated")
    if source_state != "current":
        limitations.append("indexed_heading_not_used_without_current_version")
    uri = "obsidian://open?" + urlencode(
        {"vault": vault, "file": destination}, quote_via=quote, safe="",
    )
    return {**base, "uri": uri, "scope": scope, "limitations": limitations}
