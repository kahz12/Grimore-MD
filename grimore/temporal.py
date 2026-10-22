"""Normalize document dates without inventing publication or verification."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping


_PUBLISHED_KEYS = (
    "published_at", "published", "publication_date", "date_published",
)
_MODIFIED_KEYS = (
    "modified_at", "modified", "updated_at", "updated", "last_modified",
    "modification_date",
)


def normalize_document_date(value: Any) -> str | None:
    """Return an ISO date/timestamp for trusted metadata, or ``None``."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("D:") and len(raw) >= 10:
            # PDF info-dict form: D:YYYYMMDDHHmmSSOHH'mm'. Missing pieces
            # are allowed by the spec; keep only values we can state safely.
            digits = "".join(char for char in raw[2:16] if char.isdigit())
            if len(digits) < 4:
                return None
            try:
                year = int(digits[:4])
                month = int(digits[4:6]) if len(digits) >= 6 else 1
                day = int(digits[6:8]) if len(digits) >= 8 else 1
                hour = int(digits[8:10]) if len(digits) >= 10 else 0
                minute = int(digits[10:12]) if len(digits) >= 12 else 0
                second = int(digits[12:14]) if len(digits) >= 14 else 0
                parsed = datetime(year, month, day, hour, minute, second)
            except ValueError:
                return None
        else:
            if len(raw) == 10:
                try:
                    return date.fromisoformat(raw).isoformat()
                except ValueError:
                    pass
            candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                try:
                    return date.fromisoformat(raw).isoformat()
                except ValueError:
                    return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.isoformat(timespec="seconds")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z",
    )


def derive_document_dates(
    metadata: Mapping[str, Any],
    document_format: str,
    *,
    mtime_ns: int | None,
) -> dict[str, str | None]:
    """Extract publication/modification dates and record exact provenance.

    Publication is never inferred from filesystem or creation timestamps.
    Modification falls back to the source file's mtime, which is useful but
    explicitly labelled so consumers cannot mistake it for embedded metadata.
    """
    fmt = (document_format or "unknown").lower()
    values = {str(key).lower(): value for key, value in metadata.items()}
    published_keys = list(_PUBLISHED_KEYS)
    if fmt in {"md", "markdown", "epub", "html", "htm"}:
        published_keys.append("date")

    published_at = published_source = None
    for key in published_keys:
        value = normalize_document_date(values.get(key))
        if value is not None:
            published_at = value
            published_source = f"document:{fmt}:{key}"
            break

    modified_at = modified_source = None
    for key in _MODIFIED_KEYS:
        value = normalize_document_date(values.get(key))
        if value is not None:
            modified_at = value
            modified_source = f"document:{fmt}:{key}"
            break
    if modified_at is None and mtime_ns is not None and mtime_ns >= 0:
        modified_at = datetime.fromtimestamp(
            mtime_ns / 1_000_000_000, tz=timezone.utc,
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        modified_source = "filesystem:mtime"

    return {
        "published_at": published_at,
        "published_source": published_source,
        "modified_at": modified_at,
        "modified_source": modified_source,
    }
