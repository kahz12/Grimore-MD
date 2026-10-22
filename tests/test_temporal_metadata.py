"""Distinct document dates with provenance and honest unknowns."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from grimore.memory.db import Database
from grimore.temporal import derive_document_dates, normalize_document_date


def test_document_dates_keep_semantics_and_provenance():
    dates = derive_document_dates(
        {
            "date": "2024-05-06",
            "modified": "2025-07-08T09:10:11Z",
        },
        "md",
        mtime_ns=1_000_000_000,
    )

    assert dates == {
        "published_at": "2024-05-06",
        "published_source": "document:md:date",
        "modified_at": "2025-07-08T09:10:11Z",
        "modified_source": "document:md:modified",
    }


def test_unknown_publication_is_not_inferred_from_filesystem_mtime():
    dates = derive_document_dates({}, "pdf", mtime_ns=1_000_000_000)

    assert dates["published_at"] is None
    assert dates["published_source"] is None
    assert dates["modified_at"] == "1970-01-01T00:00:01Z"
    assert dates["modified_source"] == "filesystem:mtime"


def test_pdf_date_and_invalid_values_are_normalized_safely():
    assert normalize_document_date("D:20230809101112") == "2023-08-09T10:11:12"
    assert normalize_document_date("not-a-date") is None


def test_timeline_separates_indexing_from_explicit_verification(tmp_path):
    db = Database(str(tmp_path / "timeline.db"))
    note_id = db.upsert_note("/vault/a.md", "A", "")
    assert note_id is not None
    db.complete_note_indexing(
        note_id, "content", "file", 12, 34,
        published_at="2024-05-06",
        published_source="document:md:date",
        modified_at="2025-07-08T09:10:11Z",
        modified_source="document:md:modified",
    )
    db.upsert_freshness("/vault/a.md", None, 90)

    timeline = db.get_note_timeline(note_id)
    assert timeline is not None
    assert timeline["published"] == {
        "value": "2024-05-06", "source": "document:md:date",
    }
    assert timeline["modified"]["value"] == "2025-07-08T09:10:11Z"
    assert timeline["indexed"]["value"] is not None
    assert timeline["indexed"]["source"] == "grimore:index"
    assert timeline["verified"] == {"value": None, "source": None}

    assert db.touch_freshness_verified("/vault/a.md") is True
    verified = db.get_note_timeline(note_id)["verified"]
    assert verified["value"] is not None
    assert verified["source"] == "human:chronicler"


def test_existing_note_can_backfill_source_dates_without_inventing_index_time(tmp_path):
    db = Database(str(tmp_path / "backfill.db"))
    note_id = db.upsert_note("/vault/a.md", "A", "content")
    assert note_id is not None
    assert db.note_timeline_needs_backfill("/vault/a.md") is True

    assert db.record_document_dates(
        "/vault/a.md",
        published_at="2024-05-06",
        published_source="document:md:date",
        modified_at="2025-07-08T09:10:11Z",
        modified_source="filesystem:mtime",
    ) is True

    timeline = db.get_note_timeline(note_id)
    assert db.note_timeline_needs_backfill("/vault/a.md") is False
    assert timeline["published"]["value"] == "2024-05-06"
    assert timeline["indexed"] == {"value": None, "source": None}


def test_legacy_scan_anchor_is_migrated_as_unknown_verification(tmp_path):
    path = tmp_path / "legacy.db"
    db = Database(str(path))
    note_id = db.upsert_note("/vault/a.md", "A", "hash")
    assert note_id is not None
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE freshness")
        conn.execute(
            "CREATE TABLE freshness ("
            "note_path TEXT PRIMARY KEY, last_verified TEXT NOT NULL, "
            "window_days INTEGER NOT NULL, decay_check_at TEXT, likely_stale INTEGER)"
        )
        conn.execute(
            "INSERT INTO freshness VALUES (?, ?, ?, NULL, NULL)",
            ("/vault/a.md", "2020-01-01T00:00:00Z", 90),
        )

    migrated = Database(str(path))

    assert migrated.get_freshness_row("/vault/a.md")[0] is None
    assert migrated.get_note_timeline(note_id)["verified"] == {
        "value": None, "source": None,
    }
    with migrated._get_connection() as conn:
        info = {row[1]: row for row in conn.execute("PRAGMA table_info(freshness)")}
        legacy_anchor = conn.execute(
            "SELECT legacy_anchor_at, legacy_anchor_source FROM freshness "
            "WHERE note_path = '/vault/a.md'",
        ).fetchone()
    assert info["last_verified"][3] == 0
    assert "verification_source" in info
    assert legacy_anchor == (
        "2020-01-01T00:00:00Z", "legacy:last_verified-untrusted",
    )


def test_normalizer_preserves_aware_instants_in_utc():
    value = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert normalize_document_date(value) == "2026-01-02T03:04:00Z"
