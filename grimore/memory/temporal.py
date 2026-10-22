"""Document publication, modification, indexing and verification timeline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from grimore.memory._base import DbBase


_SQL_BATCH = 900


def _point(value, source) -> dict[str, str | None]:
    return {"value": value, "source": source}


class TemporalMixin(DbBase):
    """Persist factual dates with provenance and preserve unknown values."""

    def record_indexed_dates(
        self,
        note_id: int,
        *,
        published_at: str | None,
        published_source: str | None,
        modified_at: str | None,
        modified_source: str | None,
        indexed_at: str | None = None,
    ) -> None:
        indexed_at = indexed_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        ).replace("+00:00", "Z")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO note_temporal
                    (note_id, published_at, published_source,
                     modified_at, modified_source, indexed_at, indexed_source)
                VALUES (?, ?, ?, ?, ?, ?, 'grimore:index')
                ON CONFLICT(note_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    published_source = excluded.published_source,
                    modified_at = excluded.modified_at,
                    modified_source = excluded.modified_source,
                    indexed_at = excluded.indexed_at,
                    indexed_source = excluded.indexed_source
                """,
                (
                    note_id, published_at, published_source,
                    modified_at, modified_source, indexed_at,
                ),
            )
            conn.commit()

    def note_timeline_needs_backfill(self, note_path: str) -> bool:
        """Whether an existing indexed note lacks its temporal record."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT t.note_id FROM notes n LEFT JOIN note_temporal t "
                "ON t.note_id = n.id WHERE n.path = ?",
                (note_path,),
            ).fetchone()
        return row is not None and row[0] is None

    def record_document_dates(
        self,
        note_path: str,
        *,
        published_at: str | None,
        published_source: str | None,
        modified_at: str | None,
        modified_source: str | None,
    ) -> bool:
        """Backfill source dates without pretending a new indexing occurred."""
        with self._get_connection() as conn:
            note = conn.execute(
                "SELECT id FROM notes WHERE path = ?", (note_path,),
            ).fetchone()
            if note is None:
                return False
            conn.execute(
                """
                INSERT INTO note_temporal
                    (note_id, published_at, published_source,
                     modified_at, modified_source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    published_source = excluded.published_source,
                    modified_at = excluded.modified_at,
                    modified_source = excluded.modified_source
                """,
                (
                    int(note[0]), published_at, published_source,
                    modified_at, modified_source,
                ),
            )
            conn.commit()
        return True

    def get_note_timeline(self, note_id: int) -> dict | None:
        return self.get_note_timelines([note_id]).get(int(note_id))

    def get_note_timelines(self, note_ids: Iterable[int]) -> dict[int, dict]:
        unique = sorted({int(note_id) for note_id in note_ids})
        if not unique:
            return {}
        out: dict[int, dict] = {}
        with self._get_connection() as conn:
            for start in range(0, len(unique), _SQL_BATCH):
                batch = unique[start:start + _SQL_BATCH]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    "SELECT n.id, t.published_at, t.published_source, "
                    "t.modified_at, t.modified_source, t.indexed_at, "
                    "t.indexed_source, f.last_verified, f.verification_source "
                    "FROM notes n LEFT JOIN note_temporal t ON t.note_id = n.id "
                    "LEFT JOIN freshness f ON f.note_path = n.path "
                    f"WHERE n.id IN ({marks})",
                    tuple(batch),
                ).fetchall()
                for row in rows:
                    out[int(row[0])] = {
                        "published": _point(row[1], row[2]),
                        "modified": _point(row[3], row[4]),
                        "indexed": _point(row[5], row[6]),
                        "verified": _point(row[7], row[8]),
                    }
        return out
