"""Human category/tag overrides that survive automated rescans."""
from __future__ import annotations

import json
from typing import Iterable

from grimore.memory._base import DbBase


_SQL_BATCH = 900


def _decode_tags(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(tag) for tag in value] if isinstance(value, list) else []


def apply_classification_override(db, note_path: str, cognition_data: dict) -> dict:
    """Overlay persisted human tags/category on a fresh model proposal."""
    effective = db.resolve_classification_override(
        note_path,
        proposed_tags=list(cognition_data.get("tags") or []),
        proposed_category=cognition_data.get("category") or None,
    )
    if not effective["overridden"]:
        return cognition_data
    return {
        **cognition_data,
        "tags": effective["tags"],
        "category": effective["category"] or "",
    }


class ClassificationOverridesMixin(DbBase):
    """Persist manual classification and apply it over model proposals."""

    def resolve_classification_override(
        self,
        note_path: str,
        *,
        proposed_tags: list[str],
        proposed_category: str | None,
    ) -> dict:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT category, tags_json, updated_at, source "
                "FROM classification_overrides WHERE note_path = ?",
                (note_path,),
            ).fetchone()
        if row is None:
            return {
                "tags": list(proposed_tags),
                "category": proposed_category,
                "overridden": False,
                "updated_at": None,
                "source": None,
            }
        return {
            "tags": _decode_tags(row[1]),
            "category": row[0],
            "overridden": True,
            "updated_at": row[2],
            "source": row[3],
        }

    def set_classification_override(
        self,
        *,
        note_id: int,
        category: str | None,
        tags: list[str],
        updated_at: str,
        source: str = "web",
    ) -> dict:
        tags_json = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
        with self._get_connection() as conn:
            note = conn.execute(
                "SELECT path, title, category FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if note is None:
                raise LookupError(f"note {note_id} not found")
            note_path, title, previous_category = note
            previous_tags = [
                row[0] for row in conn.execute(
                    "SELECT t.name FROM tags t "
                    "JOIN note_tags nt ON nt.tag_id = t.id "
                    "WHERE nt.note_id = ? ORDER BY t.name",
                    (note_id,),
                ).fetchall()
            ]
            conn.execute(
                "UPDATE notes SET category = ? WHERE id = ?",
                (category, note_id),
            )
            conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
            for tag in tags:
                conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                tag_row = conn.execute(
                    "SELECT id FROM tags WHERE name = ?", (tag,),
                ).fetchone()
                if tag_row:
                    conn.execute(
                        "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                        (note_id, tag_row[0]),
                    )
            conn.execute(
                """
                INSERT INTO classification_overrides
                    (note_path, category, tags_json, updated_at, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(note_path) DO UPDATE SET
                    category = excluded.category,
                    tags_json = excluded.tags_json,
                    updated_at = excluded.updated_at,
                    source = excluded.source
                """,
                (note_path, category, tags_json, updated_at, source),
            )
            cursor = conn.execute(
                """
                INSERT INTO classification_override_log
                    (note_path, note_id, previous_category, previous_tags_json,
                     category, tags_json, changed_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note_path, note_id, previous_category,
                    json.dumps(previous_tags, ensure_ascii=False, separators=(",", ":")),
                    category, tags_json, updated_at, source,
                ),
            )
            conn.commit()
            assert cursor.lastrowid is not None  # always set after INSERT
            event_id = int(cursor.lastrowid)
        return {
            "event_id": event_id,
            "note_id": int(note_id),
            "path": note_path,
            "title": title or "",
            "category": category,
            "tags": list(tags),
            "previous_category": previous_category,
            "previous_tags": previous_tags,
            "updated_at": updated_at,
            "source": source,
        }

    def get_note_classifications(self, note_ids: Iterable[int]) -> dict[int, dict]:
        unique = sorted({int(note_id) for note_id in note_ids})
        if not unique:
            return {}
        out: dict[int, dict] = {}
        with self._get_connection() as conn:
            for start in range(0, len(unique), _SQL_BATCH):
                batch = unique[start:start + _SQL_BATCH]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    "SELECT n.id, n.category, o.tags_json, o.updated_at, o.source, "
                    "o.note_path IS NOT NULL "
                    "FROM notes n LEFT JOIN classification_overrides o "
                    "ON o.note_path = n.path "
                    f"WHERE n.id IN ({marks})",
                    tuple(batch),
                ).fetchall()
                for note_id, category, override_tags, updated_at, source, manual in rows:
                    out[int(note_id)] = {
                        "category": category,
                        "tags": _decode_tags(override_tags) if manual else [],
                        "manual_override": bool(manual),
                        "override_updated_at": updated_at,
                        "override_source": source,
                    }
                tag_rows = conn.execute(
                    "SELECT nt.note_id, t.name FROM note_tags nt "
                    "JOIN tags t ON t.id = nt.tag_id "
                    f"WHERE nt.note_id IN ({marks}) ORDER BY t.name",
                    tuple(batch),
                ).fetchall()
                for note_id, tag in tag_rows:
                    if int(note_id) in out and not out[int(note_id)]["manual_override"]:
                        out[int(note_id)]["tags"].append(tag)
        return out
