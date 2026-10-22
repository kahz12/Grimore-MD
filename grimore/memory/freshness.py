"""
Chronicler freshness tracking: per-note verification timestamps,
staleness windows, and the LLM decay verdicts persisted between runs.
"""
from datetime import datetime, timezone
from typing import Optional

from grimore.memory._base import DbBase


class FreshnessMixin(DbBase):
    """Freshness (Chronicler) rows for :class:`Database`."""

    def list_freshness(self) -> list[tuple[str, int]]:
        """``[(note_path, window_days), …]`` for every freshness row."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT note_path, window_days FROM freshness"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def upsert_freshness(
        self,
        path: str,
        last_verified: Optional[str],
        window_days: int,
        *,
        verification_source: Optional[str] = None,
    ) -> None:
        """Insert if missing; on conflict only the window updates.

        ``last_verified`` is preserved across re-seeds. New notes pass
        ``None`` because indexing is not verification. A timestamp without an
        explicit source is labelled legacy/unspecified rather than invented.
        """
        if last_verified is not None and verification_source is None:
            verification_source = "legacy:unspecified"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO freshness
                    (note_path, last_verified, verification_source, window_days)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(note_path) DO UPDATE SET window_days = excluded.window_days
                """,
                (path, last_verified, verification_source, window_days),
            )
            conn.commit()

    def delete_freshness(self, path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM freshness WHERE note_path = ?", (path,))
            conn.commit()

    def get_freshness_with_notes(self) -> list[
        tuple[str, str, Optional[str], Optional[str], int, Optional[int]]
    ]:
        """Join used by ``chronicler list``.

        Returns rows of ``(path, title, category, last_verified,
        window_days, likely_stale)``.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.path, n.title, n.category,
                       f.last_verified, f.window_days, f.likely_stale
                FROM freshness f
                JOIN notes n ON n.path = f.note_path
                """
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def get_freshness_row(
        self, path: str,
    ) -> Optional[tuple[Optional[str], int, Optional[str], Optional[int]]]:
        """``(last_verified, window_days, decay_check_at, likely_stale)``."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT last_verified, window_days, decay_check_at, likely_stale
                FROM freshness
                WHERE note_path = ?
                """,
                (path,),
            ).fetchone()
        return (row[0], row[1], row[2], row[3]) if row else None

    def touch_freshness_verified(self, path: str) -> bool:
        """Advance ``last_verified`` to now and clear any decay verdict.

        Returns False if no freshness row exists for ``path`` — callers
        treat that as a silent no-op (the user verified a note that
        Chronicler doesn't track because of its category).
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z",
        )
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE freshness
                SET last_verified = ?, verification_source = 'human:chronicler',
                    decay_check_at = NULL, likely_stale = NULL
                WHERE note_path = ?
                """,
                (now, path),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_freshness_decay(self, path: str, likely_stale: bool) -> None:
        """Persist the LLM decay verdict for ``path``."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE freshness
                SET decay_check_at = ?, likely_stale = ?
                WHERE note_path = ?
                """,
                (now, 1 if likely_stale else 0, path),
            )
            conn.commit()

    def get_notes_for_freshness_seed(self) -> list[tuple[str, Optional[str], Optional[str]]]:
        """All notes with the columns needed to seed freshness.

        Returns ``(path, category, last_tagged_or_seen)`` for compatibility.
        The third field is a scan anchor, never a human verification, and
        Chronicler intentionally ignores it when creating a freshness row.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, category, COALESCE(last_tagged, last_seen) FROM notes"
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
