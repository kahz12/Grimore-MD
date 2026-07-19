"""
Black Mirror persistence: extracted claims, flagged contradictions, and
the scan audit trail. The detection logic lives in
:mod:`grimore.cognition.mirror`; this is its storage surface.
"""
from datetime import datetime
from typing import Optional

from grimore.memory._base import DbBase


class MirrorStoreMixin(DbBase):
    """Claims, contradictions, and mirror-run rows for :class:`Database`."""

    # ── Black Mirror: claims ─────────────────────────────────────────────

    def replace_claims_for_note(
        self,
        note_path: str,
        claims: list[tuple[str, Optional[int], Optional[int], Optional[bytes]]],
        extracted_at: str,
    ) -> int:
        """Replace this note's claim set, preserving claim ids for
        identical text across re-extractions.

        This is what makes the "dismissed contradictions stay dismissed"
        invariant work: cascading FK deletes only fire for claims whose
        text actually disappeared from the note. Claims whose text is
        unchanged keep their id (and therefore keep any contradiction
        rows that reference them, including dismissed ones).

        Each entry is ``(text, char_start, char_end, embedding_blob)``.
        ``embedding_blob`` may be None for claims whose embedding failed.
        Returns the *new* row count — i.e. claims that didn't exist for
        this note before this call.
        """
        new_count = 0
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            existing = {
                row[1]: row[0]
                for row in conn.execute(
                    "SELECT id, claim_text FROM claims WHERE note_path = ?",
                    (note_path,),
                )
            }
            new_texts = {text for text, _, _, _ in claims}
            for text, old_id in existing.items():
                if text not in new_texts:
                    conn.execute("DELETE FROM claims WHERE id = ?", (old_id,))
            for text, cs, ce, blob in claims:
                if text in existing:
                    conn.execute(
                        """
                        UPDATE claims
                        SET char_start = ?, char_end = ?, embedding = ?, extracted_at = ?
                        WHERE id = ?
                        """,
                        (cs, ce, blob, extracted_at, existing[text]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO claims
                            (note_path, claim_text, char_start, char_end, embedding, extracted_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (note_path, text, cs, ce, blob, extracted_at),
                    )
                    new_count += 1
            conn.commit()
        return new_count

    def get_claim_extraction_state(self) -> list[tuple[str, str]]:
        """``[(note_path, max_extracted_at), …]`` over every note that has
        at least one claim row. Used by Mirror.scan to decide which notes
        need re-extraction (mtime > extracted_at)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT note_path, MAX(extracted_at)
                FROM claims
                GROUP BY note_path
                """
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_all_claims_with_vectors(self) -> list[tuple[int, str, str, bytes]]:
        """``[(claim_id, note_path, claim_text, embedding_blob), …]``.

        Returns rows with non-NULL embeddings only — claim rows whose
        embedding failed to compute are excluded from neighbor search.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, note_path, claim_text, embedding
                FROM claims
                WHERE embedding IS NOT NULL
                """
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def count_claims(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
        return int(row[0]) if row else 0

    # ── Black Mirror: contradictions ─────────────────────────────────────

    def contradiction_pair_exists(self, claim_a_id: int, claim_b_id: int) -> bool:
        """Whether any row already references this canonical claim pair.

        Used by Mirror.scan to avoid re-running the LLM contradiction
        check for pairs we've already seen — including dismissed and
        resolved pairs (that's the whole point of dismissal-persistence).
        """
        a, b = (claim_a_id, claim_b_id) if claim_a_id < claim_b_id else (claim_b_id, claim_a_id)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM contradictions WHERE claim_a_id = ? AND claim_b_id = ? LIMIT 1",
                (a, b),
            ).fetchone()
        return row is not None

    def insert_contradiction(
        self,
        claim_a_id: int,
        claim_b_id: int,
        severity: str,
        explanation: str,
        detected_at: str,
    ) -> Optional[int]:
        """Insert a contradiction with canonical order ``a < b``.

        Returns the new row id, or ``None`` when the pair already exists
        (UNIQUE on (claim_a_id, claim_b_id)) — that's the dismissal-persists
        invariant: dismissed pairs aren't re-inserted as 'open'.
        """
        a, b = (claim_a_id, claim_b_id) if claim_a_id < claim_b_id else (claim_b_id, claim_a_id)
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO contradictions
                    (claim_a_id, claim_b_id, severity, explanation, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (a, b, severity, explanation, detected_at),
            )
            conn.commit()
            return cur.lastrowid if cur.rowcount else None

    def list_contradictions(self, status: Optional[str] = "open") -> list[tuple]:
        """Rows for ``mirror`` listing.

        Returns ``(id, severity, explanation, status, detected_at,
        note_a, note_b, claim_a_text, claim_b_text)`` joined with the
        claims table. ``status=None`` returns rows in every state.
        """
        sql = """
            SELECT c.id, c.severity, c.explanation, c.status, c.detected_at,
                   a.note_path, b.note_path,
                   a.claim_text, b.claim_text
            FROM contradictions c
            JOIN claims a ON a.id = c.claim_a_id
            JOIN claims b ON b.id = c.claim_b_id
        """
        params: tuple = ()
        if status is not None:
            sql += " WHERE c.status = ?"
            params = (status,)
        sql += """
            ORDER BY
              CASE c.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
              c.detected_at DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [tuple(r) for r in rows]

    def get_contradiction(self, contradiction_id: int):
        """Full row for ``mirror show <id>``.

        ``(id, severity, explanation, status, detected_at, resolved_at,
        claim_a_id, claim_b_id, note_a, note_b, claim_a_text,
        claim_b_text, char_start_a, char_end_a, char_start_b, char_end_b)``.
        """
        sql = """
            SELECT c.id, c.severity, c.explanation, c.status, c.detected_at,
                   c.resolved_at,
                   c.claim_a_id, c.claim_b_id,
                   a.note_path, b.note_path,
                   a.claim_text, b.claim_text,
                   a.char_start, a.char_end,
                   b.char_start, b.char_end
            FROM contradictions c
            JOIN claims a ON a.id = c.claim_a_id
            JOIN claims b ON b.id = c.claim_b_id
            WHERE c.id = ?
        """
        with self._get_connection() as conn:
            row = conn.execute(sql, (contradiction_id,)).fetchone()
        return tuple(row) if row else None

    def set_contradiction_status(self, contradiction_id: int, status: str) -> bool:
        """Update ``status`` and (when terminal) ``resolved_at``.

        Returns False if no row exists with that id, so the surface can
        say "no such contradiction" instead of silently doing nothing.
        """
        if status not in {"open", "dismissed", "resolved"}:
            raise ValueError(f"invalid contradiction status: {status!r}")
        now = datetime.now().isoformat() if status in {"dismissed", "resolved"} else None
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE contradictions
                SET status = ?, resolved_at = ?
                WHERE id = ?
                """,
                (status, now, contradiction_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def count_open_contradictions(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM contradictions WHERE status = 'open'"
            ).fetchone()
        return int(row[0]) if row else 0

    # ── Black Mirror: runs ───────────────────────────────────────────────

    def record_mirror_run(
        self,
        ran_at: str,
        notes_scanned: int,
        claims_extracted: int,
        pairs_checked: int,
        contradictions_found: int,
    ) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO mirror_runs
                    (ran_at, notes_scanned, claims_extracted,
                     pairs_checked, contradictions_found)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ran_at, notes_scanned, claims_extracted, pairs_checked, contradictions_found),
            )
            conn.commit()
            assert cur.lastrowid is not None  # always set after INSERT
            return int(cur.lastrowid)

    def latest_mirror_run_time(self) -> Optional[str]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(ran_at) FROM mirror_runs"
            ).fetchone()
        return row[0] if row and row[0] else None
