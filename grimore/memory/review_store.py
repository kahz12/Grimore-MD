"""Persistent state and append-only audit log for review candidates."""
from __future__ import annotations

import json
from typing import Iterable

from grimore.memory._base import DbBase


_DECISIONS = {"confirm", "dismiss", "postpone"}
_SQL_BATCH = 900


def _decode_snapshot(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class ReviewStoreMixin(DbBase):
    """Store the latest candidate decision and every decision event."""

    def record_review_decision(
        self,
        *,
        candidate_id: str,
        kind: str,
        decision: str,
        decided_at: str,
        postpone_until: str | None,
        snapshot: dict,
        source: str = "web",
    ) -> dict:
        if decision not in _DECISIONS:
            raise ValueError(f"invalid review decision: {decision!r}")
        if decision == "postpone" and not postpone_until:
            raise ValueError("postpone_until is required for a postponed candidate")
        if decision != "postpone" and postpone_until is not None:
            raise ValueError("postpone_until is only valid for a postponed candidate")

        snapshot_json = json.dumps(
            snapshot, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        with self._get_connection() as conn:
            previous = conn.execute(
                "SELECT decision FROM review_candidate_states WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            cursor = conn.execute(
                """
                INSERT INTO review_decision_log
                    (candidate_id, kind, decision, decided_at, postpone_until,
                     snapshot_json, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id, kind, decision, decided_at, postpone_until,
                    snapshot_json, source,
                ),
            )
            conn.execute(
                """
                INSERT INTO review_candidate_states
                    (candidate_id, kind, decision, decided_at, postpone_until,
                     snapshot_json, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    kind = excluded.kind,
                    decision = excluded.decision,
                    decided_at = excluded.decided_at,
                    postpone_until = excluded.postpone_until,
                    snapshot_json = excluded.snapshot_json,
                    source = excluded.source
                """,
                (
                    candidate_id, kind, decision, decided_at, postpone_until,
                    snapshot_json, source,
                ),
            )
            conn.commit()
            assert cursor.lastrowid is not None  # always set after INSERT
            event_id = int(cursor.lastrowid)
        return {
            "event_id": event_id,
            "candidate_id": candidate_id,
            "kind": kind,
            "decision": decision,
            "decided_at": decided_at,
            "postpone_until": postpone_until,
            "previous_decision": previous[0] if previous else None,
            "source": source,
            "snapshot": snapshot,
        }

    def get_review_candidate_states(self, candidate_ids: Iterable[str]) -> dict[str, dict]:
        unique = sorted({str(candidate_id) for candidate_id in candidate_ids})
        if not unique:
            return {}
        states: dict[str, dict] = {}
        with self._get_connection() as conn:
            for start in range(0, len(unique), _SQL_BATCH):
                batch = unique[start:start + _SQL_BATCH]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    "SELECT candidate_id, kind, decision, decided_at, "
                    "postpone_until, snapshot_json, source "
                    f"FROM review_candidate_states WHERE candidate_id IN ({marks})",
                    tuple(batch),
                ).fetchall()
                for row in rows:
                    states[row[0]] = {
                        "candidate_id": row[0],
                        "kind": row[1],
                        "decision": row[2],
                        "decided_at": row[3],
                        "postpone_until": row[4],
                        "snapshot": _decode_snapshot(row[5]),
                        "source": row[6],
                    }
        return states

    def list_review_decisions(self, *, limit: int = 50) -> list[dict]:
        bounded = max(1, min(int(limit), 500))
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, candidate_id, kind, decision, decided_at,
                       postpone_until, snapshot_json, source
                FROM review_decision_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        return [{
            "event_id": int(row[0]),
            "candidate_id": row[1],
            "kind": row[2],
            "decision": row[3],
            "decided_at": row[4],
            "postpone_until": row[5],
            "snapshot": _decode_snapshot(row[6]),
            "source": row[7],
        } for row in rows]
