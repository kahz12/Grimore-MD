"""
Chronicler — temporal staleness tracking.

Two signals:

* AGE — purely deterministic, computed from ``days-since-last-verified``
  against a per-category window. Cheap to run on every list.
* DECAY — opt-in LLM check that asks whether a single note references
  technologies, APIs, versions or external resources that have likely
  changed since its verification date.

Only notes whose category resolves to a *finite* freshness window get a
``freshness`` row. Anything else is "never stale" and is invisible to
``chronicler list``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter

from grimoire.session import Session
from grimoire.utils.config import ChroniclerConfig
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime:
    """Best-effort parse of an ISO timestamp; tolerates trailing ``Z``
    and timestamps written without timezone (treats those as UTC)."""
    raw = ts
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def resolve_window_days(category: Optional[str], cfg: ChroniclerConfig) -> int:
    """Return the freshness window in days for a category.

    Matching is case-insensitive prefix matching so a category of
    ``Tech/Linux/Termux`` matches the rule ``tech/``. The first matching
    rule wins (dict insertion order is preserved in Python 3.7+).

    Returns ``0`` when the category is exempt or unrecognized — callers
    treat 0 as "never stale" and skip the freshness row entirely.
    """
    if not category:
        return 0
    normalized = category.lower().rstrip("/") + "/"
    for prefix, window in cfg.windows.items():
        if normalized.startswith(prefix.lower()):
            return int(window)
    return 0


@dataclass
class StaleNote:
    """One row of ``chronicler list`` output, denormalized for display."""
    path: str
    title: str
    category: Optional[str]
    last_verified: str
    window_days: int
    days_overdue: int
    likely_stale: Optional[bool]


@dataclass
class SeedReport:
    added: int = 0
    updated: int = 0
    removed: int = 0


class Chronicler:
    """Engine that owns freshness state.

    Holds a :class:`~grimoire.session.Session` so the same DB connection
    pool and LLM router are reused across calls in the interactive shell.
    """

    def __init__(self, session: Session, cfg: Optional[ChroniclerConfig] = None) -> None:
        self.session = session
        self.cfg = cfg if cfg is not None else session.config.chronicler

    # ── seeding ────────────────────────────────────────────────────────

    def seed(self) -> SeedReport:
        """Walk every note; create/update freshness rows for those whose
        category resolves to a finite window. Drop rows whose category
        no longer matches any rule.

        Idempotent — ``last_verified`` is preserved across re-seeds.
        Re-running with a changed config updates ``window_days`` in place
        without resetting verification.
        """
        db = self.session.db
        report = SeedReport()
        existing = dict(db.list_freshness())
        notes = db.get_notes_for_freshness_seed()
        now = _now_iso()
        for path, category, last_anchor in notes:
            window = resolve_window_days(category, self.cfg)
            if window <= 0:
                if path in existing:
                    db.delete_freshness(path)
                    report.removed += 1
                continue
            if path in existing:
                if existing[path] != window:
                    db.upsert_freshness(path, last_anchor or now, window)
                    report.updated += 1
            else:
                db.upsert_freshness(path, last_anchor or now, window)
                report.added += 1
        return report

    # ── age signal ─────────────────────────────────────────────────────

    def list_stale(self, *, today: Optional[datetime] = None, auto_seed: bool = True) -> list[StaleNote]:
        """Notes past their freshness window, sorted most-overdue first.

        ``today`` is overridable in tests. ``auto_seed`` defaults to True
        so the first call after ``scan`` doesn't need a manual seeding
        step; flip it off in unit tests that want to exercise stale rows
        without rebuilding the seed.
        """
        if auto_seed:
            self.seed()
        now = today or _now_utc()
        out: list[StaleNote] = []
        for path, title, category, last_verified, window_days, likely in (
            self.session.db.get_freshness_with_notes()
        ):
            try:
                lv = _parse_iso(last_verified)
            except ValueError:
                logger.warning("freshness_bad_timestamp", path=path, ts=last_verified)
                continue
            days = (now - lv).days
            if days <= window_days:
                continue
            out.append(StaleNote(
                path=path,
                title=title or Path(path).stem,
                category=category,
                last_verified=last_verified,
                window_days=window_days,
                days_overdue=days - window_days,
                likely_stale=None if likely is None else bool(likely),
            ))
        out.sort(key=lambda r: r.days_overdue, reverse=True)
        return out

    # ── verification ───────────────────────────────────────────────────

    def verify(self, path: str) -> bool:
        """Mark ``path`` as freshly verified.

        Returns True if a row was updated, False when no freshness row
        exists for that note (silent no-op — the user may verify notes
        outside Chronicler's scope without it being an error).
        """
        return self.session.db.touch_freshness_verified(path)

    # ── decay signal (LLM) ─────────────────────────────────────────────

    def check_decay(self, path: str) -> Optional[dict]:
        """Run the LLM decay check on a single note.

        Reads the note from disk so we see the user's most-recent edits,
        not the chunked text stored in ``embeddings``. Persists the
        verdict on the freshness row.

        Returns the parsed LLM payload, or ``None`` when:
          * the note has no freshness row (out of scope)
          * the file can't be read
          * the LLM call fails or returns malformed output
        """
        db = self.session.db
        row = db.get_freshness_row(path)
        if row is None:
            logger.info("decay_skipped_no_freshness_row", path=path)
            return None
        last_verified, _window, _, _ = row

        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("decay_read_failed", path=path, error=str(e))
            return None
        try:
            body = frontmatter.loads(content).content
        except Exception:
            body = content

        prompt = (
            "You audit personal notes for staleness.\n"
            f"This note was last verified on {last_verified}.\n"
            "Examine the content below. Output ONLY a JSON object of the form:\n"
            '  {"likely_stale": <true|false>, "reasons": ["<short reason>", ...]}\n'
            "Mark likely_stale=true ONLY if the note references technologies,\n"
            "APIs, versions, or external resources that may have changed since\n"
            "the verification date. Do NOT flag stable concepts, theory, or\n"
            "personal observations.\n\n"
            f"NOTE CONTENT:\n{body[:4000]}"
        )
        result = self.session.router.complete(prompt, json_format=True)
        if not isinstance(result, dict) or "likely_stale" not in result:
            return None
        likely = bool(result.get("likely_stale"))
        db.update_freshness_decay(path, likely)
        return result
