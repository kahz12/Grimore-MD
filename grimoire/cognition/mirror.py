"""
Black Mirror — claim-level contradiction detection.

Pipeline (per the v2.1 plan §1):

  PHASE 0 — claim extraction
      For each note (or each note whose mtime > extracted_at), the
      LLM produces atomic factual claims. Each claim is embedded.

  PHASE 1 — candidate generation
      For each claim, find its K nearest neighbor claims across other
      notes via cosine similarity. Drop pairs below a similarity floor.

  PHASE 2 — contradiction check
      For each candidate pair, the LLM returns a structured verdict.
      Positives are persisted; pairs already seen (in any status) are
      skipped, which is the dismissal-persistence guarantee.

The engine preserves dismissed/resolved status across re-scans by
keeping claim ids stable when the underlying claim text is unchanged
(see :py:meth:`grimoire.memory.db.Database.replace_claims_for_note`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from grimoire.cognition.claims import ClaimExtractor
from grimoire.cognition.embedder import Embedder
from grimoire.session import Session
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


_CONTRADICTION_SYSTEM = (
    "You compare two atomic claims drawn from different personal notes\n"
    "and decide whether they contradict each other.\n"
    "Output ONLY a JSON object of the form:\n"
    '  {"contradicts": <true|false>,\n'
    '   "severity":    "low" | "medium" | "high",\n'
    '   "explanation": "<one short sentence>"}\n'
    "Mark contradicts=true ONLY when both claims cannot be true at the\n"
    "same time about the same subject. Stylistic, scope, or temporal\n"
    "differences alone are NOT contradictions.\n"
    "Severity is high when the claims directly assert opposite facts;\n"
    "medium when they conflict but the framing differs; low when the\n"
    "conflict is partial or context-dependent."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class MirrorRunReport:
    notes_scanned: int = 0
    claims_extracted: int = 0
    pairs_checked: int = 0
    contradictions_found: int = 0


@dataclass
class ContradictionRow:
    """One row of ``mirror`` listing output."""
    id: int
    severity: str
    explanation: str
    status: str
    detected_at: str
    note_a: str
    note_b: str
    claim_a: str
    claim_b: str


@dataclass
class ContradictionDetail:
    """Full row for ``mirror show <id>``."""
    id: int
    severity: str
    explanation: str
    status: str
    detected_at: str
    resolved_at: Optional[str]
    note_a: str
    note_b: str
    claim_a: str
    claim_b: str
    context_a: Optional[str]
    context_b: Optional[str]


class Mirror:
    """Engine that owns claim extraction, neighbor search, and the
    contradiction pipeline. Holds a :class:`Session` so the embedder /
    LLM router stay warm across calls in the interactive shell."""

    # Cosine-similarity floor below which a candidate pair is dropped
    # before it ever reaches the LLM check. Tuned for unit-normalized
    # embeddings of short claim sentences — too low spams the LLM with
    # unrelated pairs, too high misses subtler conflicts.
    DEFAULT_SIMILARITY_FLOOR = 0.55

    def __init__(
        self,
        session: Session,
        *,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    ) -> None:
        self.session = session
        self.similarity_floor = similarity_floor
        self._extractor = ClaimExtractor(session.router)

    # ── scan ───────────────────────────────────────────────────────────

    def scan(
        self,
        *,
        top_k: int = 5,
        full: bool = False,
        progress: Optional[callable] = None,
    ) -> MirrorRunReport:
        """Run the full pipeline.

        ``full=False`` (default) — incremental: re-extract claims only
        for notes whose mtime is newer than the recorded extraction
        timestamp. Pair checks then run only against new/changed claims.

        ``full=True`` — re-extract every note, then re-check every pair
        the dismissal-persistence layer hasn't already filed.

        ``progress`` — optional ``(stage, current, total)`` callback so
        the CLI can render a bar without coupling the engine to Rich.
        """
        report = MirrorRunReport()

        # Phase 0
        notes_to_extract, all_note_paths = self._notes_for_extraction(full=full)
        report.notes_scanned = len(all_note_paths)
        for i, note_path in enumerate(notes_to_extract):
            if progress:
                progress("extract", i, len(notes_to_extract))
            inserted = self._extract_and_persist(note_path)
            report.claims_extracted += inserted

        # Phase 1+2
        all_claims = self.session.db.get_all_claims_with_vectors()
        if not all_claims:
            self._record_run(report)
            return report
        embedder = self.session.embedder
        # Pre-deserialize once so we don't blob-decode in the inner loop.
        decoded = [
            (cid, npath, ctext, embedder.deserialize_vector(blob))
            for cid, npath, ctext, blob in all_claims
        ]

        focus_ids: Optional[set[int]]
        if full:
            focus_ids = None  # check every claim against every other
        else:
            # In incremental mode, only run pair-checks involving claims
            # extracted in this run (or any newer than the last run).
            cutoff = self.session.db.latest_mirror_run_time()
            focus_ids = self._focus_claim_ids(cutoff)

        seen_pairs: set[tuple[int, int]] = set()
        for j, (cid, npath, ctext, vec) in enumerate(decoded):
            if focus_ids is not None and cid not in focus_ids:
                continue
            if progress:
                progress("pairs", j, len(decoded))
            neighbors = self._top_k_neighbors(cid, npath, vec, decoded, top_k=top_k)
            for n_cid, n_npath, n_ctext, score in neighbors:
                a, b = (cid, n_cid) if cid < n_cid else (n_cid, cid)
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if self.session.db.contradiction_pair_exists(a, b):
                    continue  # dismissal-persistence: don't re-flag
                report.pairs_checked += 1
                verdict = self._check_contradiction(ctext, n_ctext, npath, n_npath)
                if verdict and verdict.get("contradicts"):
                    new_id = self.session.db.insert_contradiction(
                        a, b,
                        severity=self._normalize_severity(verdict.get("severity")),
                        explanation=str(verdict.get("explanation") or "").strip(),
                        detected_at=_now_iso(),
                    )
                    if new_id is not None:
                        report.contradictions_found += 1

        self._record_run(report)
        return report

    # ── surface helpers ────────────────────────────────────────────────

    def list_open(self) -> list[ContradictionRow]:
        rows = self.session.db.list_contradictions(status="open")
        return [
            ContradictionRow(
                id=r[0], severity=r[1], explanation=r[2], status=r[3],
                detected_at=r[4], note_a=r[5], note_b=r[6],
                claim_a=r[7], claim_b=r[8],
            )
            for r in rows
        ]

    def show(self, contradiction_id: int) -> Optional[ContradictionDetail]:
        row = self.session.db.get_contradiction(contradiction_id)
        if row is None:
            return None
        (cid, severity, explanation, status, detected_at, resolved_at,
         _claim_a_id, _claim_b_id, note_a, note_b,
         claim_a, claim_b, cs_a, ce_a, cs_b, ce_b) = row
        ctx_a = self._read_context(note_a, cs_a, ce_a)
        ctx_b = self._read_context(note_b, cs_b, ce_b)
        return ContradictionDetail(
            id=cid, severity=severity, explanation=explanation, status=status,
            detected_at=detected_at, resolved_at=resolved_at,
            note_a=note_a, note_b=note_b,
            claim_a=claim_a, claim_b=claim_b,
            context_a=ctx_a, context_b=ctx_b,
        )

    def dismiss(self, contradiction_id: int) -> bool:
        return self.session.db.set_contradiction_status(contradiction_id, "dismissed")

    def resolve(self, contradiction_id: int) -> bool:
        return self.session.db.set_contradiction_status(contradiction_id, "resolved")

    # ── internals ──────────────────────────────────────────────────────

    def _notes_for_extraction(self, *, full: bool) -> tuple[list[str], list[str]]:
        """Return ``(notes_to_extract, all_known_note_paths)``.

        Walks the ``notes`` table — claim extraction follows ``scan``,
        not the bare filesystem, so we extract from the same set of
        notes the rest of the pipeline already knows about.
        """
        with self.session.db._get_connection() as conn:
            rows = conn.execute("SELECT path FROM notes").fetchall()
        all_paths = [r[0] for r in rows]
        if full:
            return list(all_paths), all_paths

        last_extracted = dict(self.session.db.get_claim_extraction_state())
        out: list[str] = []
        for path in all_paths:
            previous = last_extracted.get(path)
            if previous is None:
                out.append(path)
                continue
            try:
                mtime_iso = datetime.fromtimestamp(
                    Path(path).stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except OSError:
                # File vanished — pruning is the user's responsibility
                # (`grimoire prune`). Mirror just skips it.
                continue
            if mtime_iso > previous:
                out.append(path)
        return out, all_paths

    def _focus_claim_ids(self, cutoff: Optional[str]) -> set[int]:
        """Claim ids whose extracted_at is strictly after ``cutoff``.

        With no prior run, every claim is in focus (initial cold pass).
        """
        with self.session.db._get_connection() as conn:
            if cutoff is None:
                rows = conn.execute("SELECT id FROM claims").fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM claims WHERE extracted_at > ?",
                    (cutoff,),
                ).fetchall()
        return {int(r[0]) for r in rows}

    def _extract_and_persist(self, note_path: str) -> int:
        """Read note → extract claims → embed → persist."""
        try:
            content = Path(note_path).read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("mirror_read_failed", path=note_path, error=str(e))
            return 0
        extracted = self._extractor.extract(content)
        if not extracted:
            # Even if no claims came back, drop existing rows so we
            # don't keep stale claims around.
            self.session.db.replace_claims_for_note(note_path, [], _now_iso())
            return 0
        embedder = self.session.embedder
        rows: list[tuple[str, Optional[int], Optional[int], Optional[bytes]]] = []
        for c in extracted:
            vec = embedder.embed(c.text)
            blob = embedder.serialize_vector(vec) if vec else None
            rows.append((c.text, c.char_start, c.char_end, blob))
        return self.session.db.replace_claims_for_note(
            note_path, rows, _now_iso()
        )

    def _top_k_neighbors(
        self,
        cid: int,
        npath: str,
        vec: list[float],
        all_decoded: list[tuple[int, str, str, list[float]]],
        *,
        top_k: int,
    ) -> list[tuple[int, str, str, float]]:
        """Cross-note top-K cosine neighbors, filtered by similarity floor."""
        scored: list[tuple[int, str, str, float]] = []
        for n_cid, n_npath, n_ctext, n_vec in all_decoded:
            if n_cid == cid or n_npath == npath:
                continue
            score = Embedder.dot_product(vec, n_vec)
            if score < self.similarity_floor:
                continue
            scored.append((n_cid, n_npath, n_ctext, score))
        scored.sort(key=lambda r: r[3], reverse=True)
        return scored[:top_k]

    def _check_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        note_a: str,
        note_b: str,
    ) -> Optional[dict]:
        prompt = (
            f"Claim A (from note {Path(note_a).name}):\n  {claim_a}\n\n"
            f"Claim B (from note {Path(note_b).name}):\n  {claim_b}\n"
        )
        result = self.session.router.complete(
            prompt,
            system_prompt=_CONTRADICTION_SYSTEM,
            json_format=True,
        )
        return result if isinstance(result, dict) else None

    @staticmethod
    def _normalize_severity(value: object) -> str:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"low", "medium", "high"}:
                return v
        return "medium"

    @staticmethod
    def _read_context(note_path: str, cs: Optional[int], ce: Optional[int]) -> Optional[str]:
        """Return the surrounding paragraph for a claim span, or None.

        Reads the file, strips frontmatter, and returns the paragraph
        containing the (cs, ce) span. Used by ``mirror show`` to give
        the user enough context to judge the contradiction.
        """
        if cs is None or ce is None:
            return None
        try:
            raw = Path(note_path).read_text(encoding="utf-8")
        except OSError:
            return None
        # If there's frontmatter, the offsets came from `body` (the full
        # file in our extractor) so we use it as-is — no re-mapping.
        if not (0 <= cs < ce <= len(raw)):
            return None
        # Walk outward to nearest blank lines.
        start = raw.rfind("\n\n", 0, cs)
        start = 0 if start == -1 else start + 2
        end = raw.find("\n\n", ce)
        end = len(raw) if end == -1 else end
        snippet = raw[start:end].strip()
        return snippet or None

    def _record_run(self, report: MirrorRunReport) -> None:
        self.session.db.record_mirror_run(
            ran_at=_now_iso(),
            notes_scanned=report.notes_scanned,
            claims_extracted=report.claims_extracted,
            pairs_checked=report.pairs_checked,
            contradictions_found=report.contradictions_found,
        )
