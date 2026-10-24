"""
Black Mirror — claim-level contradiction detection.

Processing pipeline:

  Claim extraction
      For each note (or each note whose mtime > extracted_at), the
      LLM produces atomic factual claims. Each claim is embedded.

  Candidate generation
      For each claim, find its K nearest neighbor claims across other
      notes via cosine similarity. Drop pairs below a similarity floor.

  Contradiction check
      For each candidate pair, the LLM returns a structured verdict.
      Positives are persisted; pairs already seen (in any status) are
      skipped, which is the dismissal-persistence guarantee.

The engine preserves dismissed/resolved status across re-scans by
keeping claim ids stable when the underlying claim text is unchanged
(see :py:meth:`grimore.memory.db.Database.replace_claims_for_note`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from grimore.cognition.claims import ClaimExtractor
from grimore.ingest.parser import MarkdownParser, read_document_text
from grimore.session import Session
from grimore.utils.logger import get_logger

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
    coverage_gaps: list[ClaimCoverageGap] = field(default_factory=list)


@dataclass(frozen=True)
class ClaimCoverageGap:
    note_path: str
    section: int
    part: int
    page: Optional[int]
    heading: Optional[str]
    reason: str


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
        cognition = session.config.cognition
        self._extractor = ClaimExtractor(
            session.router,
            max_chars=cognition.claim_max_chars,
            max_sections=cognition.claim_max_sections,
            max_claims_per_section=cognition.claim_max_claims_per_section,
        )

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

        # Stage 0 — extract claims from notes
        notes_to_extract, all_note_paths = self._notes_for_extraction(full=full)
        report.notes_scanned = len(all_note_paths)
        for i, note_path in enumerate(notes_to_extract):
            if progress:
                progress("extract", i, len(notes_to_extract))
            inserted, gaps = self._extract_and_persist(note_path)
            report.claims_extracted += inserted
            report.coverage_gaps.extend(gaps)

        # Stages 1+2 — pair candidate claims and check for contradictions
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

        neighbors_by_claim = self._neighbor_map(
            decoded, focus_ids=focus_ids, top_k=top_k,
        )

        seen_pairs: set[tuple[int, int]] = set()
        for j, (cid, npath, ctext, _vec) in enumerate(decoded):
            if focus_ids is not None and cid not in focus_ids:
                continue
            if progress:
                progress("pairs", j, len(decoded))
            neighbors = neighbors_by_claim.get(cid, [])
            for n_cid, n_npath, n_ctext, _score in neighbors:
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
                # (`grimore prune`). Mirror just skips it.
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

    def _extract_and_persist(self, note_path: str) -> tuple[int, list[ClaimCoverageGap]]:
        """Read note → extract claims → embed → persist."""
        try:
            note = MarkdownParser().parse_file(
                note_path,
                vault_root=self.session.vault_root,
                config=self.session.config,
            )
        except (OSError, UnicodeError, ValueError) as e:
            logger.warning("mirror_read_failed", path=note_path, error=str(e))
            return 0, []
        extraction = self._extractor.extract_with_coverage(
            note.content, sections=note.sections,
        )
        gaps = [ClaimCoverageGap(
            note_path, ref.section, ref.part, ref.page, ref.heading, ref.reason,
        ) for ref in extraction.unanalyzed_sections]
        if extraction.failed:
            # The model failed on a selected source part, so the result is
            # incomplete: return without touching the stored claims or
            # recording a successful-extraction timestamp.
            logger.warning("mirror_claim_extraction_incomplete", path=note_path)
            return 0, gaps
        extracted = extraction.claims
        if not extracted:
            # Even if no claims came back, drop existing rows so we
            # don't keep stale claims around.
            self.session.db.replace_claims_for_note(note_path, [], _now_iso())
            return 0, gaps
        embedder = self.session.embedder
        vectors = embedder.embed_batch([claim.text for claim in extracted])
        rows: list[tuple[str, Optional[int], Optional[int], Optional[bytes]]] = []
        for c, vec in zip(extracted, vectors, strict=True):
            blob = embedder.serialize_vector(vec) if vec else None
            rows.append((c.text, c.char_start, c.char_end, blob))
        return self.session.db.replace_claims_for_note(
            note_path, rows, _now_iso()
        ), gaps

    def _neighbor_map(
        self,
        all_decoded: list[tuple[int, str, str, list[float]]],
        *,
        focus_ids: Optional[set[int]],
        top_k: int,
    ) -> dict[int, list[tuple[int, str, str, float]]]:
        """Vectorized cross-note top-K candidates, bounded in memory.

        Claims are grouped by embedding dimension so a partially migrated DB
        remains usable. Matrix multiplication runs in blocks: memory is
        ``O(block_size * claims)`` instead of materialising the full square
        similarity matrix, and ``argpartition`` avoids sorting every claim.
        """
        if top_k <= 0:
            return {}
        by_dim: dict[int, list[tuple[int, str, str, list[float]]]] = {}
        for row in all_decoded:
            if row[3]:
                by_dim.setdefault(len(row[3]), []).append(row)

        out: dict[int, list[tuple[int, str, str, float]]] = {}
        block_size = 256
        for rows in by_dim.values():
            matrix = np.asarray([row[3] for row in rows], dtype=np.float32)
            note_paths = np.asarray([row[1] for row in rows], dtype=object)
            focus_indices = [
                i for i, row in enumerate(rows)
                if focus_ids is None or row[0] in focus_ids
            ]
            for start in range(0, len(focus_indices), block_size):
                block_indices = focus_indices[start:start + block_size]
                scores = matrix[block_indices] @ matrix.T
                for local_i, source_i in enumerate(block_indices):
                    row_scores = scores[local_i]
                    valid = np.flatnonzero(
                        (row_scores >= self.similarity_floor)
                        & (note_paths != note_paths[source_i])
                    )
                    if valid.size > top_k:
                        valid_scores = row_scores[valid]
                        cutoff = np.partition(valid_scores, -top_k)[-top_k]
                        stronger = valid[valid_scores > cutoff]
                        tied = valid[valid_scores == cutoff]
                        remaining = top_k - stronger.size
                        valid = np.concatenate((stronger, tied[:remaining]))
                    selected = sorted(
                        valid.tolist(),
                        key=lambda i: (-float(row_scores[i]), i),
                    )
                    out[rows[source_i][0]] = [
                        (rows[i][0], rows[i][1], rows[i][2], float(row_scores[i]))
                        for i in selected
                    ]
        return out

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

    def _read_context(
        self, note_path: str, cs: Optional[int], ce: Optional[int],
    ) -> Optional[str]:
        """Return the surrounding paragraph for a claim span, or None.

        Reads the file, strips frontmatter, and returns the paragraph
        containing the (cs, ce) span. Used by ``mirror show`` to give
        the user enough context to judge the contradiction.
        """
        if cs is None or ce is None:
            return None
        try:
            raw = read_document_text(
                note_path,
                vault_root=self.session.vault_root,
                config=self.session.config,
            )
        except (OSError, UnicodeError, ValueError):
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
