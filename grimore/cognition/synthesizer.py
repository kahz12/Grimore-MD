"""
Synthesizer — distillation notes.

Given a *selector* (tag or category), pull every note that matches,
exclude any note already flagged ``grimore_generated: true`` in
frontmatter, and ask the LLM to fold the lot into a single reference
note. Disagreements are preserved as ``> conflict:`` blockquotes rather
than silently picked between — Mirror is the place to surface those.

The output lands in ``<vault>/_synthesis/<slug>_<YYYY-MM-DD>.md`` with
``grimore_generated: true`` so subsequent ``distill`` runs skip it.
The rest of the pipeline (scan/connect/Mirror) processes generated
notes normally — the flag only excludes them from the next distill.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter
import yaml

from grimore.cognition.embedder import Embedder
from grimore.session import Session
from grimore.utils.atomic import atomic_write
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)


# Default per-note passage budget: the K best chunks (by similarity to
# the centroid) from each note. Bounded so a single huge note can't
# dominate the LLM context window.
_DEFAULT_PASSAGES_PER_NOTE = 3
# Hard cap on the prompt body. Mirrors Oracle's headroom strategy: leave
# room for the system prompt + the model's own answer space within a
# typical 32k-token local window.
_SYNTHESIZER_MAX_CONTEXT_CHARS = 16_000
# Where generated notes live, relative to the vault root.
SYNTHESIS_DIRNAME = "_synthesis"
# Frontmatter flag that marks a note as Synthesizer output (and therefore
# excluded from the source set of subsequent distills).
GENERATED_FLAG_KEY = "grimore_generated"


_SYSTEM_PROMPT = (
    "You synthesize a set of personal notes into a single reference note.\n"
    "Output ONLY a JSON object of the form:\n"
    '  {"title": "<short title>",\n'
    '   "body":  "<markdown body, no frontmatter>"}\n'
    "Rules for the body:\n"
    "  • Markdown only. Use ## headings to organise sections.\n"
    "  • Cite sources inline as [[<note title>]] when a fact comes from\n"
    "    a single note.\n"
    "  • When two sources DISAGREE, do NOT pick a side. Render the\n"
    "    disagreement as a blockquote that begins with `> conflict:` so\n"
    "    Mirror can surface it later. Example:\n"
    "      > conflict: [[Note A]] says X, but [[Note B]] says Y.\n"
    "  • Do not invent facts. If the sources do not support a claim,\n"
    "    leave it out.\n"
    "  • Keep it concise — a synthesis, not a transcription."
)


@dataclass
class SourcePassage:
    """One source-note + its top-K passages by centroid similarity."""
    note_id: int
    note_path: str
    title: str
    passages: list[str]
    available_passages: int = 0


@dataclass
class SynthesisReport:
    """Outcome of one ``distill`` invocation."""
    selector: str
    output_path: Optional[str]
    sources: list[str]
    notes_considered: int
    notes_excluded_generated: int
    notes_used: int
    title: Optional[str] = None
    skipped_reason: Optional[str] = None
    body: Optional[str] = None
    source_details: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    review_status: str = "draft"

    def as_dict(self) -> dict:
        """JSON-safe representation used by the web preview."""
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slugify(text: str) -> str:
    """File-safe slug. Keeps ascii alphanumerics + ``-``; folds anything
    else (including ``/``) to ``_``. Length-capped so deep category
    paths don't produce a 200-char filename."""
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "_", text).strip("_")
    return (cleaned or "synthesis")[:60]


class Synthesizer:
    """Engine that turns a selector → distilled markdown note.

    Holds a :class:`Session` so the embedder + router stay warm in the
    interactive shell, exactly like Chronicler / Mirror.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── public API ─────────────────────────────────────────────────────

    def distill(
        self,
        *,
        tag: Optional[str] = None,
        category: Optional[str] = None,
        passages_per_note: int = _DEFAULT_PASSAGES_PER_NOTE,
        dry_run: bool = False,
        change_journal=None,
    ) -> SynthesisReport:
        """Resolve the selector, build the source set, call the LLM, and
        write the output note (unless ``dry_run`` is True).

        Exactly one of ``tag`` / ``category`` must be provided.
        """
        selector_kind, selector_value = self._normalize_selector(tag, category)
        selector = f"{selector_kind}:{selector_value}"

        candidate_notes = self._resolve_selector(selector_kind, selector_value)
        if not candidate_notes:
            coverage = self._coverage(
                matched=0, eligible=0, sources=[], included=[], dropped=[],
                excluded_generated=0, passages_per_note=passages_per_note,
                prompt_chars=0,
            )
            coverage["status"] = "unavailable"
            coverage["limitations"].append("no_matching_notes")
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[],
                notes_considered=0,
                notes_excluded_generated=0,
                notes_used=0,
                skipped_reason="No notes match the selector.",
                coverage=coverage,
                review_status="unavailable",
            )

        # Exclude Synthesizer output from the sources: distilling a previous
        # distillation would feed the model its own text back (feedback loop).
        usable: list[tuple[int, str, str]] = []
        excluded = 0
        for nid, npath, ntitle in candidate_notes:
            if self._is_generated_note(npath):
                excluded += 1
                continue
            usable.append((nid, npath, ntitle))

        if not usable:
            coverage = self._coverage(
                matched=len(candidate_notes), eligible=0, sources=[], included=[],
                dropped=[], excluded_generated=excluded,
                passages_per_note=passages_per_note, prompt_chars=0,
            )
            coverage["status"] = "unavailable"
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[],
                notes_considered=len(candidate_notes),
                notes_excluded_generated=excluded,
                notes_used=0,
                skipped_reason=(
                    "Every matching note is grimore_generated; "
                    "nothing to distill."
                ),
                coverage=coverage,
                review_status="unavailable",
            )

        sources = self._gather_sources(usable, passages_per_note=passages_per_note)
        if not sources:
            coverage = self._coverage(
                matched=len(candidate_notes), eligible=len(usable), sources=[],
                included=[], dropped=[], excluded_generated=excluded,
                passages_per_note=passages_per_note, prompt_chars=0,
            )
            coverage["status"] = "unavailable"
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[],
                notes_considered=len(candidate_notes),
                notes_excluded_generated=excluded,
                notes_used=0,
                source_details=self._source_details([], [], eligible=usable),
                coverage=coverage,
                skipped_reason=(
                    "No embedded passages found for the matching notes. "
                    "Run `grimore scan --no-dry-run` first."
                ),
                review_status="unavailable",
            )

        prompt, included_sources, dropped_sources = self._build_prompt_plan(
            selector_kind, selector_value, sources,
        )
        coverage = self._coverage(
            matched=len(candidate_notes), eligible=len(usable), sources=sources,
            included=included_sources, dropped=dropped_sources,
            excluded_generated=excluded, passages_per_note=passages_per_note,
            prompt_chars=len(prompt),
        )
        source_details = self._source_details(
            sources, included_sources, eligible=usable,
        )
        if not included_sources:
            coverage["status"] = "unavailable"
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[],
                notes_considered=len(candidate_notes),
                notes_excluded_generated=excluded,
                notes_used=0,
                skipped_reason="The context limit excluded every source passage.",
                source_details=source_details,
                coverage=coverage,
                review_status="unavailable",
            )
        result = self.session.router.complete(
            prompt, system_prompt=_SYSTEM_PROMPT, json_format=True
        )
        if not isinstance(result, dict):
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[s.title for s in included_sources],
                notes_considered=len(candidate_notes),
                notes_excluded_generated=excluded,
                notes_used=len(included_sources),
                skipped_reason="LLM returned no structured response.",
                source_details=source_details,
                coverage=coverage,
                review_status="unavailable",
            )

        title = self._extract_title(result, selector_kind, selector_value)
        body = result.get("body")
        if not isinstance(body, str) or not body.strip():
            return SynthesisReport(
                selector=selector,
                output_path=None,
                sources=[s.title for s in included_sources],
                notes_considered=len(candidate_notes),
                notes_excluded_generated=excluded,
                notes_used=len(included_sources),
                title=title,
                skipped_reason="LLM payload had no usable body.",
                source_details=source_details,
                coverage=coverage,
                review_status="unavailable",
            )

        self._review_citations(body, included_sources, coverage)

        out_path = self._write_synthesis_note(
            selector_kind=selector_kind,
            selector_value=selector_value,
            title=title,
            body=body,
            sources=included_sources,
            coverage=coverage,
            dry_run=dry_run,
            change_journal=change_journal,
        )

        return SynthesisReport(
            selector=selector,
            output_path=str(out_path) if out_path else None,
            sources=[s.title for s in included_sources],
            notes_considered=len(candidate_notes),
            notes_excluded_generated=excluded,
            notes_used=len(included_sources),
            title=title,
            body=body.strip(),
            source_details=source_details,
            coverage=coverage,
        )

    # ── selector resolution ────────────────────────────────────────────

    @staticmethod
    def _normalize_selector(
        tag: Optional[str], category: Optional[str]
    ) -> tuple[str, str]:
        if tag and category:
            raise ValueError("Provide --tag OR --category, not both.")
        if tag:
            value = tag.strip()
            if not value:
                raise ValueError("--tag must not be empty.")
            return "tag", value
        if category:
            value = category.strip().rstrip("/")
            if not value:
                raise ValueError("--category must not be empty.")
            return "category", value
        raise ValueError("A selector is required: pass --tag or --category.")

    def _resolve_selector(
        self, kind: str, value: str
    ) -> list[tuple[int, str, str]]:
        """Return ``[(note_id, path, title), …]`` matching the selector.

        Categories include descendants by design (matches the rest of
        the CLI's behaviour for hierarchical paths).
        """
        if kind == "tag":
            return self.session.db.get_notes_by_tag(value)
        if kind == "category":
            return self.session.db.get_notes_by_category(value, recursive=True)
        raise ValueError(f"Unknown selector kind: {kind!r}")

    @staticmethod
    def _is_generated_note(path: str) -> bool:
        """Whether the note's frontmatter sets ``grimore_generated: true``.

        File-read failures are treated as "not generated" — the LLM
        prompt-build path can still fail safely later if the file is
        truly broken.
        """
        try:
            post = frontmatter.load(path)
        except Exception:
            return False
        return bool(post.metadata.get(GENERATED_FLAG_KEY))

    # ── source gathering ───────────────────────────────────────────────

    def _gather_sources(
        self,
        notes: list[tuple[int, str, str]],
        *,
        passages_per_note: int,
    ) -> list[SourcePassage]:
        """Compute centroid over the resolved notes' chunks, then keep
        the top-K chunks per note ranked by similarity to that centroid.

        Centroid math: average the unit-normalized chunk vectors then
        re-normalize. Equivalent to the document-mean-then-normalize
        recipe used in standard centroid retrieval.
        """
        embedder = self.session.embedder
        chunks_by_note: dict[int, list[tuple[str, list[float]]]] = {}
        wanted_ids = {nid for nid, _, _ in notes}
        for nid, text, blob in self.session.db.get_embeddings_for_notes(wanted_ids):
            try:
                vec = embedder.deserialize_vector(blob)
            except Exception:
                continue
            chunks_by_note.setdefault(nid, []).append((text, vec))
        if not chunks_by_note:
            return []

        all_vecs = [v for chunks in chunks_by_note.values() for _, v in chunks]
        centroid = self._centroid(all_vecs)
        if centroid is None:
            return []

        out: list[SourcePassage] = []
        title_by_id = {nid: title for nid, _, title in notes}
        path_by_id = {nid: path for nid, path, _ in notes}
        for nid, chunks in chunks_by_note.items():
            ranked = sorted(
                chunks,
                key=lambda tv: Embedder.dot_product(centroid, tv[1]),
                reverse=True,
            )
            top_passages = [text for text, _ in ranked[:passages_per_note]]
            if not top_passages:
                continue
            out.append(SourcePassage(
                note_id=nid,
                note_path=path_by_id[nid],
                title=title_by_id.get(nid) or Path(path_by_id[nid]).stem,
                passages=top_passages,
                available_passages=len(chunks),
            ))
        # Stable ordering by title so prompt + frontmatter sources line up
        # and outputs are deterministic for tests.
        out.sort(key=lambda s: (s.title, s.note_path))
        return out

    @staticmethod
    def _centroid(vectors: list[list[float]]) -> Optional[list[float]]:
        """Mean of unit-normalized vectors, re-normalized. Returns None
        on empty input or zero-magnitude mean."""
        if not vectors:
            return None
        dim = len(vectors[0])
        if dim == 0:
            return None
        acc = [0.0] * dim
        for v in vectors:
            if len(v) != dim:
                continue
            for i, x in enumerate(v):
                acc[i] += x
        n = len(vectors)
        mean = [x / n for x in acc]
        normed = Embedder.normalize(mean)
        if not any(normed):
            return None
        return normed

    # ── prompt + output ────────────────────────────────────────────────

    def _build_prompt(
        self, kind: str, value: str, sources: list[SourcePassage]
    ) -> str:
        """Compose the user-prompt with the source excerpts.

        Keeps adding sources until the running total would exceed the
        char cap; truncated entries are skipped wholesale (don't tear
        a wrap_untrusted block, same defence Oracle uses).
        """
        prompt, _included, _dropped = self._build_prompt_plan(kind, value, sources)
        return prompt

    def _build_prompt_plan(
        self, kind: str, value: str, sources: list[SourcePassage],
    ) -> tuple[str, list[SourcePassage], list[SourcePassage]]:
        """Return prompt plus the sources included or omitted by its cap."""
        header = (
            f"Selector: {kind}={value}\n"
            f"Use the excerpts below as your only source material. "
            f"Cite each note by its [[title]].\n\n"
        )
        parts: list[str] = []
        included: list[SourcePassage] = []
        dropped: list[SourcePassage] = []
        used = len(header)
        for src in sources:
            block_lines = [f"--- Source: [[{src.title}]] ---"]
            for passage in src.passages:
                safe = SecurityGuard.wrap_untrusted(
                    SecurityGuard.sanitize_prompt(passage),
                    label="passage",
                )
                block_lines.append(safe)
            block = "\n".join(block_lines)
            extra = len(block) + (2 if parts else 0)  # for the join
            if used + extra > _SYNTHESIZER_MAX_CONTEXT_CHARS:
                logger.info(
                    "synthesizer_source_dropped_for_size",
                    title=src.title,
                    used_chars=used,
                    cap=_SYNTHESIZER_MAX_CONTEXT_CHARS,
                )
                dropped.append(src)
                continue
            parts.append(block)
            included.append(src)
            used += extra
        return header + "\n\n".join(parts), included, dropped

    @staticmethod
    def _source_details(
        sources: list[SourcePassage],
        included: list[SourcePassage],
        *,
        eligible: Optional[list[tuple[int, str, str]]] = None,
    ) -> list[dict]:
        included_ids = {source.note_id for source in included}
        source_by_id = {source.note_id: source for source in sources}
        candidates = eligible or [
            (source.note_id, source.note_path, source.title) for source in sources
        ]
        details = []
        for note_id, note_path, title in candidates:
            source = source_by_id.get(note_id)
            details.append({
                "note_id": note_id,
                "path": note_path,
                "title": title or Path(note_path).stem,
                "passages": list(source.passages) if source else [],
                "passages_selected": len(source.passages) if source else 0,
                "passages_available": source.available_passages if source else 0,
                "included_in_prompt": note_id in included_ids,
                "omission_reason": (
                    None if note_id in included_ids
                    else "context_limit" if source
                    else "missing_indexed_passages"
                ),
            })
        return sorted(details, key=lambda item: (item["title"], item["path"]))

    @staticmethod
    def _coverage(
        *,
        matched: int,
        eligible: int,
        sources: list[SourcePassage],
        included: list[SourcePassage],
        dropped: list[SourcePassage],
        excluded_generated: int,
        passages_per_note: int,
        prompt_chars: int,
    ) -> dict:
        missing = max(0, eligible - len(sources))
        representative = any(
            len(source.passages) < source.available_passages for source in sources
        )
        limitations = []
        if excluded_generated:
            limitations.append("generated_notes_excluded")
        if missing:
            limitations.append("missing_indexed_passages")
        if dropped:
            limitations.append("context_limit")
        if representative:
            limitations.append("representative_passages_only")
        return {
            "status": "partial" if limitations else "complete",
            "notes": {
                "matched": matched,
                "eligible": eligible,
                "with_passages": len(sources),
                "included_in_prompt": len(included),
                "without_passages": missing,
                "excluded_generated": excluded_generated,
                "omitted_by_context": len(dropped),
            },
            "passages": {
                "available": sum(source.available_passages for source in sources),
                "selected": sum(len(source.passages) for source in sources),
                "per_note_limit": passages_per_note,
            },
            "context": {
                "characters": prompt_chars,
                "max_characters": _SYNTHESIZER_MAX_CONTEXT_CHARS,
            },
            "limitations": limitations,
            "citations": {"cited": [], "uncited": [], "unknown": []},
        }

    @staticmethod
    def _review_citations(
        body: str, sources: list[SourcePassage], coverage: dict,
    ) -> None:
        labels = [match.strip() for match in re.findall(r"\[\[([^\[\]]+)\]\]", body)]
        cited_names = {label.split("#", 1)[0].strip() for label in labels}
        known = {source.title for source in sources}
        citations = {
            "cited": sorted(cited_names & known),
            "uncited": sorted(known - cited_names),
            "unknown": sorted(cited_names - known),
        }
        coverage["citations"] = citations
        if citations["unknown"]:
            coverage["limitations"].append("unknown_citations")
        if citations["uncited"]:
            coverage["limitations"].append("sources_not_cited")
        coverage["status"] = "partial" if coverage["limitations"] else "complete"

    @staticmethod
    def _extract_title(
        payload: dict, selector_kind: str, selector_value: str
    ) -> str:
        raw = payload.get("title")
        if isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned:
                return cleaned
        return f"Synthesis: {selector_kind} {selector_value}"

    def _write_synthesis_note(
        self,
        *,
        selector_kind: str,
        selector_value: str,
        title: str,
        body: str,
        sources: list[SourcePassage],
        coverage: dict,
        dry_run: bool,
        change_journal=None,
    ) -> Optional[Path]:
        """Atomic-write the markdown file. Path is validated to live
        inside ``<vault>/_synthesis/`` to defend against a maliciously
        crafted selector smuggling traversal characters.
        """
        vault_root = self.session.vault_root.resolve()
        slug = f"{selector_kind}-{_slugify(selector_value)}_{_today_slug()}"
        target = vault_root / SYNTHESIS_DIRNAME / f"{slug}.md"
        # Re-validate even though we control every input — defence in
        # depth, mirrors the rest of the codebase's path handling.
        SecurityGuard.resolve_within_vault(target, vault_root)

        meta = {
            GENERATED_FLAG_KEY: True,
            "title": title,
            "selector": f"{selector_kind}:{selector_value}",
            "sources": [s.note_path for s in sources],
            "generated_at": _now_iso(),
            "synthesis_review": "pending",
            "coverage": coverage,
        }
        # ``yaml.safe_dump`` keeps the keys ordered as inserted (we set
        # default_flow_style=False so list values render as block-style).
        front = yaml.safe_dump(
            meta,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        document = f"---\n{front}---\n\n# {title}\n\n{body.strip()}\n"
        if dry_run:
            logger.info(
                "synthesizer_dry_run",
                target=str(target),
                bytes=len(document.encode("utf-8")),
            )
            return target
        if change_journal is not None:
            change_journal.capture_before(target)
        atomic_write(target, lambda fh: fh.write(document.encode("utf-8")), mode="wb")
        if change_journal is not None:
            change_journal.capture_after(target)
        return target
