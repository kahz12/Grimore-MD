"""RAG evaluation harness.

Measures retrieval and answer quality against a checked-in YAML golden set
so future retrieval-quality changes (rerankers, chunkers, backend swaps)
can be evaluated against a stable baseline instead of vibes.

Ranking metrics (``hit_at_1`` / ``hit_at_3`` / ``mrr``) are scored against
the Oracle's *rank-ordered* retrieval list, exposed as ``result["retrieved"]``.
This matters: ``result["sources"]`` is order-destroyed (``ask`` returns it
through ``list(set(...))`` after a context-budget cap), so any rank-sensitive
metric computed over ``sources`` is noise. When ``retrieved`` is absent
(legacy callers / mocks) we fall back to ``sources``, which is only sound for
single-expected-source cases.

Metrics computed per case (and aggregated over the suite):

* ``hit_at_1``        — 1.0 when the top-ranked retrieved source matches any
                        ``expected_sources``, else 0.0. Matching is token-
                        normalised (accent/emoji/case-insensitive), so a short
                        golden title matches a note's decorated H1.
* ``hit_at_3``        — 1.0 when any of the top-3 retrieved sources matches.
* ``recall_at_k``     — fraction of ``expected_sources`` present in the
                        retrieved sources for the ask.
* ``mrr``             — mean reciprocal rank of the first ``expected_sources``
                        hit (0 when none of them appear).
* ``faithfulness``    — ``1 - dropped_citations / total_citations``; 1.0 means
                        every ``[[wikilink]]`` the model emitted was grounded
                        in a retrieved source.
* ``keyword_recall``  — fraction of ``expected_keywords`` present in the
                        answer, matched accent-folded and word-boundary
                        anchored (see :func:`_keyword_pattern`).
* ``answer_relevance``— LLM-as-judge: the local model rates the answer 0-10
                        against the question and we normalise to ``[0, 1]``.
                        Skipped per-case when the router returns None;
                        excluded from the aggregate when no case scored.
* ``latency_s``       — wall-clock for the ``ask`` call. The Oracle also
                        reports a per-stage breakdown (rewrite/embed/retrieve/
                        rerank/generate), aggregated into ``stage_latency`` so a
                        slowdown can be pinned to a stage. (The connector
                        additionally debug-logs the RRF rank inputs — each
                        surviving doc's dense- and BM25-rank.)

Cases are stratified by a ``category`` tag (single-hop, multi-hop, …) so the
report breaks ranking down by query type and shows *where* retrieval struggles.
``negative`` cases — questions whose answer is deliberately not in the vault —
are excluded from the ranking aggregates (their empty expectations would count
as vacuous hits) and instead scored by ``abstention_rate``: did the Oracle
correctly decline rather than confabulate?

Every turn is also classified into a **failure taxonomy** (``classify_outcome``)
naming the first pipeline stage that broke — ``retrieval_miss`` → ``ranking_miss``
→ ``generation_miss`` → ``citation_miss`` (or ``hallucinated`` for a negative
that answered). That turns "the number dropped" into "fix *this* component", and
drives the Failures section + ``pass_rate``.

Two run modes, because retrieval and generation have different cost and
determinism profiles:

* **Full** (default) — ``oracle.ask`` per turn: retrieval ranking *and* the
  generated answer, so every metric is populated.
* **Retrieval-only** (``generate=False``) — ``oracle.retrieve`` per turn:
  ranks sources without an answer LLM call. Fast and deterministic given the
  index, so it's the mode to gate CI on. Generation metrics (faithfulness,
  keyword recall, judge) are reported as ``None``, not 0.0.

Ranking depth is decoupled from context depth: ``top_k`` sizes the answer's
context window, while ``retrieval_k`` (≥ ``top_k``, default 10) sizes the
ranked pool the metrics see — so MRR@10 / Hit@3 are real even behind a
top-5 answer.

The harness never raises on a model failure: a bad ask shows up as zero
recall and a None judge score, not a crashed run. CI should never gate on
the judge metric — it's directional, not absolute.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import statistics
import subprocess
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from grimore.utils.logger import get_logger

logger = get_logger(__name__)

# Schema version of the golden YAML. Bumped only when the file shape
# changes incompatibly; current loaders accept v1.
_GOLDEN_SCHEMA_VERSION = 1

# Allowed keys on each case (top-level and follow-ups). Anything else is a
# typo we want to catch loudly rather than silently ignore.
_CASE_KEYS = {
    "id", "question", "expected_sources", "expected_keywords", "follow_ups",
    "category", "negative",
}

# Wikilink pattern used to count citations the model actually emitted in
# the answer text. Same shape as Oracle's verify_citations regex so the
# two stay in lockstep.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


# ── data classes ────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """One golden Q&A item plus optional follow-up turns.

    Follow-ups feed ``session.record_turn`` between asks so the conversation
    rewrite + history block in :class:`~grimore.cognition.oracle.Oracle` are
    actually exercised (and measured) rather than mocked away.
    """
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    follow_ups: list["EvalCase"] = field(default_factory=list)
    # Stratification tag (single-hop, multi-hop, negative, …) so metrics can
    # be broken down by query type. Follow-ups inherit the parent's category
    # unless they set their own.
    category: str = "uncategorized"
    # A negative case has no answer in the vault: ``expected_sources`` is
    # empty and the desired behaviour is for the Oracle to abstain. Scored by
    # abstention, and excluded from the ranking aggregates (where empty
    # expectations would otherwise count as vacuous hits).
    negative: bool = False


@dataclass
class TurnResult:
    """Per-turn metrics. A top-level case has one of these; a case with N
    follow-ups produces 1 + N TurnResults under the same ``case_id``."""
    case_id: str
    turn_index: int                        # 0 = first ask, 1+ = follow-ups
    question: str
    answer: str
    sources: list[str]
    dropped_citations: int
    total_citations: int
    recall_at_k: float
    mrr: float
    # Generation-dependent metrics are ``None`` (not 0.0) in retrieval-only
    # mode, where no answer is produced — averaging zeros would understate them.
    faithfulness: Optional[float]
    keyword_recall: Optional[float]
    answer_relevance: Optional[float]      # None when judge wasn't called / failed
    latency_s: float
    # ── ranking signal (added with the retrieval-rank exposure) ──
    # ``retrieved`` is the rank-ordered, note-deduped list the Oracle actually
    # returned; the hit flags are derived from it. Defaulted so older callers
    # constructing a TurnResult by hand keep working.
    retrieved: list[str] = field(default_factory=list)
    source_hit_at_1: bool = False
    source_hit_at_3: bool = False
    # ── stratification + negative-case signal ──
    category: str = "uncategorized"
    negative: bool = False
    abstained: Optional[bool] = None       # set only for negative cases
    # ── failure taxonomy ──
    # Which pipeline stage broke (or "ok"). Tells you *what to fix*, not just
    # that a number dropped. See ``classify_outcome``. ``expected_sources`` and
    # ``missing_keywords`` are carried so the Failures section can explain each.
    outcome: str = "ok"
    expected_sources: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    # ── per-stage latency (embed/retrieve/rerank/generate/…) ──
    # The Oracle's stage breakdown for this turn, so a latency regression can be
    # pinned to a stage instead of just "slower". Empty when the Oracle didn't
    # report timings (e.g. a hand-built TurnResult).
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """The full run: every turn's metrics + a top-line aggregate.

    Use :meth:`summary` for the JSON shape the CLI exports, and
    :meth:`render` for the Rich console table the user actually reads.
    """
    top_k: int
    turns: list[TurnResult]

    def summary(self) -> dict[str, Any]:
        if not self.turns:
            return {"top_k": self.top_k, "n": 0, "aggregate": {}, "turns": []}
        # Ranking metrics are scored over positive turns only — a negative
        # case has empty expectations, which the metrics treat as a vacuous
        # 1.0, and that would silently inflate Hit@k / MRR.
        pos = [t for t in self.turns if not t.negative]
        neg = [t for t in self.turns if t.negative]
        agg: dict[str, Any] = {
            "hit_at_1":    _mean(1.0 if t.source_hit_at_1 else 0.0 for t in pos),
            "hit_at_3":    _mean(1.0 if t.source_hit_at_3 else 0.0 for t in pos),
            "recall_at_k": _mean(t.recall_at_k for t in pos),
            "mrr":         _mean(t.mrr for t in pos),
            "faithfulness": _present_mean(t.faithfulness for t in self.turns),
            "keyword_recall": _present_mean(t.keyword_recall for t in self.turns),
            "latency_p50": _quantile([t.latency_s for t in self.turns], 0.50),
            "latency_p95": _quantile([t.latency_s for t in self.turns], 0.95),
        }
        # Per-stage latency means (embed/retrieve/rerank/generate/…) so a
        # slowdown can be pinned to a stage, not just the end-to-end number.
        agg["stage_latency"] = _stage_latency_means(self.turns)
        judged = [t.answer_relevance for t in self.turns if t.answer_relevance is not None]
        agg["answer_relevance"] = _mean(judged) if judged else None
        agg["judged_n"] = len(judged)
        # Negative cases: did the Oracle correctly abstain instead of
        # confabulating an answer? Higher is better.
        if neg:
            agg["abstention_rate"] = _mean(1.0 if t.abstained else 0.0 for t in neg)
            agg["negatives_n"] = len(neg)
        # Failure taxonomy: which stage broke, and how often. ``pass_rate`` is
        # the fraction of turns whose outcome is "ok" (excluding the
        # not-assessable ones).
        outcomes: dict[str, int] = {}
        for t in self.turns:
            outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        agg["outcomes"] = outcomes
        scored = [t for t in self.turns if t.outcome != "na"]
        agg["pass_rate"] = _mean(1.0 if t.outcome == "ok" else 0.0 for t in scored)
        return {
            "top_k": self.top_k,
            "n": len(self.turns),
            "positives_n": len(pos),
            "negatives_n": len(neg),
            "aggregate": agg,
            "by_category": _by_category(pos),
            "failures": [_failure_detail(t) for t in self.turns if t.outcome not in ("ok", "na")],
            "turns": [asdict(t) for t in self.turns],
        }

    def render(self, console) -> None:
        """Pretty-print the report. Lazy-imports Rich so eval.py can be
        loaded in headless contexts (CI workers, library use) without
        dragging the whole UI stack into the import graph."""
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        if not self.turns:
            console.print(Panel("No cases were evaluated.", title="Eval", border_style="grimore.warning"))
            return

        table = Table(box=box.SIMPLE, header_style="grimore.muted", pad_edge=False)
        table.add_column("Case", style="grimore.primary", no_wrap=True)
        table.add_column("Turn", justify="right", no_wrap=True)
        table.add_column("Cat", style="grimore.muted", no_wrap=True)
        table.add_column("H@1", justify="center")
        table.add_column("H@3", justify="center")
        table.add_column(f"Recall@{self.top_k}", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Faith.", justify="right")
        table.add_column("KW", justify="right")
        table.add_column("Judge", justify="right")
        table.add_column("Outcome", no_wrap=True)
        table.add_column("Latency", justify="right")
        for t in self.turns:
            neg = t.negative
            table.add_row(
                t.case_id,
                str(t.turn_index),
                (t.category or "")[:10],
                # Ranking columns are meaningless for negatives (empty
                # expectations) — show a dash instead of a vacuous hit.
                "—" if neg else ("✓" if t.source_hit_at_1 else "·"),
                "—" if neg else ("✓" if t.source_hit_at_3 else "·"),
                "—" if neg else f"{t.recall_at_k:.2f}",
                "—" if neg else f"{t.mrr:.2f}",
                "—" if t.faithfulness is None else f"{t.faithfulness:.2f}",
                "—" if t.keyword_recall is None else f"{t.keyword_recall:.2f}",
                "—" if t.answer_relevance is None else f"{t.answer_relevance:.2f}",
                _outcome_markup(t.outcome),
                f"{t.latency_s:.1f}s",
            )
        console.print(table)

        summary = self.summary()
        agg = summary["aggregate"]
        judge = "—" if agg.get("answer_relevance") is None else f"{agg['answer_relevance']:.2f} (n={agg['judged_n']})"
        faith = "—" if agg.get("faithfulness") is None else f"{agg['faithfulness']:.2f}"
        kw = "—" if agg.get("keyword_recall") is None else f"{agg['keyword_recall']:.2f}"
        body = (
            f"Hit@1: [bold]{agg['hit_at_1']:.2f}[/]   "
            f"Hit@3: [bold]{agg['hit_at_3']:.2f}[/]   "
            f"(over {summary['positives_n']} positive turn(s))\n"
            f"Recall@{self.top_k}: [bold]{agg['recall_at_k']:.2f}[/]   "
            f"MRR: [bold]{agg['mrr']:.2f}[/]   "
            f"Faithfulness: [bold]{faith}[/]   "
            f"Keyword recall: [bold]{kw}[/]\n"
            f"Answer relevance (judge): [bold]{judge}[/]\n"
        )
        if summary["negatives_n"]:
            body += (
                f"Abstention (negatives): [bold]{agg['abstention_rate']:.2f}[/] "
                f"(n={agg['negatives_n']})\n"
            )
        # Pass rate + failure-stage distribution (ok hidden — it's the pass).
        dist = "  ".join(
            f"{k} {v}" for k, v in sorted(agg["outcomes"].items()) if k != "ok"
        )
        body += f"Pass rate: [bold]{agg['pass_rate']:.2f}[/]"
        if dist:
            body += f"   ·   {dist}"
        body += (
            f"\nLatency p50/p95: [bold]{agg['latency_p50']:.1f}s[/] / "
            f"[bold]{agg['latency_p95']:.1f}s[/]   ·   n={summary['n']}"
        )
        stages = agg.get("stage_latency") or {}
        if stages:
            parts = "  ".join(
                f"{s.removesuffix('_s')} [bold]{v:.2f}s[/]" for s, v in stages.items()
            )
            body += f"\nStage means: {parts}"
        console.print(Panel(body, title="Aggregate", border_style="grimore.primary"))

        # Per-category ranking breakdown — where retrieval actually struggles.
        by_cat = summary["by_category"]
        if len(by_cat) > 1:
            cat_table = Table(box=box.SIMPLE, header_style="grimore.muted",
                              pad_edge=False, title="By category")
            cat_table.add_column("Category", style="grimore.primary", no_wrap=True)
            cat_table.add_column("n", justify="right")
            cat_table.add_column("Hit@1", justify="right")
            cat_table.add_column("Hit@3", justify="right")
            cat_table.add_column("MRR", justify="right")
            for cat, m in by_cat.items():
                cat_table.add_row(
                    cat, str(m["n"]),
                    f"{m['hit_at_1']:.2f}", f"{m['hit_at_3']:.2f}", f"{m['mrr']:.2f}",
                )
            console.print(cat_table)

        # Failures section — what broke and why, so it's actionable at a glance.
        failures = summary["failures"]
        if failures:
            lines = []
            for f in failures:
                where = f"turn {f['turn_index']}" if f["turn_index"] else "turn 0"
                lines.append(
                    f"[bold]{f['case_id']}[/] ({where}) — {_outcome_markup(f['outcome'])}"
                )
                if f["expected_sources"]:
                    lines.append(
                        f"    expected: {f['expected_sources']}   "
                        f"top-retrieved: {f['retrieved_top'] or '∅'}"
                    )
                if f["missing_keywords"]:
                    lines.append(f"    missing keywords: {f['missing_keywords']}")
            console.print(Panel("\n".join(lines), title=f"Failures ({len(failures)})",
                                border_style="grimore.warning"))


# ── golden loader ───────────────────────────────────────────────────────────


def load_golden(path: Path) -> list[EvalCase]:
    """Parse a golden YAML file into :class:`EvalCase` objects.

    Strict on shape: unknown keys raise ``ValueError`` (typos in the golden
    file should fail loudly, not silently skip questions). Missing
    ``id``/``question`` also raises. Returns an empty list when ``questions``
    is empty but the file is otherwise valid.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    version = raw.get("version")
    if version != _GOLDEN_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema version {version!r} "
            f"(expected {_GOLDEN_SCHEMA_VERSION})"
        )
    questions = raw.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError(f"{path}: 'questions' must be a list")
    return [_parse_case(q, path) for q in questions]


def _parse_case(node: dict, path: Path, *, parent_category: str = "uncategorized") -> EvalCase:
    if not isinstance(node, dict):
        raise ValueError(f"{path}: case is not a mapping: {node!r}")
    unknown = set(node) - _CASE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s) on case {node.get('id', '?')}: {sorted(unknown)}")
    if "id" not in node or "question" not in node:
        raise ValueError(f"{path}: case missing required 'id' or 'question': {node!r}")
    follow_ups_raw = node.get("follow_ups") or []
    if not isinstance(follow_ups_raw, list):
        raise ValueError(f"{path}: 'follow_ups' on {node['id']} must be a list")
    negative = bool(node.get("negative", False))
    expected_sources = list(node.get("expected_sources") or [])
    if negative and expected_sources:
        raise ValueError(
            f"{path}: negative case {node['id']!r} must not list expected_sources "
            f"(its answer is, by definition, not in the vault)"
        )
    # Follow-ups inherit the parent's category unless they declare their own,
    # so a multi-turn case stays grouped under one stratum.
    category = str(node.get("category", parent_category))
    return EvalCase(
        id=str(node["id"]),
        question=str(node["question"]),
        expected_sources=expected_sources,
        expected_keywords=list(node.get("expected_keywords") or []),
        follow_ups=[_parse_case(fu, path, parent_category=category) for fu in follow_ups_raw],
        category=category,
        negative=negative,
    )


# ── metrics ─────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    """NFKD accent-fold to ASCII, then lowercase — structure and spacing kept.

    Shared by title and keyword matching so both are accent-insensitive in the
    same way: golden ``energia`` matches an answer's ``energía`` and vice versa
    (LLM accenting is non-deterministic, so we can't rely on either side).
    """
    return (
        unicodedata.normalize("NFKD", text or "")
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )


def _norm_tokens(title: str) -> frozenset[str]:
    """Normalise a note title to a set of comparison tokens.

    Drops the ``#anchor`` suffix, folds accents to ASCII, lowercases, and
    keeps only alphanumeric runs — so emoji, punctuation and Markdown
    emphasis don't affect matching. This lets a short, stable golden title
    (``"roman empire"``) match the note's decorated H1 (``"🏛️ The Roman
    Empire: An Exhaustive…"``) and a stem title (``"gothic_architecture"``,
    → ``{gothic, architecture}``) alike. It also fixes the prior exact-string
    match, which silently failed whenever a note's stored title wasn't
    byte-identical to the golden entry.
    """
    bare = title.split("#", 1)[0]
    return frozenset(_TOKEN_RE.findall(_fold(bare)))


def _title_matches(expected: str, retrieved_title: str) -> bool:
    """True when ``expected`` identifies ``retrieved_title``.

    Rule: every token of the normalised expected title appears in the
    retrieved title's token set. Authors keep ``expected_sources`` short and
    distinctive; extra decoration on the retrieved side is ignored.
    """
    exp = _norm_tokens(expected)
    return bool(exp) and exp <= _norm_tokens(retrieved_title)


def _any_match(expected: list[str], retrieved_title: str) -> bool:
    return any(_title_matches(e, retrieved_title) for e in expected)


def ranked_sources(result: dict) -> list[str]:
    """Note titles in true retrieval-rank order for a single ``ask`` result.

    Prefers ``result["retrieved"]`` (the rank-ordered, note-deduped list the
    Oracle now exposes) and falls back to ``result["sources"]`` for callers
    that predate it. Titles come back bare; downstream metrics anchor-strip
    again, which is harmless. The fallback is order-destroyed, so rank metrics
    over it are only trustworthy when a single source is expected.
    """
    retrieved = result.get("retrieved")
    if retrieved:
        return [str(r.get("title", "")).strip() for r in retrieved]
    return list(result.get("sources") or [])


def source_hit_at_k(retrieved_sources: list[str], expected: list[str], k: int) -> bool:
    """``True`` when any ``expected`` title appears in the top-``k`` retrieved.

    Matching is token-normalised (see :func:`_title_matches`) and anchor-
    tolerant. Empty ``expected`` is vacuously satisfied, mirroring
    :func:`recall_at_k`."""
    if not expected:
        return True
    return any(_any_match(expected, r) for r in retrieved_sources[: max(k, 0)])


def recall_at_k(retrieved_sources: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0  # vacuously satisfied — nothing was asked for
    hits = sum(
        1 for e in expected
        if any(_title_matches(e, r) for r in retrieved_sources)
    )
    return hits / len(expected)


def mrr(retrieved_sources: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the *first* expected source that appears.

    A single-question MRR is just ``1 / rank``; averaging across questions
    is done by the aggregate. Returns 0 when none of the expected sources
    appear in the retrieved list.
    """
    if not expected:
        return 1.0
    for rank, title in enumerate(retrieved_sources, start=1):
        if _any_match(expected, title):
            return 1.0 / rank
    return 0.0


def faithfulness(answer: str, dropped_citations: int) -> tuple[float, int]:
    """Returns ``(score, total_citations_in_answer)``.

    ``dropped_citations`` is the count Oracle's ``verify_citations`` already
    computed (citations that didn't match any retrieved source); ``total`` is
    every ``[[...]]`` the cleaned answer still carries plus the dropped ones,
    so ``1 - dropped/total`` is well-defined. A citation-free answer scores
    1.0 (nothing to be unfaithful about).
    """
    kept = len(_WIKILINK_RE.findall(answer or ""))
    total = kept + max(dropped_citations, 0)
    if total == 0:
        return 1.0, 0
    return 1.0 - (dropped_citations / total), total


def _keyword_pattern(keyword: str) -> Optional[re.Pattern[str]]:
    """Compile a robust matcher for one expected keyword (``None`` if empty).

    Matches against the accent-folded answer with:

    * a **leading** word boundary, so ``art`` no longer spuriously matches
      ``Sparta``/``start`` — the substring test's biggest false-positive source
      on short keywords;
    * **no trailing** boundary, so simple inflection still counts: ``frog``
      matches ``frogs``, ``cell`` matches ``cells``;
    * **separator-flexible** joins for multi-word keywords, so ``machine
      learning`` also matches ``machine-learning`` and ``machine  learning``.
    """
    tokens = _TOKEN_RE.findall(_fold(keyword))
    if not tokens:
        return None
    body = r"[\W_]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b{body}")


def _keyword_hits(
    answer: str, expected_keywords: list[str]
) -> tuple[list[str], list[str]]:
    """Partition expected keywords into ``(present, missing)`` for one answer.

    Single source of truth for :func:`keyword_recall` and the Failures
    section's ``missing_keywords`` — so reported recall and listed misses can
    never disagree.
    """
    folded = _fold(answer)
    present: list[str] = []
    missing: list[str] = []
    for kw in expected_keywords:
        pat = _keyword_pattern(kw)
        if pat is not None and pat.search(folded):
            present.append(kw)
        else:
            missing.append(kw)
    return present, missing


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of ``expected_keywords`` that survive into the answer.

    Empty expectation → 1.0 (nothing to satisfy). Matching is accent-folded and
    word-boundary anchored — see :func:`_keyword_pattern`.
    """
    if not expected_keywords:
        return 1.0
    present, _ = _keyword_hits(answer, expected_keywords)
    return len(present) / len(expected_keywords)


# Phrasings that indicate the Oracle declined to answer. The first two are
# Oracle's own sentinels; the rest are common refusal cues. Substring-matched
# case-insensitively against the answer.
_ABSTENTION_CUES = (
    "seems empty", "oracle is silent", "no relevant", "couldn't find",
    "could not find", "do not have", "don't have", "not in the vault",
    "no information", "i don't know", "i do not know", "cannot answer",
    "can't answer", "unable to", "nothing in the vault", "no mention",
    "isn't anything", "is not anything",
)


def answer_abstained(answer: str) -> bool:
    """Best-effort: did the Oracle decline rather than answer?

    Negative cases (no answer in the vault) want this ``True``. Deterministic
    by design — it matches the empty answer, the Oracle's empty-vault
    sentinels, and common refusal phrasings — so it runs in CI without an LLM
    judge. It will miss a confidently-worded hallucination, so pair it with
    ``--judge`` when you want a stricter false-answer signal.
    """
    text = (answer or "").strip().lower()
    if not text:
        return True
    return any(cue in text for cue in _ABSTENTION_CUES)


# ── failure taxonomy ─────────────────────────────────────────────────────────

# A turn counts as a generation miss when fewer than this fraction of its
# expected keywords survive into the answer (despite the right context being
# retrieved). Soft by design — a single missing keyword isn't a failure.
_KW_PASS_THRESHOLD = 0.5

# Outcome labels, in pipeline order. The first stage that breaks wins, so the
# label points at the component to fix.
OUTCOMES = (
    "ok",               # everything the turn could check, passed
    "retrieval_miss",   # expected note never retrieved (not in the pool at all)
    "ranking_miss",     # in the pool, but ranked below the context cutoff (top_k)
    "generation_miss",  # right context, but the answer missed the expected content
    "citation_miss",    # right answer, but citations were ungrounded/dropped
    "hallucinated",     # negative case: answered instead of abstaining
    "na",               # not assessable (e.g. retrieval-only negative)
)


def classify_outcome(
    *,
    negative: bool,
    generate: bool,
    ranked: list[str],
    expected_sources: list[str],
    top_k: int,
    keyword_recall: Optional[float],
    faithfulness: Optional[float],
    abstained: Optional[bool],
    kw_threshold: float = _KW_PASS_THRESHOLD,
) -> str:
    """Classify a turn into the first pipeline stage that broke.

    The order matters — retrieval precedes ranking precedes generation
    precedes citation — so the label names the earliest (root-cause) failure
    rather than a downstream symptom. Negative cases are judged on abstention.
    """
    if negative:
        if abstained is None:
            return "na"                       # retrieval-only: no answer to judge
        return "ok" if abstained else "hallucinated"
    if not expected_sources:
        return "ok"                           # nothing expected → vacuous pass
    if not source_hit_at_k(ranked, expected_sources, len(ranked)):
        return "retrieval_miss"               # never surfaced at any depth
    if not source_hit_at_k(ranked, expected_sources, top_k):
        return "ranking_miss"                 # retrieved, but below the context cut
    if not generate:
        return "ok"                           # retrieval succeeded; generation unseen
    if keyword_recall is not None and keyword_recall < kw_threshold:
        return "generation_miss"              # had the context, missed the content
    if faithfulness is not None and faithfulness < 1.0:
        return "citation_miss"                # content ok, citations ungrounded
    return "ok"


# Short label + colour per outcome, for the Rich table / Failures panel.
_OUTCOME_STYLE: dict[str, tuple[str, str]] = {
    "ok": ("ok", "green"),
    "retrieval_miss": ("retrieval", "red"),
    "ranking_miss": ("ranking", "yellow"),
    "generation_miss": ("generation", "yellow"),
    "citation_miss": ("citation", "yellow"),
    "hallucinated": ("halluc", "red"),
    "na": ("n/a", "grimore.muted"),
}


def _outcome_markup(outcome: str) -> str:
    label, style = _OUTCOME_STYLE.get(outcome, (outcome, "white"))
    return f"[{style}]{label}[/]"


# ── LLM-as-judge ────────────────────────────────────────────────────────────


_JUDGE_SYSTEM = (
    "You rate how well an answer addresses the user's question, on a 0-10 "
    "scale. 0 means completely off-topic or empty; 10 means a thorough, "
    "directly relevant answer. Be strict but fair."
)


def judge_relevance(router, question: str, answer: str) -> Optional[float]:
    """LLM-as-judge for one (question, answer) pair, normalised to ``[0, 1]``.

    Returns ``None`` on any failure — router unreachable, circuit open,
    JSON unparseable, score missing or non-numeric — so the harness can
    average over the cases that *did* score without being skewed by a
    half-broken Ollama. Never raises.
    """
    if not (question and answer):
        return None
    prompt = (
        f"Question:\n{question}\n\nAnswer:\n{answer[:4000]}\n\n"
        'Return ONLY JSON: {"score": <0-10>, "reason": "..."}'
    )
    try:
        resp = router.complete(
            prompt=prompt, system_prompt=_JUDGE_SYSTEM, json_format=True
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("eval_judge_failed", error=str(e))
        return None
    if not isinstance(resp, dict):
        return None
    try:
        score = float(resp.get("score"))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score / 10.0))


# ── runner ──────────────────────────────────────────────────────────────────


def run_eval(session, cases: list[EvalCase], *, top_k: int = 5,
             retrieval_k: int = 10, generate: bool = True,
             judge: bool = True) -> EvalReport:
    """Drive one Oracle call per turn, walking ``follow_ups`` with the
    session's conversation memory in between.

    The session is reset (``forget()``) at the start of every top-level case
    so cross-case bleed-through doesn't corrupt the rewrite-with-history
    signal. Within a case, follow-ups *do* see prior turns — that's the
    whole point of measuring conversation quality.

    ``top_k`` sizes the generation context; ``retrieval_k`` (≥ ``top_k``) is
    the ranking-pool depth scored by Hit@k / MRR, so MRR@10 is real even with
    a top-5 answer. ``generate=False`` is retrieval-only mode: it calls
    ``oracle.retrieve`` instead of ``oracle.ask``, skipping the answer LLM —
    fast, deterministic ranking data with the generation metrics reported as
    ``None``.
    """
    turns: list[TurnResult] = []
    for case in cases:
        # Reset conversation state between top-level cases; follow-ups in the
        # SAME case must see history, which is what record_turn below builds.
        if hasattr(session, "forget"):
            session.forget()
        _eval_case(session, case, turn_index=0, top_k=top_k,
                   retrieval_k=retrieval_k, generate=generate, judge=judge, out=turns)
    return EvalReport(top_k=top_k, turns=turns)


def _eval_case(session, case: EvalCase, *, turn_index: int, top_k: int,
               retrieval_k: int, generate: bool, judge: bool,
               out: list[TurnResult]) -> None:
    history = list(session.turns) if getattr(session, "turns", None) else None
    hist_kw = {"history": history} if history else {}
    started = time.monotonic()
    if generate:
        try:
            result = session.oracle.ask(
                case.question, top_k=top_k, retrieval_k=retrieval_k, **hist_kw,
            )
        except Exception as e:
            logger.warning("eval_ask_failed", case=case.id, error=str(e))
            result = {"answer": "", "sources": [], "retrieved": [], "dropped_citations": 0}
    else:
        # Retrieval-only: rank sources without paying for an answer. Pass a
        # timings dict so the stage breakdown survives even here (no generate).
        stage_timings: dict[str, float] = {}
        try:
            retrieved = session.oracle.retrieve(
                case.question, top_k=retrieval_k, timings=stage_timings, **hist_kw,
            )
        except Exception as e:
            logger.warning("eval_retrieve_failed", case=case.id, error=str(e))
            retrieved = []
        result = {"answer": "", "sources": [], "retrieved": retrieved,
                  "dropped_citations": 0, "timings": stage_timings}
    latency = time.monotonic() - started
    # Oracle-reported per-stage latency; may be absent on older/fake sessions.
    timings = {k: float(v) for k, v in (result.get("timings") or {}).items()}

    answer = result.get("answer", "") or ""
    sources = result.get("sources", []) or []
    # Rank-ordered retrieval list — the honest input for Hit@k / MRR. The
    # citation-facing ``sources`` stays separate and feeds faithfulness only.
    ranked = ranked_sources(result)
    dropped = int(result.get("dropped_citations", 0) or 0)

    if generate:
        faith, total_cit = faithfulness(answer, dropped)
        # Present/missing come from one matcher so recall and the Failures
        # section's ``missing_keywords`` can't disagree.
        present_kw, missing_kw = _keyword_hits(answer, case.expected_keywords)
        # Keyword recall is N/A (not 1.0) when a case lists no keywords — a
        # vacuous 1.0 would inflate the aggregate.
        kw: Optional[float] = (
            len(present_kw) / len(case.expected_keywords)
            if case.expected_keywords else None
        )
        rel = judge_relevance(session.router, case.question, answer) if judge else None
    else:
        # No answer → generation metrics are not applicable (not zero).
        faith, total_cit, kw, rel = None, 0, None, None
        missing_kw = []

    # Negative cases score on whether the Oracle abstained (only meaningful
    # when an answer was actually generated).
    abstained = answer_abstained(answer) if (case.negative and generate) else None

    outcome = classify_outcome(
        negative=case.negative, generate=generate, ranked=ranked,
        expected_sources=case.expected_sources, top_k=top_k,
        keyword_recall=kw, faithfulness=faith, abstained=abstained,
    )

    out.append(TurnResult(
        case_id=case.id,
        turn_index=turn_index,
        question=case.question,
        answer=answer,
        sources=list(sources),
        retrieved=list(ranked),
        dropped_citations=dropped,
        total_citations=total_cit,
        recall_at_k=recall_at_k(ranked, case.expected_sources),
        source_hit_at_1=source_hit_at_k(ranked, case.expected_sources, 1),
        source_hit_at_3=source_hit_at_k(ranked, case.expected_sources, 3),
        mrr=mrr(ranked, case.expected_sources),
        faithfulness=faith,
        keyword_recall=kw,
        answer_relevance=rel,
        latency_s=latency,
        category=case.category,
        negative=case.negative,
        abstained=abstained,
        outcome=outcome,
        expected_sources=list(case.expected_sources),
        missing_keywords=missing_kw,
        timings=timings,
    ))

    # Feed this turn into conversation memory before walking follow-ups.
    if hasattr(session, "record_turn"):
        session.record_turn(case.question, answer, sources)
    for i, fu in enumerate(case.follow_ups, start=turn_index + 1):
        _eval_case(session, fu, turn_index=i, top_k=top_k, retrieval_k=retrieval_k,
                   generate=generate, judge=judge, out=out)


# ── baseline comparison (RRF vs dense-only) ──────────────────────────────────


@contextlib.contextmanager
def _override_hybrid(session, value: bool):
    """Temporarily force ``config.cognition.hybrid_search`` and restore it.

    The Oracle reads this flag live on every retrieval, and ``Session`` hands
    the Oracle the *same* config object, so flipping it here switches the
    retrieval path for the duration of the block.
    """
    cfg = session.config.cognition
    sentinel = object()
    old = getattr(cfg, "hybrid_search", sentinel)
    cfg.hybrid_search = value
    try:
        yield
    finally:
        if old is sentinel:
            with contextlib.suppress(AttributeError):
                delattr(cfg, "hybrid_search")
        else:
            cfg.hybrid_search = old


# Metrics surfaced in the baseline delta, in display order. Ranking first
# (that's what the retrieval mode actually moves); generation/abstention
# follow when present.
_COMPARE_METRICS: tuple[tuple[str, str], ...] = (
    ("hit_at_1", "Hit@1"),
    ("hit_at_3", "Hit@3"),
    ("mrr", "MRR"),
    ("recall_at_k", "Recall@k"),
    ("keyword_recall", "Keyword recall"),
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Judge relevance"),
    ("abstention_rate", "Abstention"),
)


def run_baseline(session, cases: list[EvalCase], *, top_k: int = 5,
                 retrieval_k: int = 10, generate: bool = True,
                 judge: bool = True) -> tuple[EvalReport, EvalReport]:
    """Run the suite twice to quantify what hybrid fusion buys.

    Arm A forces ``hybrid_search=True`` (RRF over dense + BM25); arm B forces
    it off (``find_similar_notes`` → dense/vector-only). Note this is *not*
    BM25-only: in this codebase the ``hybrid_search`` toggle selects fusion vs
    pure vector search, so the delta measures the uplift fusion adds over dense
    retrieval. Returns ``(hybrid_report, baseline_report)``.
    """
    with _override_hybrid(session, True):
        hybrid = run_eval(session, cases, top_k=top_k, retrieval_k=retrieval_k,
                          generate=generate, judge=judge)
    with _override_hybrid(session, False):
        baseline = run_eval(session, cases, top_k=top_k, retrieval_k=retrieval_k,
                            generate=generate, judge=judge)
    return hybrid, baseline


def comparison_summary(hybrid: EvalReport, baseline: EvalReport) -> dict[str, Any]:
    """Per-metric ``{hybrid, baseline, delta}`` (delta = hybrid − baseline,
    positive = RRF uplift). Metrics absent from both arms are skipped."""
    h = hybrid.summary()["aggregate"]
    b = baseline.summary()["aggregate"]
    metrics: dict[str, Any] = {}
    for key, _ in _COMPARE_METRICS:
        hv, bv = h.get(key), b.get(key)
        if hv is None and bv is None:
            continue
        delta = (hv - bv) if isinstance(hv, (int, float)) and isinstance(bv, (int, float)) else None
        metrics[key] = {"hybrid": hv, "baseline": bv, "delta": delta}
    return {
        "arms": {"hybrid": "RRF (dense + BM25 fusion)", "baseline": "dense-only (no fusion)"},
        "metrics": metrics,
    }


def render_comparison(hybrid: EvalReport, baseline: EvalReport, console) -> None:
    """Side-by-side RRF-vs-dense delta table."""
    from rich.table import Table
    from rich import box

    cmp = comparison_summary(hybrid, baseline)
    labels = dict(_COMPARE_METRICS)
    table = Table(box=box.SIMPLE, header_style="grimore.muted", pad_edge=False,
                  title="Baseline — hybrid (RRF) vs dense-only")
    table.add_column("Metric", style="grimore.primary", no_wrap=True)
    table.add_column("Hybrid (RRF)", justify="right")
    table.add_column("Dense-only", justify="right")
    table.add_column("Δ", justify="right")

    def _fmt(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.2f}"

    for key, m in cmp["metrics"].items():
        d = m["delta"]
        if d is None:
            dstr = "—"
        elif d > 1e-9:
            dstr = f"[green]+{d:.2f}[/]"
        elif d < -1e-9:
            dstr = f"[red]{d:.2f}[/]"
        else:
            dstr = "0.00"
        table.add_row(labels.get(key, key), _fmt(m["hybrid"]), _fmt(m["baseline"]), dstr)
    console.print(table)


def export_comparison(comparison: dict, path: Path) -> None:
    """Dump the combined baseline result (comparison + both arms) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")


# ── provenance, history ledger & regression comparison ───────────────────────

# Config fields that actually move retrieval/answer quality. Two runs with the
# same fingerprint are comparable; a different fingerprint explains a metric
# shift without it being a code regression.
_FINGERPRINT_KEYS = (
    "model_embeddings_local", "model_llm_local", "hybrid_search", "rrf_k",
    "rerank", "rerank_engine", "rerank_model", "rerank_pool", "vector_backend",
)

# Aggregate metrics compared run-over-run (ranking + generation + pass rate).
_RUN_COMPARE_METRICS: tuple[tuple[str, str], ...] = _COMPARE_METRICS + (
    ("pass_rate", "Pass rate"),
)


def _short_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _git_provenance() -> dict[str, Any]:
    """Best-effort current commit + dirty flag; ``None`` outside a git repo."""
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""
    sha = _git("rev-parse", "--short", "HEAD") or None
    dirty = bool(_git("status", "--porcelain")) if sha else None
    return {"git_sha": sha, "git_dirty": dirty}


def run_provenance(session, golden_path: Path) -> dict[str, Any]:
    """Identity of a run — commit, working-tree cleanliness, a fingerprint of
    the retrieval-relevant config, and a hash of the golden file. Lets the
    history ledger separate genuine regressions from "different inputs"."""
    fingerprint = {
        k: getattr(getattr(session.config, "cognition", None), k, None)
        for k in _FINGERPRINT_KEYS
    }
    golden_path = Path(golden_path)
    prov: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden": str(golden_path),
        "golden_hash": (
            _short_hash(golden_path.read_text(encoding="utf-8"))
            if golden_path.exists() else None
        ),
        "config_hash": _short_hash(fingerprint),
        "config": fingerprint,
    }
    prov.update(_git_provenance())
    return prov


_LEDGER_METRICS = (
    "hit_at_1", "hit_at_3", "mrr", "recall_at_k", "keyword_recall",
    "faithfulness", "answer_relevance", "abstention_rate", "pass_rate",
)


def append_history(path: Path, summary: dict, provenance: dict, *,
                   run_meta: dict) -> None:
    """Append one compact run record to a JSONL ledger for trend tracking.

    One line per run keyed by commit + config fingerprint, so you can plot
    metric-over-time and tell whether a drop coincides with a code change or a
    config/golden change."""
    agg = summary.get("aggregate", {})
    row = {
        **provenance,
        **run_meta,
        "n": summary.get("n"),
        "positives_n": summary.get("positives_n"),
        "negatives_n": summary.get("negatives_n"),
        "metrics": {k: agg.get(k) for k in _LEDGER_METRICS},
        "outcomes": agg.get("outcomes", {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_run(path: Path) -> dict:
    """Load a previously ``--export``-ed run summary for ``--compare``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _turn_key(turn: dict) -> str:
    return f"{turn.get('case_id')}#{turn.get('turn_index')}"


def _num(v: Any) -> float:
    return v if isinstance(v, (int, float)) else 0.0


def _turn_status(prev: dict, cur: dict) -> str:
    """Classify a turn's movement between two runs.

    Outcome is the primary signal (``ok`` ↔ not-``ok`` is the headline
    regression/improvement); ranking deltas break ties when both runs failed.
    """
    eps = 1e-9
    p_ok, c_ok = prev.get("outcome") == "ok", cur.get("outcome") == "ok"
    hit_lost = bool(prev.get("source_hit_at_1")) and not bool(cur.get("source_hit_at_1"))
    hit_gain = not bool(prev.get("source_hit_at_1")) and bool(cur.get("source_hit_at_1"))
    mrr_drop = _num(cur.get("mrr")) < _num(prev.get("mrr")) - eps
    mrr_rise = _num(cur.get("mrr")) > _num(prev.get("mrr")) + eps
    regressed = (p_ok and not c_ok) or hit_lost or (not c_ok and not p_ok and mrr_drop)
    improved = (not p_ok and c_ok) or hit_gain or (not c_ok and not p_ok and mrr_rise)
    if regressed and not improved:
        return "regressed"
    if improved and not regressed:
        return "improved"
    if prev.get("outcome") != cur.get("outcome"):
        return "changed"
    return "same"


def compare_runs(current: dict, previous: dict) -> dict[str, Any]:
    """Diff two run summaries per turn + per aggregate, flagging regressions.

    Both args are the dicts produced by :meth:`EvalReport.summary` (what
    ``--export`` writes). Turns are matched on ``case_id#turn_index``; an id
    that exists in only one run is reported as ``new``/``dropped`` so renamed
    or added questions don't masquerade as regressions.
    """
    cur_turns = {_turn_key(t): t for t in current.get("turns", [])}
    prev_turns = {_turn_key(t): t for t in previous.get("turns", [])}

    rows: list[dict[str, Any]] = []
    for key, ct in cur_turns.items():
        pt = prev_turns.get(key)
        if pt is None:
            rows.append({
                "case_id": ct.get("case_id"), "turn_index": ct.get("turn_index"),
                "status": "new", "prev_outcome": None, "cur_outcome": ct.get("outcome"),
                "mrr_delta": None,
            })
            continue
        rows.append({
            "case_id": ct.get("case_id"), "turn_index": ct.get("turn_index"),
            "status": _turn_status(pt, ct),
            "prev_outcome": pt.get("outcome"), "cur_outcome": ct.get("outcome"),
            "mrr_delta": round(_num(ct.get("mrr")) - _num(pt.get("mrr")), 4),
        })
    dropped = [k for k in prev_turns if k not in cur_turns]

    cur_agg, prev_agg = current.get("aggregate", {}), previous.get("aggregate", {})
    agg_delta: dict[str, Any] = {}
    for key, _ in _RUN_COMPARE_METRICS:
        cv, pv = cur_agg.get(key), prev_agg.get(key)
        if cv is None and pv is None:
            continue
        delta = (cv - pv) if isinstance(cv, (int, float)) and isinstance(pv, (int, float)) else None
        agg_delta[key] = {"current": cv, "previous": pv, "delta": delta}

    regressions = [r for r in rows if r["status"] == "regressed"]
    improvements = [r for r in rows if r["status"] == "improved"]
    return {
        "aggregate_delta": agg_delta,
        "turns": rows,
        "regressions": regressions,
        "improvements": improvements,
        "dropped": dropped,
        "n_regressions": len(regressions),
        "n_improvements": len(improvements),
    }


def render_regression(comparison: dict, console) -> None:
    """Aggregate deltas + a Regressions panel for a run-over-run ``--compare``."""
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    labels = dict(_RUN_COMPARE_METRICS)
    table = Table(box=box.SIMPLE, header_style="grimore.muted", pad_edge=False,
                  title="Compare — current vs previous run")
    table.add_column("Metric", style="grimore.primary", no_wrap=True)
    table.add_column("Current", justify="right")
    table.add_column("Previous", justify="right")
    table.add_column("Δ", justify="right")

    def _fmt(v: Any) -> str:
        return "—" if not isinstance(v, (int, float)) else f"{v:.2f}"

    for key, m in comparison["aggregate_delta"].items():
        d = m["delta"]
        if d is None:
            dstr = "—"
        elif d > 1e-9:
            dstr = f"[green]+{d:.2f}[/]"
        elif d < -1e-9:
            dstr = f"[red]{d:.2f}[/]"
        else:
            dstr = "0.00"
        table.add_row(labels.get(key, key), _fmt(m["current"]), _fmt(m["previous"]), dstr)
    console.print(table)

    regs, imps = comparison["regressions"], comparison["improvements"]
    if regs:
        lines = [
            f"[bold]{r['case_id']}[/] (turn {r['turn_index']}): "
            f"{r['prev_outcome']} → {r['cur_outcome']}"
            for r in regs
        ]
        console.print(Panel("\n".join(lines), title=f"Regressions ({len(regs)})",
                            border_style="red"))
    summary_line = f"{len(regs)} regression(s) · {len(imps)} improvement(s)"
    if comparison["dropped"]:
        summary_line += f" · {len(comparison['dropped'])} turn(s) dropped"
    console.print(summary_line)


def export_report(report: EvalReport, path: Path) -> None:
    """Dump the full report as pretty JSON. Parent dirs are created lazily so
    callers can pass ``Path('reports/eval-2026-05-28.json')`` without ceremony."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.summary(), indent=2, ensure_ascii=False), encoding="utf-8")


# ── helpers ─────────────────────────────────────────────────────────────────


def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return float(statistics.fmean(vs)) if vs else 0.0


def _present_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over the non-``None`` values, or ``None`` when all are absent.

    Used for generation-dependent metrics that don't apply in retrieval-only
    mode — so the aggregate reads ``—`` instead of a misleading 0.0."""
    vs = [v for v in values if v is not None]
    return float(statistics.fmean(vs)) if vs else None


# Pipeline stages in execution order, for the latency breakdown. Reporting
# order only — a stage absent from every turn's timings is dropped.
_STAGE_ORDER = ("rewrite_s", "embed_s", "retrieve_s", "rerank_s", "generate_s")


def _stage_latency_means(turns: list["TurnResult"]) -> dict[str, float]:
    """Mean seconds per pipeline stage across turns that reported it.

    Stages are averaged independently — a retrieval-only turn simply doesn't
    contribute to ``generate_s`` — so mixed and single-mode runs both yield a
    meaningful breakdown. Only stages observed at least once appear.
    """
    means: dict[str, float] = {}
    for stage in _STAGE_ORDER:
        vals = [t.timings[stage] for t in turns if stage in t.timings]
        if vals:
            means[stage] = _mean(vals)
    return means


def _failure_detail(t: "TurnResult") -> dict[str, Any]:
    """Compact, actionable record of one failing turn for the Failures section."""
    return {
        "case_id": t.case_id,
        "turn_index": t.turn_index,
        "category": t.category,
        "outcome": t.outcome,
        "question": t.question,
        "expected_sources": t.expected_sources,
        "retrieved_top": t.retrieved[:5],
        "missing_keywords": t.missing_keywords,
    }


def _by_category(turns: list["TurnResult"]) -> dict[str, dict[str, Any]]:
    """Ranking metrics grouped by ``category`` over the given (positive)
    turns — so the report shows *which* query types retrieval handles well.
    """
    groups: dict[str, list["TurnResult"]] = {}
    for t in turns:
        groups.setdefault(t.category, []).append(t)
    return {
        cat: {
            "n": len(ts),
            "hit_at_1": _mean(1.0 if t.source_hit_at_1 else 0.0 for t in ts),
            "hit_at_3": _mean(1.0 if t.source_hit_at_3 else 0.0 for t in ts),
            "mrr": _mean(t.mrr for t in ts),
        }
        for cat, ts in sorted(groups.items())
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    # Small-N safe percentile: sort + nearest-rank instead of statistics.quantiles
    # so a 1- or 2-element list (common when iterating on the golden) is fine.
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[idx])
