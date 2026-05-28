"""RAG evaluation harness.

Measures retrieval and answer quality against a checked-in YAML golden set
so future retrieval-quality changes (rerankers, chunkers, backend swaps)
can be evaluated against a stable baseline instead of vibes.

Metrics computed per case (and aggregated over the suite):

* ``recall_at_k``     — fraction of ``expected_sources`` present in the top-k
                        retrieved sources for the ask.
* ``mrr``             — mean reciprocal rank of the first ``expected_sources``
                        hit (0 when none of them appear).
* ``faithfulness``    — ``1 - dropped_citations / total_citations``; 1.0 means
                        every ``[[wikilink]]`` the model emitted was grounded
                        in a retrieved source.
* ``keyword_recall``  — fraction of ``expected_keywords`` substring-present
                        in the answer (case-insensitive).
* ``answer_relevance``— LLM-as-judge: the local model rates the answer 0-10
                        against the question and we normalise to ``[0, 1]``.
                        Skipped per-case when the router returns None;
                        excluded from the aggregate when no case scored.
* ``latency_s``       — wall-clock for the ``ask`` call.

The harness never raises on a model failure: a bad ask shows up as zero
recall and a None judge score, not a crashed run. CI should never gate on
the judge metric — it's directional, not absolute.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
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
_CASE_KEYS = {"id", "question", "expected_sources", "expected_keywords", "follow_ups"}

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
    faithfulness: float
    keyword_recall: float
    answer_relevance: Optional[float]      # None when judge wasn't called / failed
    latency_s: float


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
        agg: dict[str, Any] = {
            "recall_at_k": _mean(t.recall_at_k for t in self.turns),
            "mrr":         _mean(t.mrr for t in self.turns),
            "faithfulness":_mean(t.faithfulness for t in self.turns),
            "keyword_recall": _mean(t.keyword_recall for t in self.turns),
            "latency_p50": _quantile([t.latency_s for t in self.turns], 0.50),
            "latency_p95": _quantile([t.latency_s for t in self.turns], 0.95),
        }
        judged = [t.answer_relevance for t in self.turns if t.answer_relevance is not None]
        agg["answer_relevance"] = _mean(judged) if judged else None
        agg["judged_n"] = len(judged)
        return {
            "top_k": self.top_k,
            "n": len(self.turns),
            "aggregate": agg,
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
        table.add_column(f"Recall@{self.top_k}", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Faith.", justify="right")
        table.add_column("KW", justify="right")
        table.add_column("Judge", justify="right")
        table.add_column("Latency", justify="right")
        for t in self.turns:
            table.add_row(
                t.case_id,
                str(t.turn_index),
                f"{t.recall_at_k:.2f}",
                f"{t.mrr:.2f}",
                f"{t.faithfulness:.2f}",
                f"{t.keyword_recall:.2f}",
                "—" if t.answer_relevance is None else f"{t.answer_relevance:.2f}",
                f"{t.latency_s:.1f}s",
            )
        console.print(table)

        agg = self.summary()["aggregate"]
        judge = "—" if agg.get("answer_relevance") is None else f"{agg['answer_relevance']:.2f} (n={agg['judged_n']})"
        body = (
            f"Recall@{self.top_k}: [bold]{agg['recall_at_k']:.2f}[/]   "
            f"MRR: [bold]{agg['mrr']:.2f}[/]   "
            f"Faithfulness: [bold]{agg['faithfulness']:.2f}[/]   "
            f"Keyword recall: [bold]{agg['keyword_recall']:.2f}[/]\n"
            f"Answer relevance (judge): [bold]{judge}[/]\n"
            f"Latency p50/p95: [bold]{agg['latency_p50']:.1f}s[/] / "
            f"[bold]{agg['latency_p95']:.1f}s[/]   ·   n={self.summary()['n']}"
        )
        console.print(Panel(body, title="Aggregate", border_style="grimore.primary"))


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


def _parse_case(node: dict, path: Path) -> EvalCase:
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
    return EvalCase(
        id=str(node["id"]),
        question=str(node["question"]),
        expected_sources=list(node.get("expected_sources") or []),
        expected_keywords=list(node.get("expected_keywords") or []),
        follow_ups=[_parse_case(fu, path) for fu in follow_ups_raw],
    )


# ── metrics ─────────────────────────────────────────────────────────────────


def _bare_titles(sources: Iterable[str]) -> list[str]:
    """Strip the ``#anchor`` suffix so a `[[Title#p.4]]` matches a golden
    ``Title`` entry. Order preserved for MRR."""
    return [s.split("#", 1)[0].strip() for s in sources or []]


def recall_at_k(retrieved_sources: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0  # vacuously satisfied — nothing was asked for
    got = set(_bare_titles(retrieved_sources))
    hits = sum(1 for e in expected if e in got)
    return hits / len(expected)


def mrr(retrieved_sources: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the *first* expected source that appears.

    A single-question MRR is just ``1 / rank``; averaging across questions
    is done by the aggregate. Returns 0 when none of the expected sources
    appear in the retrieved list.
    """
    if not expected:
        return 1.0
    bare = _bare_titles(retrieved_sources)
    expected_set = set(expected)
    for rank, title in enumerate(bare, start=1):
        if title in expected_set:
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


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    text = (answer or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)


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


def run_eval(session, cases: list[EvalCase], *, top_k: int = 5, judge: bool = True) -> EvalReport:
    """Drive one Oracle ``ask`` per turn, walking ``follow_ups`` with the
    session's conversation memory in between.

    The session is reset (``forget()``) at the start of every top-level case
    so cross-case bleed-through doesn't corrupt the rewrite-with-history
    signal. Within a case, follow-ups *do* see prior turns — that's the
    whole point of measuring conversation quality.
    """
    turns: list[TurnResult] = []
    for case in cases:
        # Reset conversation state between top-level cases; follow-ups in the
        # SAME case must see history, which is what record_turn below builds.
        if hasattr(session, "forget"):
            session.forget()
        _eval_case(session, case, turn_index=0, top_k=top_k, judge=judge, out=turns)
    return EvalReport(top_k=top_k, turns=turns)


def _eval_case(session, case: EvalCase, *, turn_index: int, top_k: int,
               judge: bool, out: list[TurnResult]) -> None:
    history = list(session.turns) if getattr(session, "turns", None) else None
    started = time.monotonic()
    try:
        result = session.oracle.ask(
            case.question, top_k=top_k,
            **({"history": history} if history else {}),
        )
    except Exception as e:
        logger.warning("eval_ask_failed", case=case.id, error=str(e))
        result = {"answer": "", "sources": [], "dropped_citations": 0}
    latency = time.monotonic() - started

    answer = result.get("answer", "") or ""
    sources = result.get("sources", []) or []
    dropped = int(result.get("dropped_citations", 0) or 0)

    faith, total_cit = faithfulness(answer, dropped)
    rel = judge_relevance(session.router, case.question, answer) if judge else None

    out.append(TurnResult(
        case_id=case.id,
        turn_index=turn_index,
        question=case.question,
        answer=answer,
        sources=list(sources),
        dropped_citations=dropped,
        total_citations=total_cit,
        recall_at_k=recall_at_k(sources, case.expected_sources),
        mrr=mrr(sources, case.expected_sources),
        faithfulness=faith,
        keyword_recall=keyword_recall(answer, case.expected_keywords),
        answer_relevance=rel,
        latency_s=latency,
    ))

    # Feed this turn into conversation memory before walking follow-ups.
    if hasattr(session, "record_turn"):
        session.record_turn(case.question, answer, sources)
    for i, fu in enumerate(case.follow_ups, start=turn_index + 1):
        _eval_case(session, fu, turn_index=i, top_k=top_k, judge=judge, out=out)


def export_report(report: EvalReport, path: Path) -> None:
    """Dump the full report as pretty JSON. Parent dirs are created lazily so
    callers can pass ``Path('reports/eval-2026-05-28.json')`` without ceremony."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.summary(), indent=2, ensure_ascii=False), encoding="utf-8")


# ── helpers ─────────────────────────────────────────────────────────────────


def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return float(statistics.fmean(vs)) if vs else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    # Small-N safe percentile: sort + nearest-rank instead of statistics.quantiles
    # so a 1- or 2-element list (common when iterating on the golden) is fine.
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[idx])
