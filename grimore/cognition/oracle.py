"""
The Oracle: Retrieval-Augmented Generation (RAG) Engine.
This module combines semantic search (via the Connector) with LLM completion
(via the LLMRouter) to answer questions based on the vault's content.
"""
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterator

from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.embedder import Embedder
from grimore.cognition.connector import Connector
from grimore.memory.db import Database
from grimore.utils.config import CognitionConfig
from grimore.utils.logger import get_logger
from grimore.utils.hashing import sha256_file
from grimore.utils.security import SecurityGuard
from grimore.cognition.sufficiency import assess, blocked_answer
from grimore.cognition.answer_protocol import PROTOCOL_VERSION, render_claims, response_snapshot

logger = get_logger(__name__)

# Hard cap on the total context (\n\n-joined sources) injected into the LLM
# system prompt. With top_k×500 chars per chunk, a high --top-k could push
# >25 KB into a 32 k-token window — fine on qwen2.5:3b, fragile on smaller
# local models. 16 KB leaves headroom for the question, the prompt template,
# and the model's own answer space.
_ORACLE_CONTEXT_MAX_CHARS = 16_000
_CONTEXT_SEPARATOR = "\n\n"

# Default budget for the follow-up rewrite. It is optional work that blocks
# retrieval with nothing on screen, so it gets its own deadline rather than
# the generation-sized request timeout (600 s is a plausible setting there).
#
# Sized from measurement, not taste: on the slowest configuration to hand
# (ministral-3:8b, CPU-only) rewrites ran 14.3-42.7 s, median 19.3 s. A 20 s
# budget killed 43% of them; 45 s killed none. 60 s keeps headroom above that
# worst case while still bounding the wait an order of magnitude below the
# request timeout. On hardware where a rewrite takes a second, it never fires.
_REWRITE_TIMEOUT_S = 60

# Matches a single ``[[wikilink]]`` citation; group 1 is the inner label.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
# Generation-only references, never document ids or physical page numbers.
_CITATION_REF_RE = re.compile(r"(?<!\[)\[(cite:[^\[\]\r\n]{0,32})\](?!\])")
_EMITTED_CITATION_RE = re.compile(
    r"\[\[([^\[\]]+)\]\]|(?<!\[)\[(cite:[^\[\]\r\n]{0,32})\](?!\])"
)


def _citation_only(text: str) -> bool:
    if not isinstance(text, str):
        return False
    has_reference = _WIKILINK_RE.search(text) or _CITATION_REF_RE.search(text)
    remainder = _CITATION_REF_RE.sub("", _WIKILINK_RE.sub("", text))
    return bool(has_reference) and all(
        char.isspace() or unicodedata.category(char).startswith("P")
        for char in remainder
    )


def _answer_template(template: str, *, streaming: bool = False) -> str:
    """Remove only the trusted output-format block, before inserting evidence.

    Both routes retain identical grounding instructions. Legacy templates can
    number their final JSON rule differently; never strip at a rule number.
    Custom templates without a format block pass through unchanged.
    """
    start, end = "<json-output-format>", "</json-output-format>"
    before, marker, remainder = template.partition(start)
    block, closed, after = remainder.partition(end)
    if marker and closed:
        return before + ("" if streaming else block) + after
    if streaming:
        # Compatibility for the historical template: its JSON rule and
        # example were the final instructions before the Context placeholder.
        rule = re.search(r"(?m)^\d+\.\s+Return ONLY\b", template)
        if rule:
            context = template.find("Context:", rule.end())
            if context != -1:
                return template[:rule.start()] + template[context:]
    return template


def _answer_request(question: str) -> str:
    """Keep grounding instructions salient at the generation boundary.

    This is the same reminder for both output modes, not a retrieval rewrite,
    a claim-specific hint, a citation repair or an additional model call.
    """
    return (
        f"Question: {question}\n\n"
        "Answer with the requested fact in prose, followed by the citation of "
        "the passage that explicitly states that fact. A citation alone is not "
        "an answer. Copy the passage's short citation token [cite:N] when "
        "provided; its number identifies a passage, NOT a page. Otherwise copy "
        "its exact [[...]] source header label, including any #page or #section "
        "anchor. Do not cite a different passage. "
        "Keep only explicitly supported relationships. Do not omit the citations."
    )

ANSWER_STATUSES = frozenset({
    "supported", "insufficient_evidence", "conflicting_evidence", "uncertain",
})
_INSUFFICIENT_CUES = (
    "don't know", "do not know", "not in the context", "not in the vault",
    "no information", "insufficient evidence", "not enough evidence",
    "cannot answer", "can't answer", "no hay información", "no hay informacion",
    "evidencia insuficiente", "no consta", "no se menciona",
)
_CONFLICT_CUES = (
    "conflicting", "conflict between", "sources disagree", "contradict",
    "en conflicto", "se contradicen", "fuentes discrepan", "discrepancia",
)
_UNCERTAINTY_CUES = (
    "uncertain", "unclear", "ambiguous", "may indicate", "might indicate",
    "no está claro", "no esta claro", "incierto", "ambigua", "ambiguo",
)


def answer_status(answer: str, sources, declared=None) -> str:
    """Normalise the answer's evidence state to the public four-value contract.

    The model declares the state in non-streaming mode. Streaming and older
    backends fall back to conservative text cues. No retrieved source always
    wins over a model claiming support.
    """
    text = (answer or "").strip().lower()
    if not sources or not text:
        return "insufficient_evidence"
    # A model can obey citation formatting while omitting the actual answer.
    # Do not let its declared "supported" disguise a citation-only response.
    # Preserve meaningful symbols (e.g. an emoji), numbers and ordinary prose.
    if _citation_only(text):
        return "insufficient_evidence"
    raw = str(declared or "").strip().lower()
    if raw in ANSWER_STATUSES:
        return raw
    if any(cue in text for cue in _INSUFFICIENT_CUES):
        return "insufficient_evidence"
    if any(cue in text for cue in _CONFLICT_CUES):
        return "conflicting_evidence"
    if any(cue in text for cue in _UNCERTAINTY_CUES):
        return "uncertain"
    return "supported"


def _format_source_label(title: str, page, heading) -> str:
    """Render a wikilink-style source label with the best available anchor.

    Page wins over heading (more precise) when both are present.
    Returns just ``title`` when neither anchor is available so MD / TXT
    citations stay exactly what they were in v2.0.
    """
    if page:
        return f"{title}#p.{page}"
    if heading:
        # Strip newlines / pipes that would break the wikilink syntax.
        clean = " ".join(str(heading).split())[:80]
        if clean:
            return f"{title}#{clean}"
    return title


def _indexed_extraction_summary(record: dict) -> dict | None:
    """Bounded PDF coverage at indexing time, distinct from live file state."""
    if record.get("format") != "pdf":
        return None
    raw = record.get("extraction")
    if not isinstance(raw, dict) or raw.get("status") not in {
        "complete", "recovered", "partial",
    }:
        return {"status": "unknown", "missing_pages": [], "missing_page_count": None}
    issues = raw.get("issues") if isinstance(raw.get("issues"), list) else []
    missing_pages = sorted({
        issue["page"] for issue in issues
        if isinstance(issue, dict)
        and not issue.get("recovered")
        and isinstance(issue.get("page"), int)
        and not isinstance(issue.get("page"), bool)
        and issue["page"] > 0
    })
    return {
        "status": raw["status"],
        "missing_pages": missing_pages[:24],
        "missing_page_count": len(missing_pages),
    }


def _coverage_context_note(detail: dict) -> str:
    """Trusted instruction beside a source whose indexed PDF was incomplete."""
    coverage = detail.get("indexed_extraction") or {}
    if coverage.get("status") == "partial":
        pages = ", ".join(map(str, coverage.get("missing_pages") or []))
        pages_label = f" (including page(s) {pages})" if pages else ""
        return (
            "[PDF extraction warning: text was unavailable from part of this "
            f"document{pages_label} when indexed. Do not infer that the "
            "retrieved passages cover the whole PDF.]\n"
        )
    if coverage.get("status") == "unknown":
        return (
            "[PDF extraction coverage was not recorded for this indexed "
            "document. Do not assume it was fully extracted.]\n"
        )
    return ""

class Oracle:
    """
    Implements the RAG pipeline to provide context-aware answers to user queries.
    """
    # Class-level default so the cap resolves even on instances built without
    # __init__ -- tests exercise _build_context against a bare
    # ``Oracle.__new__(Oracle)`` to isolate it from the LLM and the DB.
    # __init__ shadows this with the configured value.
    context_max_chars = _ORACLE_CONTEXT_MAX_CHARS
    conditional_rewrite = True
    rewrite_timeout_s = _REWRITE_TIMEOUT_S
    answer_sufficiency_check = False
    answer_sufficiency_timeout_s = 180

    def __init__(self, config, db: Database, router: LLMRouter, embedder: Embedder):
        self.config = config
        self.db = db
        self.router = router
        self.embedder = embedder
        self.connector = Connector(
            db, embedder, router=router,
            vector_backend=getattr(config.cognition, "vector_backend", "auto"),
            rerank_engine=getattr(config.cognition, "rerank_engine", "llm"),
            rerank_model=getattr(
                config.cognition, "rerank_model", "BAAI/bge-reranker-base"
            ),
            matrix_cache_enabled=bool(getattr(
                config.cognition, "vec_matrix_cache", True)),
            rrf_k=int(getattr(config.cognition, "rrf_k", 60)),
            bm25_weight=float(getattr(
                config.cognition, "bm25_weight", CognitionConfig.bm25_weight)),
        )
        # Snapshot the context cap once. Read via getattr so a partially
        # populated config (the SimpleNamespace stand-ins used in tests, or a
        # TOML written before this key existed) still yields the historical
        # 16 KB rather than an AttributeError.
        self.context_max_chars = int(getattr(
            config.cognition, "context_max_chars", _ORACLE_CONTEXT_MAX_CHARS
        ))
        self.conditional_rewrite = bool(getattr(
            config.cognition, "conditional_rewrite", True
        ))
        self.rewrite_timeout_s = int(getattr(
            config.cognition, "rewrite_timeout_s", _REWRITE_TIMEOUT_S
        ))
        self.system_prompt_template = self._load_prompt()
        self.answer_sufficiency_check = getattr(config.cognition, "answer_sufficiency_check", False) is True
        self.answer_sufficiency_timeout_s = getattr(config.cognition, "answer_sufficiency_timeout_s", 180)

    def _load_prompt(self):
        """Loads the Oracle's system prompt template."""
        prompt_path = Path(__file__).parent / "prompts" / "oracle.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def ask(self, question: str, top_k: int = 5, extra_sources=None, history=None,
            retrieval_k: "int | None" = None,
            filter_note_ids: "set[int] | None" = None) -> dict:
        """
        Main RAG entry point:
        1. Generates an embedding for the user's question.
        2. Retrieves the most relevant chunks from the database.
        3. Constructs a system prompt containing the retrieved context.
        4. Calls the LLM to generate a cited answer.

        ``extra_sources`` carries user-explicit attachments (``@note``
        mentions or ``/pin`` pins) that get priority on the context
        budget. Same wrap_untrusted defence applies.
        """
        logger.info("oracle_query", question=question)
        # Per-stage latency, threaded through retrieval and filled here. Lets
        # eval attribute end-to-end time to embed/retrieve/rerank/generate
        # instead of a single opaque number.
        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        history = self._normalize_history(history)
        t_rewrite = time.perf_counter()
        retrieval_query = self._rewrite_query(question, history)
        timings["rewrite_s"] = time.perf_counter() - t_rewrite
        full_context, sources, retrieved, source_details = self._build_context_payload(
            retrieval_query, top_k, extra_sources=extra_sources,
            retrieval_k=retrieval_k, timings=timings,
            filter_note_ids=filter_note_ids,
        )
        if full_context is None:
            timings["generate_s"] = 0.0
            timings["total_s"] = time.perf_counter() - t_total
            return {
                "answer": "Your vault seems empty of relevant whispers on this subject.",
                "sources": [],
                "source_details": [],
                "citation_selections": [],
                "answer_status": "insufficient_evidence",
                "retrieved": retrieved,
                "dropped_citations": 0,
                "timings": timings,
            }

        full_context, sources, source_details, sufficiency, blocked = self._check_evidence(
            retrieval_query, full_context, sources, retrieved, source_details, question, timings,
        )
        if blocked is not None:
            timings["generate_s"] = 0.0
            timings["total_s"] = time.perf_counter() - t_total
            return {**blocked, "timings": timings}
        if sufficiency is not None:
            result = self._generate_checked(question, full_context, sources, retrieved,
                                            source_details, history, sufficiency, timings)
            timings["total_s"] = time.perf_counter() - t_total
            return {**result, "timings": timings}
        system_prompt = _answer_template(self.system_prompt_template).replace("{context}", full_context)
        if history:
            system_prompt = self._format_history(history) + "\n\n" + system_prompt
        t_gen = time.perf_counter()
        response = self.router.complete(
            prompt=_answer_request(question),
            system_prompt=system_prompt,
        )
        timings["generate_s"] = time.perf_counter() - t_gen

        # ``generation_error`` is internal: the API and MCP whitelist their
        # fields, so only the evaluator sees it. It separates a model that
        # could not answer (transport failure, missing answer field) from a
        # model that chose to abstain; both render the same silent sentinel.
        if not response or not isinstance(response, dict):
            logger.warning("oracle_no_response")
            timings["total_s"] = time.perf_counter() - t_total
            return {"answer": "The Oracle is silent.", "sources": sources,
                    "citation_selections": [],
                    "source_details": source_details, "retrieved": retrieved,
                    "answer_status": "insufficient_evidence",
                    "dropped_citations": 0, "timings": timings, "sufficiency": sufficiency,
                    "generation_error": "no_model_response"}

        raw_answer = response.get("answer", "The Oracle is silent.")
        answer, dropped, selections = self.resolve_citations(
            raw_answer, sources, source_details
        )
        timings["total_s"] = time.perf_counter() - t_total
        return {
            "answer": answer,
            "sources": sources,
            "source_details": source_details,
            "citation_selections": selections,
            "answer_status": answer_status(
                answer, sources,
                "insufficient_evidence" if _citation_only(raw_answer) else response.get("status")
            ),
            "retrieved": retrieved,
            "dropped_citations": dropped,
            "timings": timings,
            "sufficiency": sufficiency,
            **({} if "answer" in response else {"generation_error": "missing_answer_field"}),
        }

    def _check_evidence(self, query, context, sources, retrieved, details, question, timings):
        if self.answer_sufficiency_check is not True:
            return context, sources, details, None, None
        selected_context, selected, audit = assess(
            self.router, query, context, details, _coverage_context_note,
            timeout_s=self.answer_sufficiency_timeout_s,
        )
        timings["sufficiency_s"] = audit["elapsed_s"]
        if audit["decision"] == "answerable":
            return selected_context, list(dict.fromkeys(d["label"] for d in selected)), selected, audit, None
        failed = audit["decision"] == "unassessed"
        blocked = {"answer": blocked_answer(question, failed=failed), "sources": sources,
                   "source_details": details, "retrieved": retrieved, "citation_selections": [],
                   "answer_status": "uncertain" if failed else "insufficient_evidence",
                   "dropped_citations": 0, "sufficiency": audit}
        return context, sources, details, audit, blocked

    def _generate_checked(self, question, context, sources, retrieved, details, history, audit, timings):
        """Opt-in all-or-nothing cited claims; never repair an uncited answer.

        Both UI delivery modes use this same structured protocol. Validation
        precedes any display, and failure never triggers retries or a fallback
        with looser evidence rules. Literal references are not semantic proof.
        """
        started = time.perf_counter()
        generation = {"protocol_version": PROTOCOL_VERSION, "decision": "unassessed",
                      "error": None, "model_response": None, "model_calls": 0,
                      "semantic_support": "unverified"}
        audit["generation"] = generation
        result = {"sources": sources, "source_details": details, "retrieved": retrieved,
                  "sufficiency": audit, "citation_selections": [], "dropped_citations": 0,
                  "answer_status": "uncertain", "answer": blocked_answer(question, failed=True, stage="generation")}
        try:
            template = (Path(__file__).parent / "prompts/cited_answer.txt").read_text(encoding="utf-8")
            system_prompt = template.replace("{context}", context)
            if history:
                system_prompt = self._format_history(history) + "\n\n" + system_prompt
            safe_question = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(question), label="question")
            generation["model_calls"] = 1
            response = self.router.complete(prompt=f"Question:\n{safe_question}", system_prompt=system_prompt)
            generation["model_response"] = response_snapshot(response)
            raw, status, error = render_claims(response, details)
            generation["error"] = error
            if error:
                return result
            if status == "insufficient_evidence":
                generation["decision"] = "insufficient"
                return {**result, "answer": blocked_answer(question), "answer_status": status}
            answer, dropped, selections = self.resolve_citations(raw, sources, details)
            if dropped or not selections:
                generation["error"] = "answer_reference_resolution_failed"
                return result
            generation["decision"] = "accepted"
            return {**result, "answer": answer, "answer_status": status,
                    "citation_selections": selections}
        except Exception:
            generation["error"] = "answer_generation_failed"
            return result
        finally:
            timings["generate_s"] = time.perf_counter() - started
            generation["elapsed_s"] = timings["generate_s"]

    def retrieve(self, question: str, top_k: int = 10, history=None,
                 timings: "dict | None" = None,
                 filter_note_ids: "set[int] | None" = None) -> list[dict]:
        """Retrieval-only path: rank the top-``top_k`` sources for a question
        *without* generating an answer.

        Runs the exact retrieval pipeline ``ask`` uses — history-aware query
        rewrite, embedding, hybrid/dense search, optional rerank — and returns
        the rank-ordered, note-deduped list (same shape as the ``retrieved``
        key on :meth:`ask`). Skipping the answer LLM makes this fast and, for a
        single-turn question, deterministic given the index; the only
        non-determinism is the follow-up query rewrite, which mirrors ``ask``
        so retrieval metrics stay apples-to-apples with the full pipeline.

        Used by the eval harness's ``--retrieval-only`` mode to measure Hit@k /
        MRR cheaply on every change without paying for (or being perturbed by)
        generation. Pass a ``timings`` dict to have the per-stage latency
        (rewrite/embed/retrieve/rerank) filled in place — the return type stays
        a plain list so existing callers are unaffected.
        """
        logger.info("oracle_retrieve", question=question)
        history = self._normalize_history(history)
        t_rewrite = time.perf_counter()
        retrieval_query = self._rewrite_query(question, history)
        if timings is not None:
            timings["rewrite_s"] = time.perf_counter() - t_rewrite
        # ``top_k`` here is the ranking depth; the discarded context is built
        # at the same depth, which is fine — we only want ``retrieved``.
        _, _, retrieved = self._build_context(
            retrieval_query, top_k, timings=timings,
            filter_note_ids=filter_note_ids,
        )
        return retrieved

    def rewrite_retrieval_query(self, question: str, history=None) -> str:
        """Return the history-aware query used by the retrieval pipeline.

        The HTTP search companion uses this public boundary so related
        passages resolve follow-up references exactly like answer retrieval,
        while retaining its lightweight hit payload.
        """
        return self._rewrite_query(question, self._normalize_history(history))

    @staticmethod
    def verify_citations(text: str, sources, source_details=None) -> tuple[str, int]:
        """Unlink any ``[[wikilink]]`` the model emitted that wasn't among the
        retrieved ``sources`` — a hallucinated citation.

        Returns ``(cleaned_text, dropped_count)``. Ungrounded citations keep
        their words but lose the ``[[ ]]`` so the prose stays readable while no
        longer pointing at a note that didn't inform the answer. Matching is
        tolerant of a missing ``#anchor`` (a bare ``[[Title]]`` is accepted for
        a retrieved ``[[Title#p.4]]``) so legitimate citations aren't stripped.
        """
        cleaned, dropped, _ = Oracle.resolve_citations(text, sources, source_details)
        return cleaned, dropped

    @staticmethod
    def resolve_citations(text: str, sources, source_details=None) -> tuple[str, int, list[dict]]:
        """Verify once, recording each accepted occurrence in answer order.

        Indices address this response's delivered ``source_details``, not the
        database or physical pages. A canonical label shared by several
        passages remains ambiguous; never choose one by content or rank.
        No raw answer or rejected model text is retained in the trace.
        """
        if not text:
            return text, 0, []
        allowed = set(sources or [])
        allowed_titles = {s.split("#", 1)[0].strip() for s in (sources or [])}
        dropped = 0
        # Only references explicitly published with the delivered context can
        # resolve. No ranking heuristic, page-number inference or cached map.
        references: dict[str, tuple[str, int]] = {}
        selections: list[dict] = []
        details = source_details or []
        duplicate_refs = set()
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                continue
            ref, label = detail.get("citation_ref"), detail.get("label")
            if not isinstance(ref, str) or not re.fullmatch(r"cite:[1-9]\d*", ref):
                continue
            if not isinstance(label, str) or label not in allowed:
                continue
            if ref in references:
                duplicate_refs.add(ref)
            references[ref] = (label, index)
        for ref in duplicate_refs:
            references.pop(ref)

        def _sub(match: "re.Match") -> str:
            nonlocal dropped
            wiki, short = match.groups()
            label = wiki.strip() if wiki is not None else short
            title = label.split("#", 1)[0].strip()
            ref = None
            if wiki is not None and (label in allowed or ("#" not in label and title in allowed_titles)):
                indices = [i for i, detail in enumerate(details) if isinstance(detail, dict)
                           and isinstance(detail.get("label"), str)
                           and (detail["label"] == label or
                                ("#" not in label and detail["label"].split("#", 1)[0].strip() == title))
                           and detail["label"] in allowed]
                replacement = match.group(0)
            # Some models retain double brackets around the short token. It
            # is unambiguous because the builder skips canonical-name clashes.
            elif label in references:
                ref = label
                label, index = references[ref]
                indices = [index]
                replacement = f"[[{label}]]"
            else:
                dropped += 1
                if short is not None:
                    logger.warning("oracle_citation_ref_unknown", citation_ref=label)
                else:
                    logger.warning("oracle_citation_hallucinated", citation=label)
                return label
            selections.append({
                "citation_index": len(selections), "label": label,
                "form": "short" if ref is not None else "wikilink",
                "citation_ref": ref, "source_detail_indices": indices,
                "selection": "exact" if len(indices) == 1 else "ambiguous" if indices else "unmapped",
            })
            return replacement

        cleaned = _EMITTED_CITATION_RE.sub(_sub, text)
        return cleaned, dropped, selections

    # ── conversation memory ──────────────────────────────────────────────

    _HISTORY_MAX_CHARS = 2_000

    @staticmethod
    def _normalize_history(history) -> list[dict]:
        """Coerce request-supplied ``history`` into a safe list of turns.

        ``history`` arrives from the HTTP API / MCP caller (or the shell's
        own turn log) and is interpolated into both the retrieval-rewrite
        prompt and the system prompt. Unlike retrieved note content and
        ``extra_sources`` it previously skipped the injection guard, so a
        caller could otherwise smuggle role markers or chat-template tokens
        through it. A non-list value would also crash the ``history[-3:]``
        iteration and turn malformed input into a server error.

        Returns a list of ``{"q", "a"}`` dicts with both fields run through
        :meth:`SecurityGuard.sanitize_prompt`. Anything that isn't a list of
        dicts is dropped, so a malformed value degrades to "no history"
        rather than raising.
        """
        if not isinstance(history, list):
            return []
        turns: list[dict] = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            q = turn.get("q")
            a = turn.get("a")
            q = SecurityGuard.sanitize_prompt(q) if isinstance(q, str) else ""
            a = SecurityGuard.sanitize_prompt(a) if isinstance(a, str) else ""
            if q or a:
                turns.append({"q": q, "a": a})
        return turns

    # Words that point at something said earlier rather than naming it. Only
    # third-person and demonstrative forms: "you" and "I" refer to the speaker,
    # not to a previous turn, so "What does a Kotlin data class generate for
    # you?" is self-contained and must not match.
    #
    # Spanish needs care that English does not. "la", "los" and "las" are
    # articles far more often than pronouns, and "otro"/"mismo" are ordinary
    # adjectives, so none of them are listed -- a false positive here costs a
    # needless LLM round-trip, which is the entire thing being removed.
    _REFERENTIAL_WORDS = frozenset("""
        it its itself they them their theirs he him his she her hers
        that this these those there then such same one ones former latter
        ella ellos ellas eso esto esa ese esas esos aquel aquella aquello
        su sus le les ahi alli alla entonces
    """.split())

    # Matched *before* accent folding, because folding destroys the only thing
    # that separates them from an everyday word: "él" (pronoun) becomes "el"
    # (definite article), which appears in almost any Spanish sentence. That
    # collision made "¿Qué es el método mayéutico de Sócrates?" -- a fully
    # self-contained question -- look like a follow-up needing resolution.
    _REFERENTIAL_ACCENTED = frozenset({"él", "aquí", "ahí", "allí", "allá"})

    # A question opening with a coordinating conjunction is a continuation of
    # the previous turn almost by definition ("And what else is in it?",
    # "¿Y cuáles son sus desventajas?").
    _CONTINUATION_OPENERS = frozenset({"and", "y", "but", "pero", "or", "o",
                                       "so", "then", "entonces", "también",
                                       "tambien"})

    # Below this many words a question rarely carries enough of its own topic
    # to retrieve on ("¿Por qué?", "Expand on that"). Deliberately small: the
    # cost of guessing wrong in this direction is one round-trip, while
    # guessing wrong the other way loses the answer.
    _SHORT_QUESTION_WORDS = 5

    @classmethod
    def _needs_rewrite(cls, question: str, history) -> bool:
        """Whether this question can be retrieved on without its history.

        The rewrite costs a full LLM round-trip, and most follow-ups do not
        need it: a question that names its own subject retrieves the same
        documents either way. What genuinely needs resolving is a question
        that *points* at the previous turn instead of restating it.

        Three signals, any of which is enough:

        * a referential word (pronoun, demonstrative, possessive);
        * an opening conjunction, which marks a continuation;
        * fewer than :py:attr:`_SHORT_QUESTION_WORDS` words, too little to
          retrieve on regardless of vocabulary.

        Biased towards rewriting on purpose. A false positive wastes one
        round-trip; a false negative retrieves against an unresolved pronoun
        and can lose the answer entirely, so the asymmetry decides every
        borderline call.
        """
        if not history:
            return False
        words = re.findall(r"[^\W\d_]+", question.lower(), flags=re.UNICODE)
        if not words:
            return True
        if len(words) < cls._SHORT_QUESTION_WORDS:
            return True
        if words[0] in cls._CONTINUATION_OPENERS:
            return True
        if set(words) & cls._REFERENTIAL_ACCENTED:
            return True
        # Fold for the rest, so an unaccented "ahi" still matches.
        folded = {
            unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
            for w in words
        }
        return bool(folded & cls._REFERENTIAL_WORDS)

    def _rewrite_query(self, question: str, history) -> str:
        """Condense a follow-up + recent turns into one standalone retrieval
        query, resolving pronouns/references ("expand on that" → the topic).

        Returns the original ``question`` unchanged when there's no history,
        or whenever the rewrite LLM call fails (circuit open, Ollama down,
        unparseable JSON) — retrieval then behaves exactly as it did before
        conversation memory existed.
        """
        if not history:
            return question
        if self.conditional_rewrite and not self._needs_rewrite(question, history):
            logger.debug("oracle_query_rewrite_skipped", question=question)
            return question
        convo = "\n".join(
            f"Q: {t.get('q', '').strip()}\nA: {(t.get('a', '') or '').strip()[:300]}"
            for t in history[-3:]
        )
        prompt = (
            f"Conversation so far:\n{convo}\n\n"
            f"Follow-up question: {question}\n\n"
            "Rewrite the follow-up as a single self-contained search query that "
            "resolves any pronouns or references to earlier turns. Keep it short.\n"
            'Return ONLY JSON: {"query": "..."}'
        )
        try:
            resp = self.router.complete(
                prompt=prompt,
                system_prompt="You rewrite follow-up questions into standalone search queries.",
                json_format=True,
                timeout_s=self.rewrite_timeout_s,
                # Best-effort work on a tight deadline: its timeouts say
                # nothing about backend health, and letting them open the
                # circuit breaker would disable answer generation too.
                optional=True,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("oracle_query_rewrite_failed", error=str(e))
            return question
        if isinstance(resp, dict):
            rewritten = resp.get("query")
            if isinstance(rewritten, str) and rewritten.strip():
                logger.info(
                    "oracle_query_rewritten",
                    original=question,
                    rewritten=rewritten.strip(),
                )
                return rewritten.strip()
        return question

    def _format_history(self, history) -> str:
        """Render the last few turns as a bounded 'recent conversation' block
        for the system prompt. Capped to :attr:`_HISTORY_MAX_CHARS` so a long
        thread can't crowd out the retrieved context."""
        lines: list[str] = []
        for turn in history[-3:]:
            q = (turn.get("q", "") or "").strip()
            a = (turn.get("a", "") or "").strip()
            if q:
                lines.append(f"User: {q}")
            if a:
                lines.append(f"Oracle: {a}")
        block = (
            "Recent conversation (for continuity only — cite ONLY from the "
            "Context section below):\n" + "\n".join(lines)
        )
        return block[: self._HISTORY_MAX_CHARS]

    def _titles_for(self, items) -> dict:
        """``{note_id: title}`` for the retrieved set, in one query.

        No per-note fallback: ``self.db`` is a :class:`Database`, which always
        carries the batch methods. Duck-typing the capability here would be
        worse than useless — a ``MagicMock`` stand-in answers ``getattr`` with
        a mock that silently returns a mock instead of a mapping, so the
        "compatible" branch would be the one that breaks.
        """
        return self.db.get_note_titles([it['note_id'] for it in items])

    def _anchors_for(self, items) -> dict:
        """``{(note_id, text): (page, heading)}`` for the cited chunks, in one
        query."""
        return self.db.get_chunk_anchors_bulk(
            [(it['note_id'], it['text']) for it in items]
        )

    def _build_context_payload(self, question: str, top_k: int, extra_sources=None,
                               retrieval_k: "int | None" = None,
                               timings: "dict | None" = None,
                               filter_note_ids: "set[int] | None" = None):
        """Run retrieval and return context plus structured source details.

        Returns ``(context, sources, retrieved, source_details)``.  The public
        compatibility wrapper below keeps the historical three-tuple for
        callers that use this helper directly.

        ``extra_sources`` is a list of ``(title, raw_body)`` pairs from
        user-explicit attachments (``@note``, ``/pin``). They are prepended
        to the candidate list with the same wrap_untrusted defence applied
        so they get priority on the char budget without bypassing the
        prompt-injection guard.

        ``retrieval_k`` decouples the *ranking* depth from the *context*
        depth. When set (and larger than ``top_k``) the connector returns a
        deeper pool so the ``retrieved`` ranking list can score, say, MRR@10
        — but only the first ``top_k`` chunks ever feed the generation
        context, so the answer the model sees is unchanged. ``None`` ⇒ the
        legacy behaviour where ranking depth == context depth == ``top_k``.
        """
        # An explicitly empty scope is a final answer, not a cue to retry
        # globally. Avoid even embedding the question so direct Oracle users
        # get the same fail-closed behaviour as the HTTP API. Pinned sources
        # remain usable because they were selected explicitly by the caller.
        if filter_note_ids is not None and not filter_note_ids and not extra_sources:
            if timings is not None:
                timings.update({"embed_s": 0.0, "retrieve_s": 0.0, "rerank_s": 0.0})
            return None, [], [], []

        n_retrieve = max(top_k, retrieval_k) if retrieval_k else top_k
        t_embed = time.perf_counter()
        query_vector = self.embedder.embed(question)
        embed_s = time.perf_counter() - t_embed
        use_hybrid = (
            getattr(self.config.cognition, "hybrid_search", True)
            and self.db.fts_available
        )
        rerank_s = 0.0
        t_search = time.perf_counter()
        if use_hybrid:
            search_timings: dict[str, float] = {}
            similar = self.connector.find_hybrid(
                query_text=question,
                query_vector=query_vector,
                top_k=n_retrieve,
                filter_note_ids=filter_note_ids,
                rrf_k=getattr(self.config.cognition, "rrf_k", 60),
                rerank=getattr(self.config.cognition, "rerank", False),
                rerank_pool=getattr(self.config.cognition, "rerank_pool", 20),
                timings=search_timings,
                include_embedding_id=True,
            )
            rerank_s = search_timings.get("rerank_s", 0.0)
        elif query_vector:
            similar = self.connector.find_similar_notes(
                query_vector, top_k=n_retrieve, filter_note_ids=filter_note_ids,
                include_embedding_id=True,
            )
        else:
            similar = []
        # "retrieve" is dense + BM25 + fusion; the optional rerank is subtracted
        # out so it shows up as its own bucket rather than inflating retrieval.
        if timings is not None:
            timings["embed_s"] = embed_s
            timings["retrieve_s"] = max(time.perf_counter() - t_search - rerank_s, 0.0)
            timings["rerank_s"] = rerank_s

        if not similar and not extra_sources:
            return None, [], [], []

        # Ranked, note-deduped retrieval order. Captured here because the
        # ``set()`` dedup and the char-budget cap below both destroy rank
        # order before ``ask`` returns — this list is the *only* place the
        # true retrieval rank survives. Eval ranking metrics (Hit@k / MRR)
        # read it back via the ``retrieved`` key. Pinned ``extra_sources``
        # are deliberately excluded: they're user-supplied, not retrieved,
        # so folding them in would corrupt the ranking signal.
        retrieved: list[dict] = []
        _seen_notes: set[int] = set()

        candidate_parts: list[tuple[str, str, dict]] = []

        # User-explicit attachments first — they're the highest signal.
        for title, raw_body in (extra_sources or []):
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(raw_body),
                label="source",
            )
            candidate_parts.append(
                (
                    title,
                    f"--- Source: [[{title}]] (pinned) ---\n{safe_text}",
                    {
                        "id": f"pinned:{title}",
                        "label": title,
                        "document": {
                            "note_id": None, "title": title,
                            "path": None, "format": None,
                        },
                        "locator": {
                            "chunk_index": None, "page": None, "heading": None,
                        },
                        "passage": {"text": raw_body, "completeness": "full"},
                        "source_state": "provided",
                        "indexed_file_hash": None,
                        "current_file_hash": None,
                    },
                )
            )

        # Fetch metadata in bulk. Titles cover the full ranking pool;
        # evidence identifies the exact embedding occurrence. The text-based
        # anchor lookup remains a fallback for legacy retrieval without IDs.
        titles = self._titles_for(similar)
        anchors = self._anchors_for(similar[:top_k])
        embedding_ids = [
            int(item["embedding_id"])
            for item in similar[:top_k]
            if item.get("embedding_id") is not None
        ]
        evidence_records = self.db.get_evidence_records(embedding_ids) \
            if embedding_ids else {}
        if not isinstance(evidence_records, dict):
            evidence_records = {}

        for rank_pos, item in enumerate(similar):
            title = titles.get(item['note_id'])
            if not title:
                logger.warning("orphan_embedding", note_id=item['note_id'])
                continue
            note_id = item['note_id']
            if note_id not in _seen_notes:
                _seen_notes.add(note_id)
                retrieved.append({
                    "title": title,
                    "note_id": note_id,
                    "score": item.get("score"),
                    "rank": len(retrieved) + 1,
                })
            # Only the first ``top_k`` retrieved chunks feed the generation
            # context; anything deeper exists purely so ``retrieved`` can span
            # the ranking pool (e.g. MRR@10 behind a top-5 answer).
            if rank_pos >= top_k:
                continue
            record = evidence_records.get(item.get("embedding_id"))
            if record is not None and record.get("note_id") != note_id:
                # A stale/malformed retrieval result must never associate
                # evidence from a different document with this citation.
                record = None
            if record is not None:
                # Excerpts can repeat on different pages, or collide after
                # truncation. Even absent anchors belong to THIS occurrence;
                # don't borrow the first matching excerpt's page or heading.
                page, heading = record.get("page"), record.get("heading")
            else:
                page, heading = anchors.get(
                    (item['note_id'], item['text']), (None, None)
                )
            label = _format_source_label(title, page, heading)
            detail = self._source_detail(
                item, title=title, label=label, page=page,
                heading=heading,
                record=record,
            )
            # Generation sees the same complete indexed passage exposed to
            # the reviewer.  Legacy rows fall back to their bounded excerpt
            # only when safe reconstruction is impossible.
            context_text = detail["passage"]["text"] or item["text"]
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(context_text),
                label="source",
            )
            candidate_parts.append(
                (
                    label,
                    f"--- Source: [[{label}]] ---\n"
                    f"{_coverage_context_note(detail)}{safe_text}",
                    detail,
                )
            )

        accepted_parts: list[str] = []
        sources: list[str] = []
        source_details: list[dict] = []
        used = 0
        dropped = 0
        reserved_refs = {
            name for label, _part, _detail in candidate_parts
            for name in (label, label.split("#", 1)[0].strip())
        }
        ref_index = 1
        for title, part, detail in candidate_parts:
            while f"cite:{ref_index}" in reserved_refs:
                ref_index += 1
            citation_ref = f"cite:{ref_index}"
            part = f"Citation token: [{citation_ref}]\n{part}"
            extra = len(part) + (len(_CONTEXT_SEPARATOR) if accepted_parts else 0)
            if used + extra > self.context_max_chars:
                dropped += 1
                continue
            accepted_parts.append(part)
            sources.append(title)
            source_details.append({**detail, "citation_ref": citation_ref})
            ref_index += 1
            used += extra

        if dropped:
            logger.info(
                "oracle_context_truncated",
                kept=len(accepted_parts),
                dropped=dropped,
                cap=self.context_max_chars,
                used_chars=used,
            )

        if not accepted_parts:
            # Retrieval found notes but the char cap dropped every one — a
            # distinct failure mode from "nothing retrieved". Hand back the
            # ranked list so eval can tell the two apart.
            return None, [], retrieved, []

        return (
            _CONTEXT_SEPARATOR.join(accepted_parts),
            list(dict.fromkeys(sources)),
            retrieved,
            source_details,
        )

    def _build_context(self, question: str, top_k: int, extra_sources=None,
                       retrieval_k: "int | None" = None,
                       timings: "dict | None" = None,
                       filter_note_ids: "set[int] | None" = None):
        """Compatibility wrapper returning the historical three-tuple."""
        context, sources, retrieved, _details = self._build_context_payload(
            question, top_k, extra_sources=extra_sources,
            retrieval_k=retrieval_k, timings=timings,
            filter_note_ids=filter_note_ids,
        )
        return context, sources, retrieved

    def _source_detail(self, item: dict, *, title: str, label: str,
                       page, heading, record: "dict | None",
                       recover_legacy: bool = True) -> dict:
        """Build one inspectable reference tied to the indexed source version.

        Read-only clients can disable legacy reconstruction: semantic chunking
        may contact the embedding service, whereas reading stored text cannot.
        """
        if not record:
            # Retrieval stand-ins and legacy connectors may not expose an
            # embedding id. Preserve a useful, explicitly unverified excerpt.
            return {
                "id": f"note:{item.get('note_id')}:unresolved",
                "label": label,
                "document": {
                    "note_id": item.get("note_id"), "title": title,
                    "path": None, "format": None,
                },
                "locator": {
                    "chunk_index": None, "page": page, "heading": heading,
                },
                "passage": {
                    "text": item.get("text") or "", "completeness": "excerpt",
                },
                "source_state": "unverified",
                "indexed_file_hash": None,
                "current_file_hash": None,
                "indexed_extraction": None,
            }

        state, current_hash, display_path = self._source_file_state(record)
        passage = record.get("full_text")
        complete = passage is not None
        if not complete and state == "current" and recover_legacy:
            recovered = self._recover_full_passage(record)
            if recovered is not None:
                passage = recovered
                complete = True
        if passage is None:
            passage = record.get("excerpt") or item.get("text") or ""

        return {
            "id": (
                f"note:{record['note_id']}:chunk:{record['chunk_index']}:"
                f"embedding:{record['embedding_id']}"
            ),
            "label": label,
            "document": {
                "note_id": record["note_id"],
                "title": record.get("title") or title,
                "path": display_path,
                "format": record.get("format") or "md",
            },
            "locator": {
                "chunk_index": record["chunk_index"],
                "page": record.get("page"),
                "heading": record.get("heading"),
            },
            "passage": {
                "text": passage,
                "completeness": "full" if complete else "excerpt",
            },
            "source_state": state,
            "indexed_file_hash": record.get("file_hash"),
            "current_file_hash": current_hash,
            "indexed_extraction": _indexed_extraction_summary(record),
        }

    def _source_file_state(self, record: dict) -> tuple[str, "str | None", "str | None"]:
        """Compare the current file bytes with the version recorded at scan."""
        raw_path = record.get("path")
        if not raw_path:
            return "unverified", None, None
        path = Path(raw_path)
        vault_cfg = getattr(self.config, "vault", None)
        vault_path = getattr(vault_cfg, "path", None)
        if not vault_path:
            return "unverified", None, None
        vault_root = Path(vault_path).resolve()
        try:
            resolved = SecurityGuard.resolve_within_vault(path, vault_root)
            display_path = str(resolved.relative_to(vault_root))
        except (OSError, ValueError):
            return "unavailable", None, None
        if not resolved.is_file():
            return "missing", None, display_path
        indexed_hash = record.get("file_hash")
        if not indexed_hash:
            return "unverified", None, display_path
        try:
            current_hash = sha256_file(resolved)
        except OSError:
            return "unavailable", None, display_path
        state = "current" if current_hash == indexed_hash else "modified"
        return state, current_hash, display_path

    def _recover_full_passage(self, record: dict) -> "str | None":
        """Recreate a legacy full chunk only from an unchanged source file."""
        from grimore.cognition.chunker import build_candidate_chunks
        from grimore.ingest.parser import MarkdownParser

        try:
            vault_root = Path(self.config.vault.path).resolve()
            path = SecurityGuard.resolve_within_vault(record["path"], vault_root)
            note = MarkdownParser().parse_file(
                path, vault_root=vault_root, config=self.config,
            )
            clean = SecurityGuard.sanitize_prompt(note.content)
            chunks = build_candidate_chunks(note, clean, self.embedder, self.config)
            candidate = chunks[int(record["chunk_index"])]
        except (IndexError, KeyError, OSError, TypeError, ValueError):
            return None

        excerpt = record.get("excerpt") or ""
        if excerpt and not candidate.text.startswith(excerpt):
            return None
        stored_hash = record.get("chunk_hash")
        model = getattr(self.embedder, "model", "")
        if stored_hash and Embedder.chunk_hash(candidate.text, model) != stored_hash:
            return None
        return candidate.text

    def ask_stream(self, question: str, top_k: int = 5, extra_sources=None,
                   history=None,
                   filter_note_ids: "set[int] | None" = None) -> Iterator[dict]:
        """Streaming variant of :meth:`ask`.

        Yields events::

            {"type": "token", "text": "..."}       # zero-or-more, as LLM emits
            {"type": "done", "answer": "...", "sources": [...]}  # terminal

        Falls back to a single ``done`` with an empty ``sources`` list when
        retrieval finds nothing or when the streaming call returns no tokens
        (Ollama unreachable, circuit open). Always emits a terminal ``done``
        so the caller can render its final UI state in one place.

        The streaming path strips the JSON-only rule from the system prompt
        so the model emits plain prose; ``ask()`` keeps its JSON contract
        for callers like ``--export`` that depend on a structured payload.
        With the optional sufficiency check, both modes instead use validated
        cited claims and emit a single final event, without provisional facts.
        """
        logger.info("oracle_query_stream", question=question)
        history = self._normalize_history(history)
        retrieval_query = self._rewrite_query(question, history)
        full_context, sources, retrieved, source_details = self._build_context_payload(
            retrieval_query, top_k, extra_sources=extra_sources,
            filter_note_ids=filter_note_ids,
        )
        if full_context is None:
            yield {
                "type": "done", "sources": [], "source_details": [],
                "citation_selections": [],
                "answer_status": "insufficient_evidence",
                "retrieved": retrieved, "dropped_citations": 0,
            }
            return

        timings = {}
        full_context, sources, source_details, sufficiency, blocked = self._check_evidence(
            retrieval_query, full_context, sources, retrieved, source_details, question, timings,
        )
        if blocked is not None:
            yield {"type": "done", **blocked, "timings": timings}
            return
        if sufficiency is not None:
            # No unchecked factual tokens escape. Buffer only the opt-in
            # protocol; legacy streaming remains incremental and unchanged.
            result = self._generate_checked(question, full_context, sources, retrieved,
                                            source_details, history, sufficiency, timings)
            yield {"type": "done", **result, "timings": timings}
            return
        streaming_template = _answer_template(self.system_prompt_template, streaming=True)
        system_prompt = streaming_template.replace("{context}", full_context)
        if history:
            system_prompt = self._format_history(history) + "\n\n" + system_prompt

        produced = False
        chunks: list[str] = []
        for chunk in self.router.complete_streaming(
            prompt=_answer_request(question),
            system_prompt=system_prompt,
        ):
            produced = True
            chunks.append(chunk)
            yield {"type": "token", "text": chunk}

        if not produced:
            logger.warning("oracle_stream_no_response")
        # Tokens are already on screen, but the final event carries the verified
        # answer so clients can replace their provisional rendering and remove
        # citations that were not present in the retrieved context.
        raw_answer = "".join(chunks)
        verified_answer, dropped, selections = self.resolve_citations(raw_answer, sources, source_details)
        yield {
            "type": "done",
            "answer": verified_answer,
            "sources": sources,
            "source_details": source_details,
            "citation_selections": selections,
            "answer_status": answer_status(
                verified_answer, sources,
                "insufficient_evidence" if _citation_only(raw_answer) else None,
            ),
            "retrieved": retrieved,
            "dropped_citations": dropped,
            "sufficiency": sufficiency,
            "timings": timings,
            # Same marker as ``ask``: no tokens means the model failed, not
            # that the context lacked evidence.
            **({} if produced else {"generation_error": "no_model_response"}),
        }
