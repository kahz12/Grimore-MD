"""
The Oracle: Retrieval-Augmented Generation (RAG) Engine.
This module combines semantic search (via the Connector) with LLM completion
(via the LLMRouter) to answer questions based on the vault's content.
"""
import re
from pathlib import Path
from typing import Iterator

from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.embedder import Embedder
from grimore.cognition.connector import Connector
from grimore.memory.db import Database
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Hard cap on the total context (\n\n-joined sources) injected into the LLM
# system prompt. With top_k×500 chars per chunk, a high --top-k could push
# >25 KB into a 32 k-token window — fine on qwen2.5:3b, fragile on smaller
# local models. 16 KB leaves headroom for the question, the prompt template
# and the model's own answer space (B-07).
_ORACLE_CONTEXT_MAX_CHARS = 16_000
_CONTEXT_SEPARATOR = "\n\n"

# Matches a single ``[[wikilink]]`` citation; group 1 is the inner label.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


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

class Oracle:
    """
    Implements the RAG pipeline to provide context-aware answers to user queries.
    """
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
        )
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self):
        """Loads the Oracle's system prompt template."""
        prompt_path = Path(__file__).parent / "prompts" / "oracle.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def ask(self, question: str, top_k: int = 5, extra_sources=None, history=None,
            retrieval_k: "int | None" = None) -> dict:
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
        history = self._normalize_history(history)
        retrieval_query = self._rewrite_query(question, history)
        full_context, sources, retrieved = self._build_context(
            retrieval_query, top_k, extra_sources=extra_sources, retrieval_k=retrieval_k
        )
        if full_context is None:
            return {
                "answer": "Your vault seems empty of relevant whispers on this subject.",
                "sources": [],
                "retrieved": retrieved,
                "dropped_citations": 0,
            }

        system_prompt = self.system_prompt_template.replace("{context}", full_context)
        if history:
            system_prompt = self._format_history(history) + "\n\n" + system_prompt
        response = self.router.complete(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
        )

        if not response or not isinstance(response, dict):
            logger.warning("oracle_no_response")
            return {"answer": "The Oracle is silent.", "sources": sources,
                    "retrieved": retrieved, "dropped_citations": 0}

        answer, dropped = self.verify_citations(
            response.get("answer", "The Oracle is silent."), sources
        )
        return {
            "answer": answer,
            "sources": sources,
            "retrieved": retrieved,
            "dropped_citations": dropped,
        }

    def retrieve(self, question: str, top_k: int = 10, history=None) -> list[dict]:
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
        generation.
        """
        logger.info("oracle_retrieve", question=question)
        history = self._normalize_history(history)
        retrieval_query = self._rewrite_query(question, history)
        # ``top_k`` here is the ranking depth; the discarded context is built
        # at the same depth, which is fine — we only want ``retrieved``.
        _, _, retrieved = self._build_context(retrieval_query, top_k)
        return retrieved

    @staticmethod
    def verify_citations(text: str, sources) -> tuple[str, int]:
        """Unlink any ``[[wikilink]]`` the model emitted that wasn't among the
        retrieved ``sources`` — a hallucinated citation.

        Returns ``(cleaned_text, dropped_count)``. Ungrounded citations keep
        their words but lose the ``[[ ]]`` so the prose stays readable while no
        longer pointing at a note that didn't inform the answer. Matching is
        tolerant of a missing ``#anchor`` (a bare ``[[Title]]`` is accepted for
        a retrieved ``[[Title#p.4]]``) so legitimate citations aren't stripped.
        """
        if not text:
            return text, 0
        allowed = set(sources or [])
        allowed_titles = {s.split("#", 1)[0].strip() for s in (sources or [])}
        dropped = 0

        def _sub(match: "re.Match") -> str:
            nonlocal dropped
            label = match.group(1).strip()
            title = label.split("#", 1)[0].strip()
            if label in allowed or title in allowed_titles:
                return match.group(0)
            dropped += 1
            logger.warning("oracle_citation_hallucinated", citation=label)
            return label

        cleaned = _WIKILINK_RE.sub(_sub, text)
        return cleaned, dropped

    # ── conversation memory ──────────────────────────────────────────────

    _HISTORY_MAX_CHARS = 2_000

    @staticmethod
    def _normalize_history(history) -> list[dict]:
        """Coerce request-supplied ``history`` into a safe list of turns.

        ``history`` arrives from the HTTP API / MCP caller (or the shell's
        own turn log) and is interpolated into both the retrieval-rewrite
        prompt and the system prompt. Unlike retrieved note content and
        ``extra_sources`` it previously skipped the injection guard, so a
        caller could smuggle role markers / chat-template tokens through it
        (audit L3); a non-list value would also crash the ``history[-3:]``
        iteration with an ``AttributeError`` → 500.

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

    def _build_context(self, question: str, top_k: int, extra_sources=None,
                       retrieval_k: "int | None" = None):
        """Run retrieval + context-cap and return (full_context, sources, retrieved).

        Returns ``(None, [], retrieved)`` when no context is usable, so the
        caller can short-circuit without a second branch. Pulled out of
        ``ask()`` so ``ask_stream()`` can reuse the exact same retrieval
        logic — keeping JSON and streaming paths in lockstep.

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
        n_retrieve = max(top_k, retrieval_k) if retrieval_k else top_k
        query_vector = self.embedder.embed(question)
        use_hybrid = (
            getattr(self.config.cognition, "hybrid_search", True)
            and self.db.fts_available
        )
        if use_hybrid:
            similar = self.connector.find_hybrid(
                query_text=question,
                query_vector=query_vector,
                top_k=n_retrieve,
                rrf_k=getattr(self.config.cognition, "rrf_k", 60),
                rerank=getattr(self.config.cognition, "rerank", False),
                rerank_pool=getattr(self.config.cognition, "rerank_pool", 20),
            )
        elif query_vector:
            similar = self.connector.find_similar_notes(query_vector, top_k=n_retrieve)
        else:
            similar = []

        if not similar and not extra_sources:
            return None, [], []

        # Ranked, note-deduped retrieval order. Captured here because the
        # ``set()`` dedup and the char-budget cap below both destroy rank
        # order before ``ask`` returns — this list is the *only* place the
        # true retrieval rank survives. Eval ranking metrics (Hit@k / MRR)
        # read it back via the ``retrieved`` key. Pinned ``extra_sources``
        # are deliberately excluded: they're user-supplied, not retrieved,
        # so folding them in would corrupt the ranking signal.
        retrieved: list[dict] = []
        _seen_notes: set[int] = set()

        candidate_parts: list[tuple[str, str]] = []

        # User-explicit attachments first — they're the highest signal.
        for title, raw_body in (extra_sources or []):
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(raw_body),
                label="source",
            )
            candidate_parts.append(
                (title, f"--- Source: [[{title}]] (pinned) ---\n{safe_text}")
            )

        for rank_pos, item in enumerate(similar):
            title = self.db.get_note_title(item['note_id'])
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
            page, heading = self.db.get_chunk_anchors(item['note_id'], item['text'])
            label = _format_source_label(title, page, heading)
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(item['text']),
                label="source",
            )
            candidate_parts.append(
                (label, f"--- Source: [[{label}]] ---\n{safe_text}")
            )

        accepted_parts: list[str] = []
        sources: list[str] = []
        used = 0
        dropped = 0
        for title, part in candidate_parts:
            extra = len(part) + (len(_CONTEXT_SEPARATOR) if accepted_parts else 0)
            if used + extra > _ORACLE_CONTEXT_MAX_CHARS:
                dropped += 1
                continue
            accepted_parts.append(part)
            sources.append(title)
            used += extra

        if dropped:
            logger.info(
                "oracle_context_truncated",
                kept=len(accepted_parts),
                dropped=dropped,
                cap=_ORACLE_CONTEXT_MAX_CHARS,
                used_chars=used,
            )

        if not accepted_parts:
            # Retrieval found notes but the char cap dropped every one — a
            # distinct failure mode from "nothing retrieved". Hand back the
            # ranked list so eval can tell the two apart.
            return None, [], retrieved

        return _CONTEXT_SEPARATOR.join(accepted_parts), list(set(sources)), retrieved

    def ask_stream(self, question: str, top_k: int = 5, extra_sources=None, history=None) -> Iterator[dict]:
        """Streaming variant of :meth:`ask`.

        Yields events::

            {"type": "token", "text": "..."}       # zero-or-more, as LLM emits
            {"type": "done",  "sources": [...]}    # exactly one, terminal

        Falls back to a single ``done`` with an empty ``sources`` list when
        retrieval finds nothing or when the streaming call returns no tokens
        (Ollama unreachable, circuit open). Always emits a terminal ``done``
        so the caller can render its final UI state in one place.

        The streaming path strips the JSON-only rule from the system prompt
        so the model emits plain prose; ``ask()`` keeps its JSON contract
        for callers like ``--export`` that depend on a structured payload.
        """
        logger.info("oracle_query_stream", question=question)
        history = self._normalize_history(history)
        retrieval_query = self._rewrite_query(question, history)
        full_context, sources, retrieved = self._build_context(retrieval_query, top_k, extra_sources=extra_sources)
        if full_context is None:
            yield {"type": "done", "sources": [], "retrieved": retrieved, "dropped_citations": 0}
            return

        # Drop the JSON-output rule and example block from the template so
        # the streamed answer is plain prose. The marker we anchor on is
        # the literal "6. Return ONLY..." rule line; if the prompt is ever
        # rewritten, this falls back to the full template (model still
        # produces valid prose, just wrapped in JSON the user will see).
        streaming_template = self.system_prompt_template
        json_rule_idx = streaming_template.find('6. Return ONLY')
        if json_rule_idx != -1:
            context_idx = streaming_template.find('Context:', json_rule_idx)
            if context_idx != -1:
                streaming_template = (
                    streaming_template[:json_rule_idx]
                    + streaming_template[context_idx:]
                )

        system_prompt = streaming_template.replace("{context}", full_context)
        if history:
            system_prompt = self._format_history(history) + "\n\n" + system_prompt

        produced = False
        chunks: list[str] = []
        for chunk in self.router.complete_streaming(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
        ):
            produced = True
            chunks.append(chunk)
            yield {"type": "token", "text": chunk}

        if not produced:
            logger.warning("oracle_stream_no_response")
        # Tokens are already on screen, so we can't unlink in place; surface a
        # count instead so the caller can warn the user about ungrounded links.
        _, dropped = self.verify_citations("".join(chunks), sources)
        yield {"type": "done", "sources": sources, "retrieved": retrieved, "dropped_citations": dropped}
