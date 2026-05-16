"""
The Oracle: Retrieval-Augmented Generation (RAG) Engine.
This module combines semantic search (via the Connector) with LLM completion
(via the LLMRouter) to answer questions based on the vault's content.
"""
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

class Oracle:
    """
    Implements the RAG pipeline to provide context-aware answers to user queries.
    """
    def __init__(self, config, db: Database, router: LLMRouter, embedder: Embedder):
        self.config = config
        self.db = db
        self.router = router
        self.embedder = embedder
        self.connector = Connector(db, embedder)
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self):
        """Loads the Oracle's system prompt template."""
        prompt_path = Path(__file__).parent / "prompts" / "oracle.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def ask(self, question: str, top_k: int = 5, extra_sources=None) -> dict:
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
        full_context, sources = self._build_context(
            question, top_k, extra_sources=extra_sources
        )
        if full_context is None:
            return {"answer": "Your vault seems empty of relevant whispers on this subject.", "sources": []}

        system_prompt = self.system_prompt_template.replace("{context}", full_context)
        response = self.router.complete(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
        )

        if not response or not isinstance(response, dict):
            logger.warning("oracle_no_response")
            return {"answer": "The Oracle is silent.", "sources": sources}

        return {
            "answer": response.get("answer", "The Oracle is silent."),
            "sources": sources,
        }

    def _build_context(self, question: str, top_k: int, extra_sources=None):
        """Run retrieval + context-cap and return (full_context, sources).

        Returns ``(None, [])`` when retrieval finds nothing usable, so the
        caller can short-circuit without a second branch. Pulled out of
        ``ask()`` so ``ask_stream()`` can reuse the exact same retrieval
        logic — keeping JSON and streaming paths in lockstep.

        ``extra_sources`` is a list of ``(title, raw_body)`` pairs from
        user-explicit attachments (``@note``, ``/pin``). They are prepended
        to the candidate list with the same wrap_untrusted defence applied
        so they get priority on the char budget without bypassing the
        prompt-injection guard.
        """
        query_vector = self.embedder.embed(question)
        use_hybrid = (
            getattr(self.config.cognition, "hybrid_search", True)
            and self.db.fts_available
        )
        if use_hybrid:
            similar = self.connector.find_hybrid(
                query_text=question,
                query_vector=query_vector,
                top_k=top_k,
                rrf_k=getattr(self.config.cognition, "rrf_k", 60),
            )
        elif query_vector:
            similar = self.connector.find_similar_notes(query_vector, top_k=top_k)
        else:
            similar = []

        if not similar and not extra_sources:
            return None, []

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

        for item in similar:
            title = self.db.get_note_title(item['note_id'])
            if not title:
                logger.warning("orphan_embedding", note_id=item['note_id'])
                continue
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(item['text']),
                label="source",
            )
            candidate_parts.append(
                (title, f"--- Source: [[{title}]] ---\n{safe_text}")
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
            return None, []

        return _CONTEXT_SEPARATOR.join(accepted_parts), list(set(sources))

    def ask_stream(self, question: str, top_k: int = 5, extra_sources=None) -> Iterator[dict]:
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
        full_context, sources = self._build_context(question, top_k, extra_sources=extra_sources)
        if full_context is None:
            yield {"type": "done", "sources": []}
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

        produced = False
        for chunk in self.router.complete_streaming(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
        ):
            produced = True
            yield {"type": "token", "text": chunk}

        if not produced:
            logger.warning("oracle_stream_no_response")
        yield {"type": "done", "sources": sources}
