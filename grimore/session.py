"""
Session: shared, lazy-built service handles.

The CLI uses a one-shot Session per command (built and torn down at the
boundary). The interactive shell uses one Session for the whole loop —
which is the whole point: Database, Embedder, LLMRouter and Oracle stay
warm across commands so consecutive `ask` calls don't pay cold-start
cost on the embedder + router.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from grimore.cognition.embedder import Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.oracle import Oracle
from grimore.memory.db import Database
from grimore.utils.config import Config, load_config


@dataclass
class NoteAttachment:
    """A vault note pinned (via ``/pin``) or one-shot-attached (via ``@``)
    to the next ask. ``title`` is what the shell renders; ``path`` is the
    resolved, vault-scoped file path; ``content`` is the read-and-capped
    body. All three are filled by the @-resolver, never by user input."""
    title: str
    path: Path
    content: str


class Session:
    """Caches the four services every Knowledge-ops command needs.

    Properties build on first access and cache the instance. Calling
    ``refresh()`` drops everything so the next access rebuilds it —
    useful from the shell after the user scans the vault from another
    terminal and wants the new state visible without exiting.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()
        self._db: Optional[Database] = None
        self._router: Optional[LLMRouter] = None
        self._embedder: Optional[Embedder] = None
        self._oracle: Optional[Oracle] = None
        # Conversational state used by the redesigned shell. None of this
        # touches disk — it's per-session only.
        self.question_log: list[str] = []
        self.last_question: Optional[str] = None
        self.last_answer: Optional[dict] = None  # {"answer": str, "sources": list[str]}
        self.pinned_notes: list[NoteAttachment] = []
        # One-shot attachments parsed out of ``@…`` mentions in the next ask.
        # Cleared after every dispatched question.
        self.staged_attachments: list[NoteAttachment] = []
        # Rolling conversation memory for the shell: the last few
        # ``{"q", "a", "sources"}`` turns. Feeds the Oracle's query-rewrite
        # and answer-coherence context so follow-ups resolve. The one-shot
        # CLI never populates this, so its behaviour is unchanged.
        self.turns: list[dict] = []

    # Number of prior turns kept for conversational context. Small on
    # purpose — enough to resolve "expand on that" without blowing the
    # local model's context window or leaking the whole session.
    MAX_TURNS = 3

    def record_turn(self, question: str, answer: str, sources: list[str]) -> None:
        """Append one Q&A turn, trimming to the last :attr:`MAX_TURNS`."""
        self.turns.append({"q": question, "a": answer or "", "sources": list(sources or [])})
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]

    def forget(self) -> None:
        """Drop all conversational state (``/forget``) without touching the
        warm service handles or the on-disk shell history file."""
        self.turns = []
        self.last_question = None
        self.last_answer = None
        self.question_log = []

    @staticmethod
    def slugify(text: str, *, max_words: int = 6) -> str:
        """Turn a question into a filesystem-safe slug.

        Lowercases, keeps `[a-z0-9]` runs only, joins with ``-``, caps
        to the first ``max_words`` words. Empty input falls back to
        ``"thread"`` so callers never write a bare ``.jsonl`` filename.
        """
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if not words:
            return "thread"
        return "-".join(words[:max_words])

    def save_turns(self, path: Path) -> Path:
        """Write the rolling :attr:`turns` to ``path`` as JSONL.

        Each line is a single turn JSON object enriched with a wall-clock
        ``ts`` field so listings can sort by recency without stat'ing the
        file's mtime. Atomic rename: write to ``path.tmp`` then replace,
        so a crash mid-write can't leave a half-flushed thread on disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for turn in self.turns:
                row = {
                    "ts": now,
                    "q": turn.get("q", ""),
                    "a": turn.get("a", ""),
                    "sources": list(turn.get("sources") or []),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        return path

    def load_turns(self, path: Path) -> int:
        """Load JSONL turns from ``path`` into :attr:`turns`.

        Replaces (does not extend) the current turns, then re-applies the
        :attr:`MAX_TURNS` window. Also primes ``last_question`` /
        ``last_answer`` from the final loaded turn so ``/again`` and
        ``/why`` work immediately after a resume. Returns the number of
        turns actually loaded after the cap is applied.
        """
        path = Path(path)
        loaded: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                loaded.append({
                    "q": row.get("q", ""),
                    "a": row.get("a", ""),
                    "sources": list(row.get("sources") or []),
                })
        self.turns = loaded[-self.MAX_TURNS:] if loaded else []
        if self.turns:
            last = self.turns[-1]
            self.last_question = last["q"]
            self.last_answer = {
                "answer": last["a"],
                "sources": list(last["sources"]),
            }
        return len(self.turns)

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database(self.config.memory.db_path)
        return self._db

    @property
    def router(self) -> LLMRouter:
        if self._router is None:
            self._router = LLMRouter(self.config)
        return self._router

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.config, cache=self.db)
        return self._embedder

    @property
    def oracle(self) -> Oracle:
        if self._oracle is None:
            self._oracle = Oracle(self.config, self.db, self.router, self.embedder)
        return self._oracle

    @property
    def vault_root(self) -> Path:
        return Path(self.config.vault.path)

    def refresh(self) -> None:
        """Drop cached services so the next access rebuilds them.

        Also clears conversation memory: ``refresh()`` runs when the vault
        may have changed underneath us (e.g. a scan from another terminal),
        so stale conversational context is dropped along with the services.
        """
        self._db = None
        self._router = None
        self._embedder = None
        self._oracle = None
        self.turns = []

    def set_chat_model(self, name: str) -> None:
        """Override the LLM model for this session only.

        ``LLMRouter.complete`` reads the model name from config on every
        call, so no service rebuild is needed — but the Oracle cached the
        router reference at build time, which still picks up the new
        config (same router instance), so we leave both alone.
        """
        self.config.cognition.model_llm_local = name

    def set_embedding_model(self, name: str) -> None:
        """Override the embedding model for this session only.

        Unlike the LLM, ``Embedder`` snapshots ``self.model`` at
        construction (because it goes into the cache key), so we drop
        the cached embedder *and* the oracle (which holds a reference to
        the old embedder). Next access rebuilds both with the new model.
        """
        self.config.cognition.model_embeddings_local = name
        self._embedder = None
        self._oracle = None

    def close(self) -> None:
        """Idempotent teardown. Database has no explicit close (per-call
        sqlite3 connections), so this just drops references and lets GC
        do the rest."""
        self.refresh()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
