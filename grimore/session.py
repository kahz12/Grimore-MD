"""
Session: shared, lazy-built service handles.

The CLI uses a one-shot Session per command (built and torn down at the
boundary). The interactive shell uses one Session for the whole loop —
which is the whole point: Database, Embedder, LLMRouter and Oracle stay
warm across commands so consecutive `ask` calls don't pay cold-start
cost on the embedder + router.
"""
from __future__ import annotations

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
        """Drop cached services so the next access rebuilds them."""
        self._db = None
        self._router = None
        self._embedder = None
        self._oracle = None

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
