"""
Session: shared, lazy-built service handles.

The CLI uses a one-shot Session per command (built and torn down at the
boundary). The interactive shell uses one Session for the whole loop —
which is the whole point: Database, Embedder, LLMRouter and Oracle stay
warm across commands so consecutive `ask` calls don't pay cold-start
cost on the embedder + router.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from grimore.cognition.embedder import Embedder
from grimore.cognition.llm_router import LLMRouter
from grimore.cognition.oracle import Oracle
from grimore.memory.db import Database
from grimore.utils.config import Config, load_config


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

    def close(self) -> None:
        """Idempotent teardown. Database has no explicit close (per-call
        sqlite3 connections), so this just drops references and lets GC
        do the rest."""
        self.refresh()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
