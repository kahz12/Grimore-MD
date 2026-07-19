"""
Persistence Layer (SQLite).
This module manages the SQLite database, handling note metadata, tags,
and vector embeddings. It uses WAL mode to allow concurrent access.

:class:`Database` is the single public entry point — callers keep doing
``Database(db_path)`` and calling methods on the instance. The actual
data access is split by domain into sibling modules, mixed in here:

* :mod:`grimore.memory.schema`              — connections, init, migrations
* :mod:`grimore.memory.search`              — FTS5 + sqlite-vec retrieval
* :mod:`grimore.memory.notes`               — note metadata, categories, prune
* :mod:`grimore.memory.chunks`              — embedding rows, vec mirror, cache
* :mod:`grimore.memory.embedding_migration` — resumable model-swap workflow
* :mod:`grimore.memory.tags`                — tag sync and lookups
* :mod:`grimore.memory.upkeep`              — VACUUM / WAL checkpoint
* :mod:`grimore.memory.freshness`           — Chronicler freshness rows
* :mod:`grimore.memory.mirror_store`        — Black Mirror claims/contradictions
"""
from typing import Optional

from grimore.memory.chunks import ChunksMixin
from grimore.memory.embedding_migration import EmbeddingMigrationMixin
from grimore.memory.freshness import FreshnessMixin
from grimore.memory.mirror_store import MirrorStoreMixin
from grimore.memory.notes import NotesMixin
from grimore.memory.schema import SchemaMixin
from grimore.memory.search import SearchMixin
from grimore.memory.tags import TagsMixin
from grimore.memory.upkeep import UpkeepMixin


class Database(
    SchemaMixin,
    SearchMixin,
    NotesMixin,
    ChunksMixin,
    EmbeddingMigrationMixin,
    TagsMixin,
    UpkeepMixin,
    FreshnessMixin,
    MirrorStoreMixin,
):
    """
    Manages all database operations for Project Grimore.
    Ensures the schema is initialized and provides high-level methods for data access.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        # sqlite-vec capability is probed once at startup. ``_vec_available``
        # gates every other vec-aware code path so a missing extension
        # degrades silently to the numpy fast path.
        self._vec_available: bool = self._probe_vec_extension()
        self._vec_dim: Optional[int] = None
        self._init_db()
