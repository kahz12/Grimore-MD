"""
Typing contract shared by the :class:`~grimore.memory.db.Database` mixins.

Each mixin module implements one domain slice but calls into members
provided by its siblings (``_get_connection`` from schema, the vec
helpers from chunks, …). ``DbBase`` declares that cross-mixin surface
once so mypy can check the mixins in isolation. It exists only at
type-check time — at runtime every mixin still inherits plain
``object``, so composition, MRO, and failure behaviour (a genuine
missing attribute raises ``AttributeError``) are unchanged.

When a mixin starts relying on a member provided by a *different*
mixin, declare it here; members a mixin defines and calls on itself
don't belong in the contract.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import sqlite3
    import threading
    from collections.abc import Iterable
    from contextlib import AbstractContextManager

    class DbBase:
        # Set by Database.__init__ / SchemaMixin._init_db.
        db_path: str
        _vec_available: bool
        _vec_dim: Optional[int]
        _fts_available: bool
        # Per-thread connection storage and the registry that lets close()
        # reclaim connections opened on other threads.
        _local: threading.local
        _connections: dict[int, tuple[threading.Thread, sqlite3.Connection]]
        _conn_lock: threading.Lock
        _generation: int

        # Provided by SchemaMixin. Context manager: commit on success,
        # rollback on exception. Yields the calling thread's long-lived
        # connection; closing it is close()'s job, not the block's.
        def _get_connection(self) -> AbstractContextManager[sqlite3.Connection]:
            raise NotImplementedError

        def close(self) -> None:
            raise NotImplementedError

        def _reap_dead_locked(self) -> None:
            raise NotImplementedError

        def store_embeddings_bulk(self, note_id: int, rows: list[dict]) -> int:
            raise NotImplementedError

        def get_chunk_anchors_bulk(
            self, pairs: "Iterable[tuple[int, str]]",
        ) -> dict[tuple[int, str], tuple[Optional[int], Optional[str]]]:
            raise NotImplementedError

        def _vec_write(self, conn, sql: str, payload: list, dim: int,
                       event: str, **context) -> None:
            raise NotImplementedError

        def _create_vec_table(self, conn, dim: int) -> None:
            raise NotImplementedError

        def _migrate_vec_table(self, conn) -> None:
            raise NotImplementedError

        # Provided by SearchMixin.
        @property
        def fts_available(self) -> bool:
            raise NotImplementedError

        # Provided by ChunksMixin.
        def _delete_embeddings_for_note(self, conn, note_id: int) -> None:
            raise NotImplementedError

        def drop_vec_table(self) -> None:
            raise NotImplementedError
else:
    DbBase = object
