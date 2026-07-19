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
    from contextlib import AbstractContextManager

    class DbBase:
        # Set by Database.__init__ / SchemaMixin._init_db.
        db_path: str
        _vec_available: bool
        _vec_dim: Optional[int]
        _fts_available: bool

        # Provided by SchemaMixin. Context manager: commit on success,
        # rollback on exception, always closes.
        def _get_connection(self) -> AbstractContextManager[sqlite3.Connection]:
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
