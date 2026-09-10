"""On-disk cache for the dense scoring matrix.

A one-shot ``grimore ask`` builds the whole ``(N, D)`` matrix inside its only
query and then exits, so every invocation pays the full read. This stores the
built matrix next to the database as a ``.npy`` and reloads it with
``mmap_mode="r"``, which both skips the rebuild and keeps the pages out of RSS
until they are actually touched.

**On the seal.** The obvious key is the embeddings signature ``(count,
max_id)``, and for the Connector's in-process cache that is enough. It is *not*
enough for a cache that outlives the process:
``swap_embedding_migration`` installs a re-embedded table with
``INSERT INTO embeddings (id, ...) SELECT id, ... FROM embeddings_migration``,
preserving every id. After migrating to a different model the count and max id
are identical while every vector has changed. A stale disk cache would then be
served indefinitely, silently ranking against the old model's vectors.

So the seal also carries the total vector byte length, which catches any change
of dimension, and the migration swap calls :func:`clear` outright, which catches
a same-dimension model swap. The remaining hole is a same-dimension rewrite that
bypasses the migration path (editing the DB by hand); that is out of scope for a
cache and is why :func:`clear` is public.

Path containment needs no separate check: every path here is derived from
``db_path`` by changing its suffix, so the cache can only ever land in the
directory the database already lives in.
"""
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from grimore.utils.logger import get_logger

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a declared dep
    # Unlike the cognition modules, this one is inside mypy's gated packages,
    # so the rebinding to None needs to be declared rather than inferred.
    _np = None  # type: ignore[assignment]

logger = get_logger(__name__)

_MATRIX_SUFFIX = ".vecmat.npy"
_SEAL_SUFFIX = ".vecmat.sig"
_TOKEN_BYTES = 32


def cache_paths(db_path: str) -> Optional[tuple[Path, Path]]:
    """``(matrix_path, seal_path)`` for ``db_path``, or ``None`` when the
    database has no file to sit beside (``:memory:``, or a blank path)."""
    if not db_path or db_path == ":memory:" or db_path.startswith("file::memory:"):
        return None
    base = Path(db_path)
    if not base.name:
        return None
    return (base.with_name(base.name + _MATRIX_SUFFIX),
            base.with_name(base.name + _SEAL_SUFFIX))


def _seal_text(signature: tuple[int, int, int]) -> str:
    return ":".join(str(int(part)) for part in signature)


def load(db_path: str, signature: tuple[int, int, int], rows: int, dim: int):
    """Return the memory-mapped matrix when the cache matches, else ``None``.

    The shape is re-checked against ``rows``/``dim`` even after the seal
    matches: a truncated or half-written file can carry a valid seal if the
    process died between the two writes, and a wrong shape must be a miss
    rather than an exception on the retrieval path.
    """
    if _np is None:
        return None
    paths = cache_paths(db_path)
    if paths is None:
        return None
    matrix_path, seal_path = paths
    try:
        if not (matrix_path.exists() and seal_path.exists()):
            return None
        seal = seal_path.read_text().strip()
        prefix, separator, token = seal.rpartition(":")
        if not separator or prefix != _seal_text(signature) or len(token) != _TOKEN_BYTES:
            return None
        # Map the same open file whose publication token we validated. Loading
        # by pathname again would race another writer's os.replace(). The token
        # is a trailer after the ordinary NPY array, ignored by numpy.load.
        with matrix_path.open("rb") as fh:
            fh.seek(-_TOKEN_BYTES, os.SEEK_END)
            if fh.read(_TOKEN_BYTES) != token.encode("ascii"):
                return None
            fh.seek(0)
            version = _np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = _np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype = _np.lib.format.read_array_header_2_0(fh)
            else:
                return None
            if shape != (rows, dim) or dtype != _np.float32 or rows <= 0 or dim <= 0:
                return None
            offset = fh.tell()
            if os.fstat(fh.fileno()).st_size != offset + rows * dim * 4 + _TOKEN_BYTES:
                return None
            matrix = _np.memmap(fh, mode="r", dtype=dtype, shape=shape,
                                offset=offset, order="F" if fortran else "C")
    except Exception as exc:
        # A corrupt or unreadable cache must never break retrieval; the
        # caller rebuilds from SQLite.
        logger.warning("matrix_cache_load_failed", error=str(exc))
        return None
    if matrix.shape != (rows, dim) or matrix.dtype != _np.float32:
        logger.warning(
            "matrix_cache_shape_mismatch",
            expected=(rows, dim), found=tuple(matrix.shape),
        )
        return None
    return matrix


def save(db_path: str, matrix, signature: tuple[int, int, int]) -> None:
    """Publish matrix and seal with a shared, unique generation token.

    Each replacement is atomic; the pair is not. A token in both files makes
    any interleaving of concurrent publishers a cache miss, never a false hit.
    Old caches without a token are rebuilt automatically.
    """
    if _np is None or matrix is None:
        return
    paths = cache_paths(db_path)
    if paths is None:
        return
    matrix_path, seal_path = paths
    # Unique temp names, not "<final>.tmp": the daemon and a CLI run share one
    # database, so a fixed name lets one writer truncate the file the other is
    # still filling. The loser then publishes a mix of both writes, which has a
    # valid header and the right shape and so survives every check in load().
    tmp_matrix = tmp_seal = None
    token = uuid.uuid4().hex
    try:
        fd, tmp_matrix_name = tempfile.mkstemp(
            dir=str(matrix_path.parent), prefix=matrix_path.name + ".", suffix=".tmp")
        tmp_matrix = Path(tmp_matrix_name)
        with os.fdopen(fd, "wb") as fh:
            _np.save(fh, _np.asarray(matrix, dtype=_np.float32))
            fh.write(token.encode("ascii"))
            fh.flush()
            # Without this, a crash after the rename can leave the file at full
            # length with unwritten blocks reading as zeros -- correct shape,
            # correct dtype, all-zero scores, and nothing downstream notices.
            os.fsync(fh.fileno())
        tmp_matrix.replace(matrix_path)
        tmp_matrix = None

        fd, tmp_seal_name = tempfile.mkstemp(
            dir=str(seal_path.parent), prefix=seal_path.name + ".", suffix=".tmp")
        tmp_seal = Path(tmp_seal_name)
        with os.fdopen(fd, "w") as fh:
            fh.write(_seal_text(signature) + ":" + token)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_seal.replace(seal_path)
        tmp_seal = None
    except Exception as exc:
        # Cache writes are best-effort: a read-only directory or a full disk
        # must not take retrieval down with them.
        logger.warning("matrix_cache_save_failed", error=str(exc))
    finally:
        for leftover in (tmp_matrix, tmp_seal):
            if leftover is None:
                continue
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass


def clear(db_path: str) -> None:
    """Invalidate the cache. Called whenever the vectors may have changed in a
    way the seal cannot detect -- notably the embedding-model migration swap,
    which preserves row ids.

    The seal goes first, and its removal is what actually invalidates: a matrix
    with no seal is always a miss. That ordering matters on Windows, where a
    file another process has memory-mapped cannot be unlinked at all, so
    deleting the matrix may fail while the (unmapped) seal still deletes
    cleanly. Reversing the order would leave a live cache behind whenever a
    daemon happened to hold the mapping.
    """
    paths = cache_paths(db_path)
    if paths is None:
        return
    matrix_path, seal_path = paths
    for path in (seal_path, matrix_path):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            # A mapped matrix on Windows lands here. Invalidation already
            # happened via the seal, so this is reclaimable disk, not
            # correctness -- log it and leave the file for the next attempt.
            logger.warning("matrix_cache_clear_failed", path=str(path), error=str(exc))
