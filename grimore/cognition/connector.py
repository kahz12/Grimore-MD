"""
Semantic Connection Discovery.
This module provides logic to find related notes by comparing their vector
embeddings and, optionally, fusing vector ranking with BM25 (FTS5) ranking
via Reciprocal Rank Fusion.
"""
import time
from typing import List, Optional

from grimore.cognition.embedder import Embedder
from grimore.cognition.reranker import Reranker, build_reranker
from grimore.memory.db import Database
from grimore.utils import matrix_cache
from grimore.utils.logger import get_logger

try:  # vectorized scoring fast path; the per-row loop is the fallback.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a declared dep
    _np = None

logger = get_logger(__name__)


class Connector:
    """
    Handles the discovery of relationships between notes.

    Two retrieval strategies are exposed:

    * :py:meth:`find_similar_notes` — dense only (cosine similarity).
    * :py:meth:`find_hybrid` — dense + BM25 fused with Reciprocal Rank Fusion.

    ``router`` is optional and only consulted by the opt-in LLM re-rank pass
    in :py:meth:`find_hybrid`; dense/hybrid retrieval works without it.
    """
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        router=None,
        vector_backend: str = "auto",
        rerank_engine: str = "llm",
        rerank_model: str = "BAAI/bge-reranker-base",
        reranker: Optional[Reranker] = None,
        matrix_cache_enabled: bool = True,
    ):
        self.db = db
        self.embedder = embedder
        self.router = router
        # "auto" picks sqlite-vec when the extension + table are ready, else
        # numpy. "numpy" pins the matmul path even with sqlite-vec installed
        # (handy for parity tests). "sqlite-vec" tries the extension and
        # transparently falls back to numpy if the probe failed.
        self.vector_backend = vector_backend
        # Persist the built matrix next to the DB so a one-shot CLI run does
        # not rebuild it. Off restores the pre-cache behaviour exactly, and
        # also removes any file a previous run left behind -- the matrix is
        # the size of every vector in the vault, so leaving it orphaned when
        # the feature is switched off would be a surprising amount of disk.
        #
        # Deliberately not cleared by `maintenance run`: VACUUM preserves the
        # ids of an INTEGER PRIMARY KEY table, so the seal still matches
        # afterwards and dropping the cache would only force a needless
        # rebuild. The one thing that does invalidate it silently is the
        # embedding-model swap, which clears it itself. Reclaiming the file
        # when the feature is switched off is discard_matrix_cache()'s job,
        # called from the CLI -- constructing a Connector deletes nothing.
        self.matrix_cache_enabled = matrix_cache_enabled
        # Matrix cache for the warm shell session — see _load_dense().
        self._cache_sig: Optional[tuple] = None
        self._cache_keys: Optional[list] = None
        self._cache_blobs: Optional[list] = None
        self._cache_rows_by_note: dict = {}
        self._cache_matrix = None
        # Second-stage re-ranker. ``reranker`` (explicit injection) is
        # for tests / advanced wiring; otherwise build one from the
        # engine name, falling back to LLM when cross-encoder extras
        # aren't installed. ``None`` means re-rank is silently a no-op
        # — keeps find_hybrid simple when there's no router and no extra.
        if reranker is not None:
            self._reranker: Optional[Reranker] = reranker
        else:
            self._reranker = build_reranker(rerank_engine, router, model_name=rerank_model)

    def discard_matrix_cache(self) -> None:
        """Delete the on-disk matrix cache and forget the in-memory one.

        Housekeeping, called explicitly -- notably by ``maintenance run`` when
        the feature is switched off, so the file (the size of every vector in
        the vault) does not sit there orphaned. Deliberately not done in
        ``__init__``: constructing a Connector should not delete anything,
        least of all a file another process may be reading.
        """
        matrix_cache.clear(self.db.db_path)
        self._cache_sig = None

    def _use_vec_backend(self) -> bool:
        """Whether this call should route through ``db.vec_search`` instead
        of building the in-memory matmul matrix."""
        if self.vector_backend == "numpy":
            return False
        return self.db.vec_available

    def _load_dense(self):
        """Return ``(keys, matrix, blobs)`` for dense scoring, cached per query.

        ``keys`` are ``(embedding_id, note_id)`` pairs aligned positionally
        with the matrix rows. ``matrix`` is the ``(N, D)`` numpy matrix, or
        ``None`` when numpy is absent or the vectors are ragged. ``blobs`` is
        the per-row vector list, and is populated **only** in that fallback
        case -- when the matrix exists it stays ``None`` so the raw bytes can
        be freed rather than held alongside a copy of themselves.

        Neither the text nor (normally) the blobs are kept: the text belongs to
        the few rows that win the ranking, and the caller fetches those with
        :py:meth:`Database.get_chunk_texts`.

        The cache is keyed on the DB's cheap embeddings signature so a
        long-lived shell ``Session`` rebuilds only when the vault changes.
        """
        # The cheap signature plus the DB's data generation. COUNT/MAX cannot
        # see an embedding-model swap (it re-inserts under the original ids),
        # and folding the vector byte total in would make every query a full
        # scan; the generation is an in-memory int the swap bumps, so it costs
        # nothing and closes that hole for anyone sharing this Database.
        sig = (self.db.embeddings_signature(),
               getattr(self.db, "data_generation", 0))
        if sig != self._cache_sig:
            keys, matrix, blobs = self._build_dense()
            self._cache_keys = keys
            self._cache_matrix = matrix
            self._cache_blobs = blobs
            # note_ids as an array so a filter mask is one vectorised call
            # instead of a Python loop over every row -- the loop costs more
            # than the matmul it is supposed to be narrowing.
            # note_id -> row indices, built once per matrix. A filter then
            # costs one dict lookup per allowed note instead of a scan over
            # every row, and lets the scoring skip the excluded rows entirely
            # rather than scoring them and throwing the result away.
            rows_by_note: dict = {}
            for row, (_eid, note_id) in enumerate(keys):
                rows_by_note.setdefault(note_id, []).append(row)
            self._cache_rows_by_note = {
                n: (_np.asarray(r, dtype=_np.int64) if _np is not None else r)
                for n, r in rows_by_note.items()
            }
            self._cache_sig = sig
        return self._cache_keys, self._cache_matrix, self._cache_blobs

    def _build_dense(self):
        """Produce ``(keys, matrix, blobs)`` from the disk cache or SQLite.

        On a cache hit only the keys are read from the database and the matrix
        arrives memory-mapped, so the vectors never pass through Python at all.
        On a miss the matrix is built from the vector buffer and written back
        for the next process.
        """
        seal_before = None
        if self.matrix_cache_enabled:
            seal_before = self.db.matrix_cache_signature()
            count, _max_id, total_bytes = seal_before
            # dim comes from the seal alone. That is exact for the uniform
            # vectors the cache stores, but it cannot *prove* uniformity:
            # lengths 4 and 12 seal identically to two 8-byte rows. Ragged
            # data therefore falls under the same limitation as any
            # same-shape rewrite -- see matrix_cache.clear.
            dim = (total_bytes // count // 4) if count else 0
            if count and dim:
                matrix = matrix_cache.load(self.db.db_path, seal_before, count, dim)
                if matrix is not None:
                    keys = self.db.get_embedding_keys()
                    # The matrix and the keys are two independent reads; a
                    # writer landing between them would leave the rows
                    # misaligned, which silently attributes a score to the
                    # wrong note (or indexes past the end). Length disagreement
                    # is the observable symptom, so treat it as a miss.
                    if len(keys) == matrix.shape[0]:
                        return keys, matrix, None
                    logger.warning(
                        "matrix_cache_keys_mismatch",
                        keys=len(keys), rows=int(matrix.shape[0]),
                    )

        keys, blob, dim = self.db.get_embedding_matrix_parts()
        matrix = Embedder.buffer_to_matrix(blob, len(keys), dim)
        # The per-row list costs as much as the matrix, so it is only
        # fetched when there is no matrix to score against.
        blobs = None
        if matrix is None and keys:
            blobs = self.db.get_embedding_vectors()
        elif matrix is not None and self.matrix_cache_enabled:
            # Skip the write when the table moved under us mid-build (the
            # daemon indexing while a query runs). This is an efficiency
            # guard, not a safety one: such a file would be sealed with a
            # signature no future reader can match, so it would only ever
            # cost a write and then miss. Safety against a seal that matches
            # the wrong contents comes from the shape check in
            # matrix_cache.load, and -- for a same-shape rewrite, which no
            # signature can see -- from matrix_cache.clear.
            if self.db.matrix_cache_signature() == seal_before:
                matrix_cache.save(self.db.db_path, matrix, seal_before)
        return keys, matrix, blobs

    def _scores_for(self, query_vector, matrix, blobs) -> list[float]:
        """Cosine score for every row, aligned to the matrix / blob order.

        Vectors are unit-normalized at embed time, so cosine == dot product.
        Fast path: one ``matrix @ query`` matmul. Fallback (no numpy, ragged
        vectors, or a query/matrix dimension mismatch): per-row Python dot
        product, identical to the pre-numpy behaviour.
        """
        if matrix is not None and _np is not None and matrix.size:
            q = _np.asarray(query_vector, dtype=_np.float32)
            if q.shape[0] == matrix.shape[1]:
                return (matrix @ q).tolist()
        # A dimension mismatch lands here with a matrix but no blobs, since
        # those are only fetched when the matrix itself is missing. Cache the
        # read: without it every query re-runs a full SELECT over the vector
        # column, which is what the pre-split code avoided by keeping the rows.
        if not blobs:
            if self._cache_blobs is None:
                self._cache_blobs = self.db.get_embedding_vectors()
            blobs = self._cache_blobs
        return [
            Embedder.dot_product(query_vector, Embedder.deserialize_vector(b))
            for b in blobs
        ]

    def _candidate_scores(self, query_vector, keys, matrix, blobs,
                          filter_note_ids):
        """Return ``(rows, scores)`` for the candidates worth ranking.

        ``rows`` is ``None`` when every row is a candidate, and ``scores`` is
        then aligned to ``keys``. With a filter, ``rows`` holds the indices of
        the allowed rows and ``scores`` covers only those, so the caller maps
        position ``p`` back with ``rows[p]``.

        Scoring the subset rather than scoring everything and masking is the
        difference between a filter that costs and one that pays: masking
        still multiplies every vector in the vault, then throws most of the
        result away. Measured on 18k chunks with a filter selecting 5% of the
        notes, masking ran 48.8% slower than no filter at all; restricting the
        matrix first runs faster than no filter, which is what a narrowing
        feature ought to do.
        """
        if filter_note_ids is None:
            return None, self._scores_for(query_vector, matrix, blobs)

        by_note = self._cache_rows_by_note or {}
        present = [by_note[n] for n in filter_note_ids if n in by_note]
        if not present:
            return [], []
        if _np is not None and matrix is not None:
            rows = _np.sort(_np.concatenate(present))
            q = _np.asarray(query_vector, dtype=_np.float32)
            if q.shape[0] == matrix.shape[1]:
                # Which strategy is cheaper depends on how much the filter
                # actually removes. Restricting copies the selected rows out
                # of the matrix, so a filter that keeps almost everything pays
                # to duplicate almost the whole matrix -- measured at nearly
                # 5x the unfiltered time when 90% of notes pass. Above the
                # threshold it is cheaper to multiply the contiguous matrix
                # once and blank the few rows that lost.
                if len(rows) > len(keys) * self._RESTRICT_MAX_FRACTION:
                    scores = _np.asarray(matrix @ q, dtype=_np.float32)
                    keep = _np.zeros(len(keys), dtype=bool)
                    keep[rows] = True
                    scores[~keep] = -_np.inf
                    return None, scores
                return rows, matrix[rows] @ q
        # No matrix (or a width mismatch): fall back to the per-row path over
        # the allowed rows only.
        rows = sorted(int(r) for arr in present for r in arr)
        if not blobs:
            blobs = self.db.get_embedding_vectors()
        return rows, [
            Embedder.dot_product(query_vector,
                                 Embedder.deserialize_vector(blobs[r]))
            for r in rows
        ]

    @staticmethod
    def _topk_indices(scores, k: int) -> list[int]:
        """Indices of the top ``k`` scores, descending. Ties broken by index.

        Uses ``np.argpartition`` when numpy is available and worth it
        (``len(scores) > k > 0``): partition is O(N) and the follow-up sort
        only touches ``k`` items, so the total drops from O(N log N) (full
        Python sort) to O(N + k log k). Falls back to a Python sort when
        numpy is missing or ``k`` already covers the whole list.

        We negate the scores so ``argpartition``/``argsort`` give the
        *largest* values first; with ``kind="stable"`` ties resolve by
        original index, matching the Python sort baseline.
        """
        n = len(scores)
        if n == 0 or k <= 0:
            return []
        if _np is not None and n > k:
            arr = _np.asarray(scores, dtype=_np.float32)
            part = _np.argpartition(-arr, k - 1)[:k]
            # argpartition returns the k picks in arbitrary order, so sort them
            # by ascending index first; the stable argsort by descending score
            # then resolves score ties by original index — exactly matching the
            # Python ``sorted(..., reverse=True)`` fallback (Python's sort is
            # stable, so reverse keeps equal keys in ascending-index order).
            part.sort()
            return part[_np.argsort(-arr[part], kind="stable")].tolist()
        return sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]

    def find_similar_notes(
        self,
        query_vector: List[float],
        top_k: int = 5,
        exclude_note_id: int = None,
        dedupe_by_note: bool = False,
        with_text: bool = True,
        filter_note_ids: "set[int] | None" = None,
    ):
        """
        Finds the top_k most similar chunks in the database compared to a query vector.

        Note: Stored and query vectors are unit-normalized by the embedder,
        so cosine similarity simplifies to a basic dot product.

        When ``dedupe_by_note`` is True, only the best-scoring chunk per note
        is kept — useful for the ``connect`` pass that needs distinct notes
        rather than chunks. Oracle-style RAG keeps it False so multiple
        chunks of the same note can all feed the context window.

        ``with_text=False`` returns ``""`` for every ``text`` and skips the
        lookup that fills it. Chunk text no longer rides along with the scoring
        matrix, so it costs one query per call — which the callers that only
        read ``note_id`` and ``score`` (``connect``, the graph's suggested
        edges) would otherwise pay once per note across the whole vault.
        """
        query = list(query_vector)

        # sqlite-vec fast path: let SQLite do the ranking and skip the
        # all-vectors load entirely. Same oversample math as the numpy path
        # so post-filters keep the final set behaviour-identical.
        if filter_note_ids is not None and not filter_note_ids:
            return []

        if self._use_vec_backend():
            needed = top_k + (1 if exclude_note_id is not None else 0)
            if dedupe_by_note:
                needed = max(needed * 5, needed + 10)
            # sqlite-vec ranks inside SQLite with no predicate hook, so the
            # filter is a post-pass over an oversampled page here.
            if filter_note_ids is not None:
                needed = max(needed * 4, needed + 20)
            hits = self.db.vec_search(query, needed, exclude_note_id=exclude_note_id)
            # with_text is honoured here too: the flag is a contract about the
            # returned shape, not an artefact of how the numpy path fetches.
            similarities = [
                {"note_id": nid, "text": text if with_text else "", "score": score}
                for _eid, nid, text, score in hits
                if filter_note_ids is None or nid in filter_note_ids
            ]
            if dedupe_by_note:
                seen: set[int] = set()
                unique: list[dict] = []
                for item in similarities:
                    if item["note_id"] in seen:
                        continue
                    seen.add(item["note_id"])
                    unique.append(item)
                similarities = unique
            return similarities[:top_k]

        keys, matrix, blobs = self._load_dense()
        rows, scores = self._candidate_scores(
            query, keys, matrix, blobs, filter_note_ids)
        if len(scores) == 0:
            return []

        # Oversample headroom for the post-filters: ``exclude_note_id`` may
        # drop one of the picks, and ``dedupe_by_note`` collapses repeated
        # note_ids — both happen *after* top-k, so we need a wider window
        # going in. The 5× + 10 floor for dedupe is generous enough that
        # any realistic vault keeps the same final set as the old full-sort
        # path while still being O(N) instead of O(N log N).
        needed = top_k + (1 if exclude_note_id is not None else 0)
        if dedupe_by_note:
            needed = max(needed * 5, needed + 10)
        top_idx = self._topk_indices(scores, needed)

        # Text for the winners only -- one query instead of carrying every
        # chunk's 500 chars through the scoring pass.
        # top_idx indexes `scores`; with a filter that is the subset, so the
        # key it refers to is rows[p].
        picks = [(keys[rows[p]][0] if rows is not None else keys[p][0],
                  keys[rows[p]][1] if rows is not None else keys[p][1],
                  float(scores[p]))
                 for p in top_idx]
        texts = (
            self.db.get_chunk_texts([eid for eid, _nid, _s in picks])
            if with_text else {}
        )
        return self._hits_from_picks(picks, texts, exclude_note_id,
                                     dedupe_by_note, top_k)

    @staticmethod
    def _hits_from_picks(picks, texts, exclude_note_id,
                         dedupe_by_note, top_k) -> list[dict]:
        """Turn ranked ``(embedding_id, note_id, score)`` picks into results.

        Picks carry their score rather than an index into the score array, so
        nothing here holds a reference to that array. That matters for the
        batched path: a numpy row obtained by iterating a block is a *view*
        whose base is the whole block, so keeping one row per query would pin
        every block until the sweep ended and make the blocking pointless.

        Shared by the single-query and batched paths so they cannot drift.
        The order of the three steps is the behaviour, not an implementation
        detail: the self-note filter and the per-note dedup both run *after*
        the top-k cut, so a note whose best chunk sits outside the oversampled
        window is absent even if it would place well among notes. Reproducing
        that exactly is what makes the batched sweep a drop-in replacement.
        """
        hits: list[dict] = []
        for emb_id, note_id, score in picks:
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            hits.append(
                {"note_id": note_id, "text": texts.get(emb_id, ""),
                 "score": score}
            )

        if dedupe_by_note:
            seen: set[int] = set()
            unique: list[dict] = []
            for item in hits:
                if item["note_id"] in seen:
                    continue
                seen.add(item["note_id"])
                unique.append(item)
            hits = unique

        return hits[:top_k]

    # Scores for one block of queries against every chunk. 64 MB of float32
    # is a few hundred rows on a realistic vault -- big enough that the whole
    # sweep is one or two passes over the matrix, small enough that the score
    # buffer never approaches the size of the matrix itself.
    _BLOCK_TARGET_BYTES = 64 * 1024 * 1024

    # Above this fraction of rows surviving, restricting the matrix costs
    # more than scoring it whole and blanking the losers. Measured
    # crossover on 18k chunks sits between 25% and 90%; 0.5 is inside
    # the flat part of that curve at both ends.
    _RESTRICT_MAX_FRACTION = 0.5

    def find_similar_notes_batch(
        self,
        query_vectors,
        *,
        top_k: int = 5,
        exclude_note_ids=None,
        dedupe_by_note: bool = False,
        with_text: bool = True,
        block_rows: Optional[int] = None,
    ) -> list[list[dict]]:
        """Run many queries in one blocked matmul instead of one matmul each.

        Returns one result list per query, each identical to what
        :py:meth:`find_similar_notes` would return for that query -- the two
        share the ranking helper and the assembly step, so parity is
        structural rather than a coincidence to be re-checked.

        ``connect`` used to call the single-query method once per note, and
        each of those calls multiplied the query against every chunk in the
        vault: O(notes x chunks) products issued one at a time. Here the whole
        sweep is ``Q @ C.T``, which BLAS does in far fewer, far larger
        operations.

        Blocked from the start rather than as a later refinement: the full
        score matrix is ``len(queries) x chunks`` floats, which for a big vault
        is larger than the embeddings themselves. ``block_rows`` defaults to
        whatever keeps one block near
        :py:attr:`_BLOCK_TARGET_BYTES`, and is exposed so tests can force
        several blocks on a small fixture.

        Falls back to a plain loop when there is no matrix to multiply (no
        numpy, ragged vectors) or when the sqlite-vec backend is active, since
        that path ranks inside SQLite and has no batch form. Correctness is
        unaffected either way; only the speed-up is.
        """
        queries = list(query_vectors)
        excludes = list(exclude_note_ids) if exclude_note_ids is not None \
            else [None] * len(queries)
        if len(excludes) != len(queries):
            raise ValueError("exclude_note_ids must align with query_vectors")
        if not queries:
            return []

        def _per_query():
            return [
                self.find_similar_notes(
                    q, top_k=top_k, exclude_note_id=ex,
                    dedupe_by_note=dedupe_by_note, with_text=with_text,
                )
                for q, ex in zip(queries, excludes, strict=True)
            ]

        keys, matrix, _blobs = self._load_dense()
        if self._use_vec_backend() or matrix is None or _np is None:
            return _per_query()

        chunks, dim = matrix.shape
        if block_rows is None:
            block_rows = max(1, self._BLOCK_TARGET_BYTES // max(chunks * 4, 1))

        # Two passes so the text for every winner across the whole batch is
        # one query: collect the picks first, resolve text once, assemble after.
        per_query: list[tuple[list, list]] = []
        for start in range(0, len(queries), block_rows):
            block = queries[start:start + block_rows]
            q_block = _np.asarray(block, dtype=_np.float32)
            if q_block.ndim != 2 or q_block.shape[1] != dim:
                # A query of the wrong width cannot be scored against this
                # matrix; defer the whole batch to the per-query path, which
                # already handles the mismatch by scoring row-wise.
                return _per_query()
            block_scores = q_block @ matrix.T
            for offset, row in enumerate(block_scores):
                # The oversample window is computed per query, exactly as the
                # single-query path does: it widens by one only when *that*
                # query excludes a note. Hoisting it out of the loop would
                # diverge for a batch that mixes excluded and plain queries.
                exclude = excludes[start + offset]
                needed = top_k + (1 if exclude is not None else 0)
                if dedupe_by_note:
                    needed = max(needed * 5, needed + 10)
                top_idx = self._topk_indices(row, needed)
                # float(row[i]) copies the value out; keeping `row` itself
                # would keep the whole block alive (see _hits_from_picks).
                per_query.append(
                    [(keys[i][0], keys[i][1], float(row[i])) for i in top_idx]
                )

        texts = {}
        if with_text:
            wanted = {eid for picks in per_query for eid, _n, _s in picks}
            texts = self.db.get_chunk_texts(wanted)

        return [
            self._hits_from_picks(picks, texts, exclude, dedupe_by_note, top_k)
            for picks, exclude in zip(per_query, excludes, strict=True)
        ]

    def _vector_candidates(
        self,
        query_vector: List[float],
        limit: int,
        exclude_note_id: Optional[int] = None,
        filter_note_ids: "set[int] | None" = None,
    ) -> list[dict]:
        """Return ranked dense-similarity candidates keyed by embedding id."""
        if filter_note_ids is not None and not filter_note_ids:
            return []
        query = list(query_vector)

        if self._use_vec_backend():
            needed = limit + (1 if exclude_note_id is not None else 0)
            if filter_note_ids is not None:
                needed = max(needed * 4, needed + 20)
            hits = self.db.vec_search(query, needed, exclude_note_id=exclude_note_id)
            return [
                {
                    "embedding_id": eid,
                    "note_id": nid,
                    "text": text,
                    "score": score,
                }
                for eid, nid, text, score in hits
                if filter_note_ids is None or nid in filter_note_ids
            ][:limit]

        keys, matrix, blobs = self._load_dense()
        rows, scores = self._candidate_scores(
            query, keys, matrix, blobs, filter_note_ids)
        if len(scores) == 0:
            return []

        # Same +1 headroom as find_similar_notes: the excluded row could be
        # one of the picks. No dedupe here (the fusion pass keys on
        # embedding_id, not note_id), so a single extra slot is enough.
        needed = limit + (1 if exclude_note_id is not None else 0)
        top_idx = self._topk_indices(scores, needed)

        picks = [(keys[rows[p]][0] if rows is not None else keys[p][0],
                  keys[rows[p]][1] if rows is not None else keys[p][1], p)
                 for p in top_idx]
        texts = self.db.get_chunk_texts([eid for eid, _nid, _i in picks])

        scored: list[dict] = []
        for embedding_id, note_id, i in picks:
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            scored.append({
                "embedding_id": embedding_id,
                "note_id": note_id,
                "text": texts.get(embedding_id, ""),
                "score": float(scores[i]),
            })
        return scored[:limit]

    def find_hybrid(
        self,
        query_text: str,
        query_vector: Optional[List[float]],
        top_k: int = 5,
        rrf_k: int = 60,
        exclude_note_id: Optional[int] = None,
        rerank: bool = False,
        rerank_pool: int = 20,
        timings: Optional[dict] = None,
        filter_note_ids: "set[int] | None" = None,
    ) -> list[dict]:
        """
        Fuse dense retrieval and FTS5 BM25 with Reciprocal Rank Fusion.

        RRF is rank-based so score ranges don't need to match — each candidate
        gets ``Σ 1 / (rrf_k + rank_in_list)`` across the two rankings. A doc
        that appears in both lists beats one that only appears in one, which
        is the whole point of hybrid search.

        Degrades gracefully:

        * No vector (embedder failed) → BM25-only.
        * No FTS5 or no BM25 hits     → vector-only.

        Observability: emits a debug ``rrf_rank_inputs`` log with each
        surviving doc's dense- and BM25-rank (the actual fusion inputs), and,
        when ``timings`` is supplied, records the second-stage rerank duration
        under ``timings["rerank_s"]`` so eval can bucket latency by stage.
        """
        if filter_note_ids is not None and not filter_note_ids:
            return []

        pool = max(top_k * 4, 20)

        dense: list[dict] = []
        if query_vector:
            dense = self._vector_candidates(
                query_vector, limit=pool, exclude_note_id=exclude_note_id,
                filter_note_ids=filter_note_ids,
            )

        sparse_rows = self.db.fts_search(
            query_text, limit=pool, filter_note_ids=filter_note_ids,
        ) if query_text else []
        sparse: list[dict] = [
            {"embedding_id": eid, "note_id": nid, "text": text, "bm25": bm25}
            for eid, nid, text, bm25 in sparse_rows
            if exclude_note_id is None or nid != exclude_note_id
        ]

        if not dense and not sparse:
            return []
        # ``embedding_id`` is retained through fusion so per-doc rank inputs can
        # be logged, then stripped just before returning.
        if not sparse:
            # BM25 contributed nothing — behave exactly like the dense path.
            ranked = [dict(item) for item in dense]
        elif not dense:
            ranked = [
                {"embedding_id": s["embedding_id"], "note_id": s["note_id"],
                 "text": s["text"], "score": -s["bm25"]}
                for s in sparse
            ]
        else:
            ranks: dict[int, dict] = {}
            for rank, item in enumerate(dense):
                ranks.setdefault(item["embedding_id"], {
                    "embedding_id": item["embedding_id"],
                    "note_id": item["note_id"],
                    "text": item["text"],
                    "rrf": 0.0,
                })["rrf"] += 1.0 / (rrf_k + rank + 1)
            for rank, item in enumerate(sparse):
                ranks.setdefault(item["embedding_id"], {
                    "embedding_id": item["embedding_id"],
                    "note_id": item["note_id"],
                    "text": item["text"],
                    "rrf": 0.0,
                })["rrf"] += 1.0 / (rrf_k + rank + 1)

            ranked = [
                {"embedding_id": v["embedding_id"], "note_id": v["note_id"],
                 "text": v["text"], "score": v["rrf"]}
                for v in ranks.values()
            ]
            ranked.sort(key=lambda x: x["score"], reverse=True)

        # Optional second-stage re-rank over the head of the pool. Falls
        # back to the fusion order on any failure (handled inside _rerank).
        # Timed separately so a slow reranker doesn't hide inside "retrieve".
        if rerank and self._reranker is not None and len(ranked) > 1:
            t_rerank = time.perf_counter()
            ranked = self._rerank(query_text, ranked, rerank_pool)
            if timings is not None:
                timings["rerank_s"] = time.perf_counter() - t_rerank

        survivors = ranked[:top_k]
        self._log_rrf_inputs(dense, sparse, survivors)
        # Drop the internal embedding_id — callers key on note_id/text/score.
        return [
            {k: v for k, v in item.items() if k != "embedding_id"}
            for item in survivors
        ]

    @staticmethod
    def _log_rrf_inputs(
        dense: list[dict], sparse: list[dict], survivors: list[dict]
    ) -> None:
        """Debug-log the RRF rank inputs for the docs that survived to the result.

        For each returned doc, records its 1-indexed rank in the dense list and
        in the BM25 list (``None`` when that signal didn't surface it) — the raw
        inputs the fusion combined. Makes it visible whether hybrid is genuinely
        fusing two signals or one is carrying the result, which is the first
        thing you want to know when a query ranks worse than expected. Silent in
        normal use; turn logging up during eval to see it.
        """
        if not survivors:
            return
        dense_rank = {d["embedding_id"]: i + 1 for i, d in enumerate(dense)}
        bm25_rank = {s["embedding_id"]: i + 1 for i, s in enumerate(sparse)}
        inputs = [
            {
                "note_id": item.get("note_id"),
                "dense_rank": dense_rank.get(item.get("embedding_id")),
                "bm25_rank": bm25_rank.get(item.get("embedding_id")),
            }
            for item in survivors
        ]
        logger.debug(
            "rrf_rank_inputs",
            dense_pool=len(dense),
            bm25_pool=len(sparse),
            inputs=inputs,
        )

    def _rerank(
        self, query_text: str, candidates: list[dict], pool: int
    ) -> list[dict]:
        """Reorder the top ``pool`` candidates by reranker-judged relevance.

        The active backend (set at construction by ``rerank_engine``)
        scores each head passage; the head is re-sorted by that score
        and the tail (beyond ``pool``) is appended unchanged. Returns
        ``candidates`` untouched on any failure — no reranker, fewer
        than 2 head items, an empty score list — so re-rank is strictly
        best-effort.
        """
        head = candidates[: max(pool, 0)]
        tail = candidates[max(pool, 0):]
        if len(head) < 2 or self._reranker is None:
            return candidates

        passages = [(c.get("text") or "") for c in head]
        scores = self._reranker.score(query_text, passages)
        if not scores or len(scores) != len(head):
            return candidates

        # Stable sort: ties keep the original fusion order, and unscored
        # entries (encoded as -inf by LLMReranker) sink below real scores.
        order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        logger.info(
            "rerank_applied",
            engine=type(self._reranker).__name__,
            pool=len(head),
            scored=sum(1 for s in scores if s != float("-inf")),
        )
        return [head[i] for i in order] + tail
