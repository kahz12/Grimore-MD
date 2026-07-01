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
    ):
        self.db = db
        self.embedder = embedder
        self.router = router
        # "auto" picks sqlite-vec when the extension + table are ready, else
        # numpy. "numpy" pins the matmul path even with sqlite-vec installed
        # (handy for parity tests). "sqlite-vec" tries the extension and
        # transparently falls back to numpy if the probe failed.
        self.vector_backend = vector_backend
        # Matrix cache for the warm shell session — see _load_dense().
        self._cache_sig: Optional[tuple[int, int]] = None
        self._cache_rows: Optional[list] = None
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

    def _use_vec_backend(self) -> bool:
        """Whether this call should route through ``db.vec_search`` instead
        of building the in-memory matmul matrix."""
        if self.vector_backend == "numpy":
            return False
        return self.db.vec_available

    def _load_dense(self):
        """Return ``(rows, matrix)`` for dense scoring, cached across queries.

        ``rows`` are ``(embedding_id, note_id, text, vector_blob)`` tuples;
        ``matrix`` is the aligned ``(N, D)`` numpy matrix (or ``None`` when
        numpy is absent / vectors are ragged, in which case the caller scores
        per-row). The cache is keyed on the DB's cheap embeddings signature so
        a long-lived shell ``Session`` rebuilds only when the vault changes.
        """
        sig = self.db.embeddings_signature()
        if sig != self._cache_sig:
            rows = self.db.get_all_embeddings_with_id()
            self._cache_rows = rows
            self._cache_matrix = Embedder.vectors_to_matrix([r[3] for r in rows])
            self._cache_sig = sig
        return self._cache_rows, self._cache_matrix

    def _scores_for(self, query_vector, rows, matrix) -> list[float]:
        """Cosine score for every row, aligned to ``rows``.

        Vectors are unit-normalized at embed time, so cosine == dot product.
        Fast path: one ``matrix @ query`` matmul. Fallback (no numpy, ragged
        vectors, or a query/matrix dimension mismatch): per-row Python dot
        product, identical to the pre-numpy behaviour.
        """
        if matrix is not None and _np is not None and matrix.size:
            q = _np.asarray(query_vector, dtype=_np.float32)
            if q.shape[0] == matrix.shape[1]:
                return (matrix @ q).tolist()
        return [
            Embedder.dot_product(query_vector, Embedder.deserialize_vector(r[3]))
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
    ):
        """
        Finds the top_k most similar chunks in the database compared to a query vector.

        Note: Stored and query vectors are unit-normalized by the embedder,
        so cosine similarity simplifies to a basic dot product.

        When ``dedupe_by_note`` is True, only the best-scoring chunk per note
        is kept — useful for the ``connect`` pass that needs distinct notes
        rather than chunks. Oracle-style RAG keeps it False so multiple
        chunks of the same note can all feed the context window.
        """
        query = list(query_vector)

        # sqlite-vec fast path: let SQLite do the ranking and skip the
        # all-vectors load entirely. Same oversample math as the numpy path
        # so post-filters keep the final set behaviour-identical.
        if self._use_vec_backend():
            needed = top_k + (1 if exclude_note_id is not None else 0)
            if dedupe_by_note:
                needed = max(needed * 5, needed + 10)
            hits = self.db.vec_search(query, needed, exclude_note_id=exclude_note_id)
            similarities = [
                {"note_id": nid, "text": text, "score": score}
                for _eid, nid, text, score in hits
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

        rows, matrix = self._load_dense()
        scores = self._scores_for(query, rows, matrix)
        if not scores:
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

        similarities: list[dict] = []
        for i in top_idx:
            _emb_id, note_id, text, _blob = rows[i]
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            similarities.append(
                {"note_id": note_id, "text": text, "score": float(scores[i])}
            )

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

    def _vector_candidates(
        self,
        query_vector: List[float],
        limit: int,
        exclude_note_id: Optional[int] = None,
    ) -> list[dict]:
        """Return ranked dense-similarity candidates keyed by embedding id."""
        query = list(query_vector)

        if self._use_vec_backend():
            needed = limit + (1 if exclude_note_id is not None else 0)
            hits = self.db.vec_search(query, needed, exclude_note_id=exclude_note_id)
            return [
                {
                    "embedding_id": eid,
                    "note_id": nid,
                    "text": text,
                    "score": score,
                }
                for eid, nid, text, score in hits
            ][:limit]

        rows, matrix = self._load_dense()
        scores = self._scores_for(query, rows, matrix)
        if not scores:
            return []

        # Same +1 headroom as find_similar_notes: the excluded row could be
        # one of the picks. No dedupe here (the fusion pass keys on
        # embedding_id, not note_id), so a single extra slot is enough.
        needed = limit + (1 if exclude_note_id is not None else 0)
        top_idx = self._topk_indices(scores, needed)

        scored: list[dict] = []
        for i in top_idx:
            embedding_id, note_id, text, _blob = rows[i]
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            scored.append({
                "embedding_id": embedding_id,
                "note_id": note_id,
                "text": text,
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
        pool = max(top_k * 4, 20)

        dense: list[dict] = []
        if query_vector:
            dense = self._vector_candidates(
                query_vector, limit=pool, exclude_note_id=exclude_note_id
            )

        sparse_rows = self.db.fts_search(query_text, limit=pool) if query_text else []
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
