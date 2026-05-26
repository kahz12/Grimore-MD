"""
Semantic Connection Discovery.
This module provides logic to find related notes by comparing their vector
embeddings and, optionally, fusing vector ranking with BM25 (FTS5) ranking
via Reciprocal Rank Fusion.
"""
from typing import List, Optional

from grimore.cognition.embedder import Embedder
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
    def __init__(self, db: Database, embedder: Embedder, router=None):
        self.db = db
        self.embedder = embedder
        self.router = router
        # Matrix cache for the warm shell session — see _load_dense().
        self._cache_sig: Optional[tuple[int, int]] = None
        self._cache_rows: Optional[list] = None
        self._cache_matrix = None

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
        rows, matrix = self._load_dense()
        scores = self._scores_for(query, rows, matrix)
        similarities = []

        for (_emb_id, note_id, text, _blob), score in zip(rows, scores):
            # Skip self-comparison
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            similarities.append(
                {"note_id": note_id, "text": text, "score": float(score)}
            )

        # Sort results by similarity score in descending order
        similarities.sort(key=lambda x: x["score"], reverse=True)

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
        rows, matrix = self._load_dense()
        scores = self._scores_for(query, rows, matrix)

        scored: list[dict] = []
        for (embedding_id, note_id, text, _blob), score in zip(rows, scores):
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            scored.append({
                "embedding_id": embedding_id,
                "note_id": note_id,
                "text": text,
                "score": float(score),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
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
        if not sparse:
            # BM25 contributed nothing — behave exactly like the dense path.
            ranked = [
                {k: v for k, v in item.items() if k != "embedding_id"}
                for item in dense
            ]
        elif not dense:
            ranked = [
                {"note_id": s["note_id"], "text": s["text"], "score": -s["bm25"]}
                for s in sparse
            ]
        else:
            ranks: dict[int, dict] = {}
            for rank, item in enumerate(dense):
                ranks.setdefault(item["embedding_id"], {
                    "note_id": item["note_id"],
                    "text": item["text"],
                    "rrf": 0.0,
                })["rrf"] += 1.0 / (rrf_k + rank + 1)
            for rank, item in enumerate(sparse):
                ranks.setdefault(item["embedding_id"], {
                    "note_id": item["note_id"],
                    "text": item["text"],
                    "rrf": 0.0,
                })["rrf"] += 1.0 / (rrf_k + rank + 1)

            ranked = [
                {"note_id": v["note_id"], "text": v["text"], "score": v["rrf"]}
                for v in ranks.values()
            ]
            ranked.sort(key=lambda x: x["score"], reverse=True)

        # Optional second-stage LLM re-rank over the head of the pool. Falls
        # back to the fusion order on any failure (handled inside _llm_rerank).
        if rerank and self.router is not None and len(ranked) > 1:
            ranked = self._llm_rerank(query_text, ranked, rerank_pool)

        return ranked[:top_k]

    def _llm_rerank(
        self, query_text: str, candidates: list[dict], pool: int
    ) -> list[dict]:
        """Reorder the top ``pool`` candidates by LLM-judged relevance.

        One batched ``router.complete`` call asks the local model to rate each
        candidate 0–10 for the query; the head is re-sorted by that score and
        the tail (beyond ``pool``) is appended unchanged. Returns ``candidates``
        untouched on any failure — unreachable model, circuit open, unparseable
        JSON, or no usable scores — so re-rank is strictly best-effort.
        """
        head = candidates[: max(pool, 0)]
        tail = candidates[max(pool, 0):]
        if len(head) < 2:
            return candidates

        listing = "\n".join(
            f"[{i}] {(c.get('text') or '')[:300]}" for i, c in enumerate(head)
        )
        prompt = (
            f"Question: {query_text}\n\n"
            f"Passages:\n{listing}\n\n"
            "Rate how relevant each passage is to answering the question, on a "
            "0-10 scale.\n"
            'Return ONLY JSON: {"scores": [{"index": <int>, "score": <number>}, ...]}'
        )
        try:
            resp = self.router.complete(
                prompt=prompt,
                system_prompt="You rate passage relevance for retrieval re-ranking.",
                json_format=True,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("rerank_failed", error=str(e))
            return candidates

        if not isinstance(resp, dict) or not isinstance(resp.get("scores"), list):
            return candidates
        score_by_idx: dict[int, float] = {}
        for entry in resp["scores"]:
            if isinstance(entry, dict) and isinstance(entry.get("index"), int):
                try:
                    score_by_idx[entry["index"]] = float(entry.get("score", 0))
                except (TypeError, ValueError):
                    continue
        if not score_by_idx:
            return candidates

        # Stable sort keeps the original fusion order among ties / unscored
        # items (which default below the lowest real score).
        order = sorted(
            range(len(head)),
            key=lambda i: score_by_idx.get(i, float("-inf")),
            reverse=True,
        )
        logger.info("rerank_applied", pool=len(head), scored=len(score_by_idx))
        return [head[i] for i in order] + tail
