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

logger = get_logger(__name__)


class Connector:
    """
    Handles the discovery of relationships between notes.

    Two retrieval strategies are exposed:

    * :py:meth:`find_similar_notes` — dense only (cosine similarity).
    * :py:meth:`find_hybrid` — dense + BM25 fused with Reciprocal Rank Fusion.
    """
    def __init__(self, db: Database, embedder: Embedder):
        self.db = db
        self.embedder = embedder

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
        # Fetch all indexed embeddings for comparison
        all_embeddings = self.db.get_all_embeddings()
        similarities = []

        for note_id, text, vector_blob in all_embeddings:
            # Skip self-comparison
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue

            vector = self.embedder.deserialize_vector(vector_blob)
            score = Embedder.dot_product(query, vector)
            similarities.append({"note_id": note_id, "text": text, "score": score})

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
        rows = self.db.get_all_embeddings_with_id()

        scored: list[dict] = []
        for embedding_id, note_id, text, vector_blob in rows:
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue
            vector = self.embedder.deserialize_vector(vector_blob)
            score = Embedder.dot_product(query, vector)
            scored.append({
                "embedding_id": embedding_id,
                "note_id": note_id,
                "text": text,
                "score": score,
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
            for item in dense:
                item.pop("embedding_id", None)
            return dense[:top_k]
        if not dense:
            return [
                {"note_id": s["note_id"], "text": s["text"], "score": -s["bm25"]}
                for s in sparse[:top_k]
            ]

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

        fused = [
            {"note_id": v["note_id"], "text": v["text"], "score": v["rrf"]}
            for v in ranks.values()
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
