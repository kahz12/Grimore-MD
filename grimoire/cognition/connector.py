from typing import List

from grimoire.cognition.embedder import Embedder
from grimoire.memory.db import Database
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


class Connector:
    def __init__(self, db: Database, embedder: Embedder):
        self.db = db
        self.embedder = embedder

    def find_similar_notes(
        self,
        query_vector: List[float],
        top_k: int = 5,
        exclude_note_id: int = None,
    ):
        """
        Finds the top_k most similar chunks in the database.
        Stored and query vectors are unit-normalized by the embedder,
        so cosine similarity reduces to a dot product (one pass, no magnitudes).
        """
        query = list(query_vector)
        all_embeddings = self.db.get_all_embeddings()
        similarities = []

        for note_id, text, vector_blob in all_embeddings:
            if exclude_note_id is not None and note_id == exclude_note_id:
                continue

            vector = self.embedder.deserialize_vector(vector_blob)
            score = Embedder.dot_product(query, vector)
            similarities.append({"note_id": note_id, "text": text, "score": score})

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
