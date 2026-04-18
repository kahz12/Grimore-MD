import os
import requests
import json
import struct
from typing import List, Optional
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class Embedder:
    def __init__(self, config):
        self.config = config
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = config.cognition.model_embeddings_local

    def embed(self, text: str) -> Optional[list[float]]:
        """
        Generates an embedding vector using Ollama.
        """
        try:
            url = f"{self.ollama_host}/api/embeddings"
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            return response.json().get("embedding")
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            return None

    @staticmethod
    def serialize_vector(vector: list[float]) -> bytes:
        """Serializes a list of floats to a compact binary format (float32)."""
        return struct.pack(f'{len(vector)}f', *vector)

    @staticmethod
    def deserialize_vector(data: bytes) -> list[float]:
        """Deserializes binary data back to a list of floats."""
        num_floats = len(data) // 4
        return list(struct.unpack(f'{num_floats}f', data))

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
