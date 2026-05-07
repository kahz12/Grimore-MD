"""
Vector Embedding Generation and Management.
This module uses Ollama's embedding API to generate semantic vectors for note chunks.
It includes support for caching, unit-normalization, and serialization of vectors.
"""
import hashlib
import os
import struct
from typing import List, Optional, Protocol

from grimore.cognition.chunker import chunk_markdown
from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Hard cap on payload to the embedder to prevent memory/DoS issues
# from extremely large chunks. Roughly ~8k tokens worst case.
EMBED_MAX_CHARS = 32_000


class EmbeddingCache(Protocol):
    """Protocol for an optional persistence layer to store and retrieve vectors."""
    def get_cached_embedding(self, key: str) -> Optional[bytes]: ...
    def store_cached_embedding(self, key: str, vector_blob: bytes) -> None: ...


class Embedder:
    """
    Handles the generation and processing of vector embeddings via Ollama.
    """
    def __init__(self, config, cache: Optional[EmbeddingCache] = None):
        self.config = config
        raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=config.cognition.allow_remote
        )
        self.model = config.cognition.model_embeddings_local
        self.session = build_session()
        self.cache = cache

    def _cache_key(self, text: str) -> str:
        """Generates a unique cache key based on the model name and input text."""
        h = hashlib.sha256()
        # Include the model so a model swap invalidates cached vectors.
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def chunk(self, text: str) -> list[str]:
        """Delegates text splitting to the markdown-aware chunker."""
        return chunk_markdown(text)

    def embed(self, text: str) -> Optional[list[float]]:
        """
        Generate a unit-normalized embedding vector using Ollama.
        Cached by sha256(model + text) when a cache is configured.
        Returns a list of floats (vector).
        """
        if not isinstance(text, str) or not text.strip():
            return None
        if len(text) > EMBED_MAX_CHARS:
            logger.warning("embed_input_truncated", original=len(text), kept=EMBED_MAX_CHARS)
            text = text[:EMBED_MAX_CHARS]

        # Check cache before calling the API
        key = self._cache_key(text) if self.cache is not None else None
        if key is not None:
            cached = self.cache.get_cached_embedding(key)
            if cached is not None:
                return self.deserialize_vector(cached)

        try:
            url = f"{self.ollama_host}/api/embeddings"
            payload = {"model": self.model, "prompt": text}
            response = self.session.post(
                url, json=payload, timeout=self.config.cognition.embed_timeout_s
            )
            response.raise_for_status()
            raw = response.json().get("embedding")
            if not raw:
                return None
            
            # Unit-normalize the vector to allow dot-product similarity
            vector = self.normalize(raw)
            
            # Store result in cache
            if key is not None:
                try:
                    self.cache.store_cached_embedding(key, self.serialize_vector(vector))
                except Exception as e:
                    logger.warning("embed_cache_write_failed", error=str(e))
            return vector
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            return None

    def embed_chunks(self, text: str) -> list[tuple[str, list[float]]]:
        """
        Split ``text`` into markdown-aware chunks and embed each.
        Returns a list of (chunk_text, vector) pairs; failed chunks are skipped.
        """
        chunks = self.chunk(text)
        results: list[tuple[str, list[float]]] = []
        for chunk in chunks:
            vec = self.embed(chunk)
            if vec is not None:
                results.append((chunk, vec))
        return results

    @staticmethod
    def normalize(vector: List[float]) -> list[float]:
        """Normalizes a vector to have unit length (magnitude of 1)."""
        mag = sum(v * v for v in vector) ** 0.5
        if mag == 0:
            return list(vector)
        return [v / mag for v in vector]

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
    def dot_product(v1: list[float], v2: list[float]) -> float:
        """Calculates dot product. Equivalent to cosine similarity for unit-normalized vectors."""
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculates cosine similarity between two vectors (magnitude-aware)."""
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = sum(a * a for a in v1) ** 0.5
        m2 = sum(b * b for b in v2) ** 0.5
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)
