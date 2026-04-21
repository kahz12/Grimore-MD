import hashlib
import os
import struct
from typing import List, Optional, Protocol

from grimoire.cognition.chunker import chunk_markdown
from grimoire.utils.http import build_session
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

logger = get_logger(__name__)

# Hard cap on payload to the embedder to prevent memory/DoS issues
# from extremely large chunks. Roughly ~8k tokens worst case.
EMBED_MAX_CHARS = 32_000


class EmbeddingCache(Protocol):
    def get_cached_embedding(self, key: str) -> Optional[bytes]: ...
    def store_cached_embedding(self, key: str, vector_blob: bytes) -> None: ...


class Embedder:
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
        h = hashlib.sha256()
        # Include the model so a model swap invalidates cached vectors.
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def chunk(self, text: str) -> list[str]:
        return chunk_markdown(text)

    def embed(self, text: str) -> Optional[list[float]]:
        """
        Generate a unit-normalized embedding vector using Ollama.
        Cached by sha256(model + text) when a cache is configured.
        """
        if not isinstance(text, str) or not text.strip():
            return None
        if len(text) > EMBED_MAX_CHARS:
            logger.warning("embed_input_truncated", original=len(text), kept=EMBED_MAX_CHARS)
            text = text[:EMBED_MAX_CHARS]

        key = self._cache_key(text) if self.cache is not None else None
        if key is not None:
            cached = self.cache.get_cached_embedding(key)
            if cached is not None:
                return self.deserialize_vector(cached)

        try:
            url = f"{self.ollama_host}/api/embeddings"
            payload = {"model": self.model, "prompt": text}
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            raw = response.json().get("embedding")
            if not raw:
                return None
            vector = self.normalize(raw)
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
        """Dot product. Equivalent to cosine similarity for unit-normalized vectors."""
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculates cosine similarity between two vectors (kept for compat)."""
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = sum(a * a for a in v1) ** 0.5
        m2 = sum(b * b for b in v2) ** 0.5
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)
