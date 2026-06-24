"""
Vector Embedding Generation and Management.
This module uses Ollama's embedding API to generate semantic vectors for note chunks.
It includes support for caching, unit-normalization, and serialization of vectors.
"""
import hashlib
import os
import struct
from typing import Iterable, List, Optional, Protocol

from grimore.cognition.chunker import chunk_markdown, chunk_sections
from grimore.ingest.adapters.base import ExtractedSection
from grimore.utils.http import build_session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

try:  # numpy powers the vectorized retrieval fast path; degrade if absent.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a declared dep
    _np = None

logger = get_logger(__name__)

# Hard cap on payload to the embedder to prevent memory/DoS issues
# from extremely large chunks. Roughly ~8k tokens worst case.
EMBED_MAX_CHARS = 32_000

# Max texts per /api/embed request. Bounds per-request work + memory; the
# misses are streamed in sub-batches of this size. A sub-batch that fails
# (e.g. an Ollama too old to expose /api/embed) falls back to serial embed().
_EMBED_BATCH_SIZE = 32


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
        allow_remote = config.cognition.allow_remote
        self.ollama_host = SecurityGuard.validate_llm_host(
            raw_host, allow_remote=allow_remote
        )
        self.model = config.cognition.model_embeddings_local
        # Pin loopback HTTP to the validated address (audit I1: DNS-rebinding).
        self.session = build_session(
            pins=SecurityGuard.loopback_pins(raw_host, allow_remote=allow_remote)
        )
        self.cache = cache

    def _cache_key(self, text: str) -> str:
        """Generates a unique cache key based on the model name and input text."""
        h = hashlib.sha256()
        # Include the model so a model swap invalidates cached vectors.
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def chunk_hash(text: str, model: str) -> str:
        """Stable fingerprint over (model, chunk text) for incremental re-embed.

        Truncated to 32 hex chars — collision-resistant enough at the
        per-note scale (a doc with billions of chunks is not a realistic
        adversary) and noticeably cheaper than the 64-char form to index
        and compare. Including the model name in the digest means a model
        swap implicitly invalidates every chunk_hash, so the migrate path
        doesn't need a separate "wipe hashes" step.
        """
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()[:32]

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
            self._cache_store(text, vector)
            return vector
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            return None

    def _cache_store(self, text: str, vector: list[float]) -> None:
        """Write one normalized vector to the cache (no-op without a cache)."""
        if self.cache is None:
            return
        try:
            self.cache.store_cached_embedding(
                self._cache_key(text), self.serialize_vector(vector)
            )
        except Exception as e:
            logger.warning("embed_cache_write_failed", error=str(e))

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed many texts with as few round-trips as possible.

        Returns vectors aligned to ``texts`` (``None`` for blank entries or
        failed embeds), so callers can ``zip`` the result straight back onto
        their inputs. Cache hits are served locally; the remaining misses go
        to Ollama's batch endpoint (``/api/embed`` with an ``input`` array)
        in bounded sub-batches. This replaces the per-item ``embed()`` loop
        the chunker and re-embed paths used to run — a 100-sentence document
        drops from 100 sequential requests to a handful.

        A sub-batch that the server can't satisfy (e.g. an Ollama too old to
        expose ``/api/embed``) transparently falls back to serial ``embed()``
        for that slice, so behaviour is never worse than the old path.
        """
        results: list[Optional[list[float]]] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_txt: list[str] = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                continue
            t = text[:EMBED_MAX_CHARS] if len(text) > EMBED_MAX_CHARS else text
            if self.cache is not None:
                cached = self.cache.get_cached_embedding(self._cache_key(t))
                if cached is not None:
                    results[i] = self.deserialize_vector(cached)
                    continue
            pending_idx.append(i)
            pending_txt.append(t)

        for start in range(0, len(pending_txt), _EMBED_BATCH_SIZE):
            sub_idx = pending_idx[start:start + _EMBED_BATCH_SIZE]
            sub_txt = pending_txt[start:start + _EMBED_BATCH_SIZE]
            raw_vecs = self._embed_many_remote(sub_txt)
            if raw_vecs is None:
                # Endpoint unavailable / errored → serial fallback (still caches).
                for i, t in zip(sub_idx, sub_txt, strict=True):
                    results[i] = self.embed(t)
                continue
            for i, t, raw in zip(sub_idx, sub_txt, raw_vecs, strict=True):
                if not raw:
                    continue
                vector = self.normalize(raw)
                results[i] = vector
                self._cache_store(t, vector)
        return results

    def _embed_many_remote(
        self, texts: list[str]
    ) -> Optional[list[Optional[list[float]]]]:
        """One POST to Ollama's batch endpoint. Returns *raw* (un-normalized)
        vectors aligned to ``texts``, or ``None`` when the endpoint is missing
        or errors so the caller can fall back to serial ``embed()``."""
        try:
            url = f"{self.ollama_host}/api/embed"
            payload = {"model": self.model, "input": texts}
            response = self.session.post(
                url, json=payload, timeout=self.config.cognition.embed_timeout_s
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                return None
            return embeddings
        except Exception as e:
            logger.warning("embed_batch_unavailable", error=str(e))
            return None

    def embed_chunks(self, text: str) -> list[tuple[str, list[float]]]:
        """
        Split ``text`` into markdown-aware chunks and embed each.
        Returns a list of (chunk_text, vector) pairs; failed chunks are skipped.

        Anchor-free path. Multi-format callers should use
        :py:meth:`embed_sections` so page / heading anchors land in the
        embedding rows; this method is preserved for tests and the
        Markdown call sites that have nothing useful to anchor on.
        """
        chunks = self.chunk(text)
        results: list[tuple[str, list[float]]] = []
        for chunk in chunks:
            vec = self.embed(chunk)
            if vec is not None:
                results.append((chunk, vec))
        return results

    def embed_sections(
        self,
        sections: Iterable[ExtractedSection],
    ) -> list[tuple[str, list[float], Optional[int], Optional[str]]]:
        """
        Section-aware embedding path used by the multi-format pipeline.

        Each input section is split via :func:`chunk_sections` so the
        anchors propagate to every produced chunk. Returns a list of
        ``(chunk_text, vector, page, heading)`` tuples — failed embed
        calls are skipped (same policy as :py:meth:`embed_chunks`).

        Caller invariant: pass ``note.sections`` straight through. When
        ``sections`` is empty (Markdown / TXT), this returns an empty
        list and the caller falls back to :py:meth:`embed_chunks` on
        the raw body.
        """
        results: list[tuple[str, list[float], Optional[int], Optional[str]]] = []
        for chunk in chunk_sections(sections):
            vec = self.embed(chunk.text)
            if vec is None:
                continue
            results.append((chunk.text, vec, chunk.page, chunk.heading))
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
    def vectors_to_matrix(blobs: list[bytes]):
        """Stack raw float32 vector BLOBs into an ``(N, D)`` numpy matrix.

        This is the fast path for retrieval: building one contiguous matrix
        from the stored bytes lets the Connector score every chunk with a
        single matmul instead of a per-row Python dot product.

        Returns ``None`` (so callers fall back to the per-row path) when:
        * numpy is unavailable, or
        * ``blobs`` is empty, or
        * the vectors are ragged — mixed byte lengths, which happens when the
          embedding model was swapped without a re-scan (different dims), or a
          blob is truncated. We don't try to coerce these; the loop fallback
          handles them one vector at a time exactly as before.
        """
        if _np is None or not blobs:
            return None
        first_len = len(blobs[0])
        if first_len == 0 or first_len % 4 != 0:
            return None
        for blob in blobs:
            if len(blob) != first_len:
                return None  # ragged → caller uses the per-row fallback
        try:
            flat = _np.frombuffer(b"".join(blobs), dtype=_np.float32)
            return flat.reshape(len(blobs), first_len // 4)
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def dot_product(v1: list[float], v2: list[float]) -> float:
        """Calculates dot product. Equivalent to cosine similarity for unit-normalized vectors."""
        # strict=False: ragged dims (model swapped without re-scan) must
        # truncate and score, not raise — see the retrieval fallback path.
        return sum(a * b for a, b in zip(v1, v2, strict=False))

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculates cosine similarity between two vectors (magnitude-aware)."""
        # strict=False: tolerate ragged dims (see dot_product above).
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        m1 = sum(a * a for a in v1) ** 0.5
        m2 = sum(b * b for b in v2) ** 0.5
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)
