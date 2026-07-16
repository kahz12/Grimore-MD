import math
from unittest.mock import MagicMock

from grimore.cognition.embedder import Embedder
from grimore.utils.config import CognitionConfig, Config


class TestNormalize:
    def test_unit_magnitude(self):
        v = Embedder.normalize([3.0, 4.0])
        mag = math.sqrt(sum(x * x for x in v))
        assert math.isclose(mag, 1.0, abs_tol=1e-9)

    def test_preserves_direction(self):
        v = Embedder.normalize([3.0, 4.0])
        assert math.isclose(v[0] / v[1], 3.0 / 4.0, abs_tol=1e-9)

    def test_zero_vector_stays_zero(self):
        assert Embedder.normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_negative_components(self):
        v = Embedder.normalize([-3.0, -4.0])
        mag = math.sqrt(sum(x * x for x in v))
        assert math.isclose(mag, 1.0, abs_tol=1e-9)


class TestSerialization:
    def test_roundtrip(self):
        original = [0.1, -0.2, 0.3, 0.4, -0.5]
        blob = Embedder.serialize_vector(original)
        restored = Embedder.deserialize_vector(blob)
        for a, b in zip(original, restored, strict=True):
            assert math.isclose(a, b, abs_tol=1e-6)

    def test_blob_is_bytes(self):
        assert isinstance(Embedder.serialize_vector([0.1, 0.2]), bytes)

    def test_blob_size_is_4_bytes_per_float(self):
        assert len(Embedder.serialize_vector([0.1, 0.2, 0.3])) == 12


class TestSimilarity:
    def test_dot_product_identical_normalized_is_one(self):
        v = Embedder.normalize([1.0, 2.0, 3.0])
        assert math.isclose(Embedder.dot_product(v, v), 1.0, abs_tol=1e-9)

    def test_dot_product_orthogonal_is_zero(self):
        a = Embedder.normalize([1.0, 0.0])
        b = Embedder.normalize([0.0, 1.0])
        assert math.isclose(Embedder.dot_product(a, b), 0.0, abs_tol=1e-9)

    def test_cosine_handles_zero_vector(self):
        assert Embedder.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_cosine_magnitude_independent(self):
        # cosine_similarity should not depend on magnitude
        a = [1.0, 0.0]
        b = [5.0, 0.0]
        assert math.isclose(Embedder.cosine_similarity(a, b), 1.0, abs_tol=1e-9)

    def test_dot_product_equals_cosine_for_unit_vectors(self):
        a = Embedder.normalize([1.0, 2.0, 3.0])
        b = Embedder.normalize([4.0, 5.0, 6.0])
        dot = Embedder.dot_product(a, b)
        cos = Embedder.cosine_similarity(a, b)
        assert math.isclose(dot, cos, abs_tol=1e-9)


class TestEmbedTimeoutWiring:
    """Embedder must read its HTTP timeout from CognitionConfig."""

    def _embedder_with_config(self, cognition: CognitionConfig) -> Embedder:
        cfg = Config()
        cfg.cognition = cognition
        emb = Embedder(cfg)
        emb.session = MagicMock()
        return emb

    def test_embed_uses_embed_timeout_from_config(self):
        emb = self._embedder_with_config(CognitionConfig(embed_timeout_s=45))
        resp = MagicMock()
        resp.json.return_value = {"embedding": [1.0, 0.0, 0.0]}
        emb.session.post.return_value = resp

        emb.embed("some text")

        kwargs = emb.session.post.call_args.kwargs
        assert kwargs["timeout"] == 45


class _DictCache:
    """Minimal EmbeddingCache backed by a dict."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def get_cached_embedding(self, key):
        return self.store.get(key)

    def store_cached_embedding(self, key, blob):
        self.store[key] = blob


class TestEmbedBatch:
    def _embedder(self, cache=None):
        emb = Embedder(Config(), cache=cache)
        emb.session = MagicMock()
        return emb

    def test_one_request_for_all_texts_normalized_and_aligned(self):
        emb = self._embedder()
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[3.0, 4.0], [0.0, 2.0]]}
        emb.session.post.return_value = resp

        out = emb.embed_batch(["alpha", "beta"])

        # Exactly one round-trip, to the batch endpoint, carrying both inputs.
        assert emb.session.post.call_count == 1
        url = emb.session.post.call_args.args[0] if emb.session.post.call_args.args \
            else emb.session.post.call_args.kwargs["url"]
        assert url.endswith("/api/embed")
        assert emb.session.post.call_args.kwargs["json"]["input"] == ["alpha", "beta"]
        # Vectors are unit-normalized and aligned to input order.
        assert math.isclose(out[0][0], 0.6) and math.isclose(out[0][1], 0.8)
        assert math.isclose(out[1][0], 0.0) and math.isclose(out[1][1], 1.0)

    def test_blank_entries_are_none_and_not_sent(self):
        emb = self._embedder()
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[1.0, 0.0]]}
        emb.session.post.return_value = resp

        out = emb.embed_batch(["", "   ", "real"])
        assert out[0] is None and out[1] is None
        assert emb.session.post.call_args.kwargs["json"]["input"] == ["real"]
        assert out[2] == [1.0, 0.0]

    def test_cache_hits_served_without_hitting_api(self):
        cache = _DictCache()
        emb = self._embedder(cache=cache)
        # Pre-seed "alpha"; only "beta" should reach the API.
        cache.store_cached_embedding(emb._cache_key("alpha"),
                                     Embedder.serialize_vector([0.0, 1.0]))
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[2.0, 0.0]]}
        emb.session.post.return_value = resp

        out = emb.embed_batch(["alpha", "beta"])
        assert emb.session.post.call_args.kwargs["json"]["input"] == ["beta"]
        assert out[0] == [0.0, 1.0]            # from cache
        assert out[1] == [1.0, 0.0]            # normalized API result
        # The freshly embedded miss is now cached too.
        assert emb._cache_key("beta") in cache.store

    def test_falls_back_to_serial_when_batch_endpoint_unavailable(self):
        emb = self._embedder()

        def fake_post(url, json=None, timeout=None, **kw):
            resp = MagicMock()
            if url.endswith("/api/embed"):
                resp.json.return_value = {}          # no 'embeddings' → unusable
            else:                                    # /api/embeddings (serial)
                resp.json.return_value = {"embedding": [1.0, 0.0, 0.0]}
            return resp

        emb.session.post.side_effect = fake_post
        out = emb.embed_batch(["a", "b"])
        # One batch attempt + one serial call per text.
        urls = [c.args[0] if c.args else c.kwargs["url"]
                for c in emb.session.post.call_args_list]
        assert urls.count("http://localhost:11434/api/embed") == 1
        assert urls.count("http://localhost:11434/api/embeddings") == 2
        assert out == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
