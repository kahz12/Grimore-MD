import math

from grimoire.cognition.embedder import Embedder


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
        for a, b in zip(original, restored):
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
