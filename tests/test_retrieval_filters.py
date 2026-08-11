"""Retrieval can be narrowed to a subset of notes by category, tag or format.

Two properties carry the feature, and both are easy to get subtly wrong:

* **Nothing outside the filter is ever returned.** The filter is applied before
  the top-k cut, not after: post-filtering the global winners returns almost
  nothing exactly when the filter matters, because the excluded notes are the
  ones that scored highest.
* **An empty match is not the same as no filter.** ``None`` means "search
  everything"; an empty set means "a filter was asked for and nothing matched",
  which must return nothing rather than quietly searching the whole vault and
  answering from notes the caller excluded.
"""
import struct

import pytest

from grimore.cognition.connector import Connector
from grimore.memory.db import Database

_np = pytest.importorskip("numpy")


def _vec(*floats) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


class _StubEmbedder:
    model = "stub"

    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "grimore.db"))
    yield database
    database.close()


@pytest.fixture
def vault(db):
    """Three notes: two Markdown under a category tree, one TXT outside it."""
    ids = {}
    ids["infra"] = db.upsert_note(
        path="/v/infra.md", title="Infra", content_hash="a" * 64, format="md")
    ids["net"] = db.upsert_note(
        path="/v/net.md", title="Networking", content_hash="b" * 64, format="md")
    ids["notes"] = db.upsert_note(
        path="/v/loose.txt", title="Loose", content_hash="c" * 64, format="txt")
    db.set_note_category(ids["infra"], "infra")
    db.set_note_category(ids["net"], "infra/networking")
    db.upsert_tags(ids["infra"], ["compliance", "retention"])
    db.upsert_tags(ids["net"], ["retention"])
    # Identical text and vector everywhere, so ranking cannot distinguish them
    # and only the filter can.
    for nid in ids.values():
        db.store_embedding(nid, 0, "retention policy text", _vec(1.0, 0.0))
    return ids


def _connector(db):
    return Connector(db, _StubEmbedder(), vector_backend="numpy",
                     matrix_cache_enabled=False)


class TestResolveNoteFilter:
    def test_no_criteria_means_no_filter(self, db, vault):
        assert db.resolve_note_filter() is None

    def test_category_includes_descendants(self, db, vault):
        got = db.resolve_note_filter(category="infra")
        assert got == {vault["infra"], vault["net"]}

    def test_format_is_case_insensitive_and_dot_tolerant(self, db, vault):
        assert db.resolve_note_filter(formats=["TXT"]) == {vault["notes"]}
        assert db.resolve_note_filter(formats=[".txt"]) == {vault["notes"]}

    def test_several_formats_are_a_union(self, db, vault):
        assert db.resolve_note_filter(formats=["md", "txt"]) == set(vault.values())

    def test_a_single_tag(self, db, vault):
        assert db.resolve_note_filter(tags=["retention"]) == \
            {vault["infra"], vault["net"]}

    def test_several_tags_require_all_of_them(self, db, vault):
        """Narrowing is the point of the feature, so repeated --tag is AND."""
        assert db.resolve_note_filter(tags=["retention", "compliance"]) == \
            {vault["infra"]}

    def test_tags_are_case_insensitive(self, db, vault):
        assert db.resolve_note_filter(tags=["ReTeNtIoN"]) == \
            {vault["infra"], vault["net"]}

    def test_criteria_combine_with_and(self, db, vault):
        assert db.resolve_note_filter(category="infra", formats=["md"]) == \
            {vault["infra"], vault["net"]}
        assert db.resolve_note_filter(category="infra", formats=["txt"]) == set()

    def test_no_match_is_an_empty_set_not_none(self, db, vault):
        got = db.resolve_note_filter(tags=["does-not-exist"])
        assert got == set()
        assert got is not None, "an empty match must not read as 'no filter'"

    def test_blank_criteria_are_ignored(self, db, vault):
        assert db.resolve_note_filter(category="", tags=[], formats=[]) is None
        assert db.resolve_note_filter(tags=["  "]) is None


class TestFilteredRetrieval:
    @pytest.mark.parametrize("hybrid", [True, False])
    def test_only_filtered_notes_come_back(self, db, vault, hybrid):
        conn = _connector(db)
        allowed = {vault["notes"]}
        if hybrid:
            hits = conn.find_hybrid("retention policy", [1.0, 0.0], top_k=5,
                                    filter_note_ids=allowed)
        else:
            hits = conn.find_similar_notes([1.0, 0.0], top_k=5,
                                           filter_note_ids=allowed)
        assert hits, "the filtered note should still be retrievable"
        assert {h["note_id"] for h in hits} == allowed

    @pytest.mark.parametrize("hybrid", [True, False])
    def test_an_empty_filter_returns_nothing(self, db, vault, hybrid):
        conn = _connector(db)
        if hybrid:
            assert conn.find_hybrid("retention", [1.0, 0.0], top_k=5,
                                    filter_note_ids=set()) == []
        else:
            assert conn.find_similar_notes([1.0, 0.0], top_k=5,
                                           filter_note_ids=set()) == []

    @pytest.mark.parametrize("hybrid", [True, False])
    def test_no_filter_searches_everything(self, db, vault, hybrid):
        conn = _connector(db)
        if hybrid:
            hits = conn.find_hybrid("retention policy", [1.0, 0.0], top_k=5)
        else:
            hits = conn.find_similar_notes([1.0, 0.0], top_k=5)
        assert {h["note_id"] for h in hits} == set(vault.values())

    def test_the_filter_survives_a_full_top_k(self, db, vault):
        """With top_k smaller than the allowed set, the cut must still only
        pick from inside it."""
        conn = _connector(db)
        allowed = {vault["infra"], vault["net"]}
        hits = conn.find_similar_notes([1.0, 0.0], top_k=1,
                                       filter_note_ids=allowed)
        assert len(hits) == 1 and hits[0]["note_id"] in allowed

    def test_filtering_beats_ranking(self, db):
        """The excluded note is the strongest match. Post-filtering the global
        top-k would return nothing; filtering before the cut returns the best
        note that survives.
        """
        strong = db.upsert_note(path="/v/s.md", title="Strong",
                                content_hash="d" * 64, format="md")
        weak = db.upsert_note(path="/v/w.md", title="Weak",
                              content_hash="e" * 64, format="txt")
        db.store_embedding(strong, 0, "exact", _vec(1.0, 0.0))
        db.store_embedding(weak, 0, "distant", _vec(0.0, 1.0))

        conn = _connector(db)
        unfiltered = conn.find_similar_notes([1.0, 0.0], top_k=1)
        assert unfiltered[0]["note_id"] == strong

        hits = conn.find_similar_notes([1.0, 0.0], top_k=1,
                                       filter_note_ids={weak})
        assert hits and hits[0]["note_id"] == weak


class TestFilteredFtsSearch:
    def test_fts_restricts_to_the_allowed_notes(self, db, vault):
        if not db.fts_available:
            pytest.skip("FTS5 not available in this build")
        rows = db.fts_search("retention", limit=10,
                             filter_note_ids={vault["notes"]})
        assert rows and {r[1] for r in rows} == {vault["notes"]}

    def test_fts_empty_filter_short_circuits(self, db, vault):
        assert db.fts_search("retention", limit=10, filter_note_ids=set()) == []

    def test_fts_without_a_filter_is_unchanged(self, db, vault):
        if not db.fts_available:
            pytest.skip("FTS5 not available in this build")
        rows = db.fts_search("retention", limit=10)
        assert {r[1] for r in rows} == set(vault.values())

    def test_a_filter_too_wide_to_bind_still_filters(self, db, vault, monkeypatch):
        """Past the host-parameter budget the filter moves to a post-pass;
        the results must not change."""
        from grimore.memory import search as search_mod
        if not db.fts_available:
            pytest.skip("FTS5 not available in this build")
        monkeypatch.setattr(search_mod, "_MAX_FILTER_IDS", 1)
        rows = db.fts_search("retention", limit=10,
                             filter_note_ids={vault["infra"], vault["net"]})
        assert rows and {r[1] for r in rows} == {vault["infra"], vault["net"]}


class TestOracleFilter:
    def test_retrieve_only_cites_filtered_notes(self, db, vault):
        from grimore.cognition.oracle import Oracle
        oracle = Oracle.__new__(Oracle)
        oracle.db = db
        oracle.embedder = _StubEmbedder()
        oracle.connector = _connector(db)
        oracle.conditional_rewrite = True
        oracle.config = type("C", (), {"cognition": type("K", (), {
            "hybrid_search": False, "rrf_k": 60})()})()

        allowed = {vault["net"]}
        hits = oracle.retrieve("retention", top_k=5, filter_note_ids=allowed)
        assert hits and {h["note_id"] for h in hits} == allowed


class TestSearchApiFilter:
    """`POST /api/search` accepts the same narrowing, and refuses a malformed
    filter rather than silently returning unfiltered results -- a client that
    believes a filter is applied is the dangerous case."""

    def _client(self, db, vault):
        pytest.importorskip("starlette")
        from starlette.testclient import TestClient
        from grimore.api.app import build_app

        class _Session:
            def __init__(self):
                self.db = db
                self.embedder = _StubEmbedder()
                self.oracle = type("O", (), {"connector": _connector(db)})()
                self.config = type("C", (), {"cognition": type("K", (), {
                    "hybrid_search": False})()})()

        return TestClient(build_app(_Session()))

    def test_tags_narrow_the_results(self, db, vault):
        client = self._client(db, vault)
        r = client.post("/api/search",
                        json={"query": "retention", "tags": ["compliance"]})
        assert r.status_code == 200
        assert {row["note_id"] for row in r.json()["hits"]} == {vault["infra"]}

    def test_no_filter_keys_search_everything(self, db, vault):
        client = self._client(db, vault)
        r = client.post("/api/search", json={"query": "retention"})
        assert {row["note_id"] for row in r.json()["hits"]} == set(vault.values())

    def test_a_filter_matching_nothing_returns_no_results(self, db, vault):
        client = self._client(db, vault)
        r = client.post("/api/search",
                        json={"query": "retention", "tags": ["nope"]})
        assert r.status_code == 200 and r.json()["hits"] == []

    def test_a_malformed_filter_is_a_400(self, db, vault):
        client = self._client(db, vault)
        for bad in ({"tags": 5}, {"category": ["a"]}, {"formats": [1, 2]}):
            r = client.post("/api/search", json={"query": "retention", **bad})
            assert r.status_code == 400, f"{bad} should be rejected, not ignored"
