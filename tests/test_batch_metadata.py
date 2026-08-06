"""opt.2 — the Oracle's per-chunk metadata lookups collapsed into batches.

``_build_context`` used to call ``get_note_title`` once per retrieved chunk
and ``get_chunk_anchors`` once per cited chunk. These tests pin the two batch
replacements to the singular methods they stand in for: same results, same
tie-breaking, same treatment of misses — and, for the anchors query, no
cross-contamination from the ``IN x IN`` selection it uses to stay on the
note_id index.
"""
import struct

import pytest

from grimore.memory import chunks as chunks_mod
from grimore.memory import notes as notes_mod
from grimore.memory.db import Database


def _vec() -> bytes:
    return struct.pack("2f", 1.0, 0.0)


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "grimore.db"))
    yield database
    database.close()


def _note(db, path, title):
    return db.upsert_note(
        path=path, title=title, content_hash="c" * 64,
        format="pdf", file_hash="f" * 64, size_bytes=100,
    )


class TestGetNoteTitles:
    def test_matches_the_singular_lookup(self, db):
        ids = [_note(db, f"/v/n{i}.md", f"Title {i}") for i in range(5)]
        batch = db.get_note_titles(ids)
        assert batch == {i: db.get_note_title(i) for i in ids}

    def test_empty_input_makes_no_query(self, db):
        assert db.get_note_titles([]) == {}

    def test_missing_ids_are_absent_not_none(self, db):
        # The singular method returns None for an unknown id and the Oracle
        # treats a falsy title as an orphan embedding; `.get()` on a mapping
        # that simply omits the key reproduces that exactly.
        known = _note(db, "/v/known.md", "Known")
        result = db.get_note_titles([known, 9999])
        assert result == {known: "Known"}
        assert result.get(9999) is None
        assert db.get_note_title(9999) is None

    def test_duplicate_ids_collapse(self, db):
        known = _note(db, "/v/dup.md", "Dup")
        assert db.get_note_titles([known, known, known]) == {known: "Dup"}

    def test_chunks_past_the_host_parameter_limit(self, db, monkeypatch):
        # Lowering the cap is enough to exercise the loop; creating 900+ notes
        # to hit the real limit would only make the test slow.
        monkeypatch.setattr(notes_mod, "_MAX_SQL_VARS", 3)
        ids = [_note(db, f"/v/m{i}.md", f"T{i}") for i in range(10)]
        assert db.get_note_titles(ids) == {i: f"T{n}"
                                           for n, i in enumerate(ids)}


class TestGetChunkAnchorsBulk:
    def test_matches_the_singular_lookup_on_a_paginated_note(self, db):
        note_id = _note(db, "/v/doc.pdf", "Doc")
        db.store_embedding(note_id, 0, "page-one text", _vec(), page=1)
        db.store_embedding(note_id, 1, "page-two text", _vec(), page=2)
        db.store_embedding(note_id, 2, "heading-only", _vec(), heading="Intro")

        texts = ["page-one text", "page-two text", "heading-only"]
        pairs = [(note_id, t) for t in texts]
        assert db.get_chunk_anchors_bulk(pairs) == {
            (note_id, t): db.get_chunk_anchors(note_id, t) for t in texts
        }

    def test_misses_are_absent_so_get_yields_the_singular_contract(self, db):
        note_id = _note(db, "/v/doc.pdf", "Doc")
        db.store_embedding(note_id, 0, "real text", _vec(), page=7)
        result = db.get_chunk_anchors_bulk([(note_id, "no such chunk")])
        assert result == {}
        assert result.get((note_id, "no such chunk"), (None, None)) == (None, None)
        assert db.get_chunk_anchors(note_id, "no such chunk") == (None, None)

    def test_identical_text_in_two_notes_keeps_each_notes_anchor(self, db):
        """The selection is ``note_id IN (...) AND text_content IN (...)``,
        which also matches cross combinations. If the re-pairing were wrong,
        one note would inherit the other's page number.
        """
        a = _note(db, "/v/a.pdf", "A")
        b = _note(db, "/v/b.pdf", "B")
        shared = "the same paragraph in both documents"
        db.store_embedding(a, 0, shared, _vec(), page=3)
        db.store_embedding(b, 0, shared, _vec(), page=88)

        result = db.get_chunk_anchors_bulk([(a, shared), (b, shared)])
        assert result[(a, shared)] == (3, None)
        assert result[(b, shared)] == (88, None)
        # And the singular method agrees on both.
        assert result[(a, shared)] == db.get_chunk_anchors(a, shared)
        assert result[(b, shared)] == db.get_chunk_anchors(b, shared)

    def test_result_holds_only_the_requested_pairs(self, db):
        """``note_id IN (...) AND text_content IN (...)`` selects the full
        cross product, so with 2 notes x 2 texts the query returns 4 rows for
        2 requested pairs. The 2 nobody asked for must not appear: callers
        iterate the result in the Oracle's citation path, and an extra key
        there is an anchor for a chunk that was never retrieved.
        """
        a = _note(db, "/v/a.pdf", "A")
        b = _note(db, "/v/b.pdf", "B")
        # Both notes contain both texts, at different pages.
        db.store_embedding(a, 0, "alpha", _vec(), page=1)
        db.store_embedding(a, 1, "beta", _vec(), page=2)
        db.store_embedding(b, 0, "alpha", _vec(), page=50)
        db.store_embedding(b, 1, "beta", _vec(), page=51)

        requested = [(a, "alpha"), (b, "beta")]
        result = db.get_chunk_anchors_bulk(requested)

        assert set(result) == set(requested)
        assert result[(a, "alpha")] == (1, None)
        assert result[(b, "beta")] == (51, None)

    def test_duplicate_text_within_a_note_picks_the_same_row_as_limit_1(self, db):
        """``LIMIT 1`` with no ORDER BY returns the first row the scan reaches
        (lowest id under this plan). The batch orders by id and keeps the
        first occurrence, so both must land on the earlier chunk.
        """
        note_id = _note(db, "/v/doc.pdf", "Doc")
        dup = "a paragraph repeated verbatim"
        db.store_embedding(note_id, 0, dup, _vec(), page=1)
        db.store_embedding(note_id, 1, dup, _vec(), page=2)

        assert db.get_chunk_anchors_bulk([(note_id, dup)])[(note_id, dup)] == \
            db.get_chunk_anchors(note_id, dup)

    def test_empty_input_makes_no_query(self, db):
        assert db.get_chunk_anchors_bulk([]) == {}

    def test_chunks_past_the_host_parameter_limit(self, db, monkeypatch):
        # Both IN lists share one statement, so the cap has to bound their sum.
        monkeypatch.setattr(chunks_mod, "_MAX_SQL_VARS", 4)
        note_ids = [_note(db, f"/v/d{i}.pdf", f"D{i}") for i in range(6)]
        pairs = []
        for n, note_id in enumerate(note_ids):
            text = f"body {n}"
            db.store_embedding(note_id, 0, text, _vec(), page=n + 1)
            pairs.append((note_id, text))

        result = db.get_chunk_anchors_bulk(pairs)
        assert result == {p: (n + 1, None) for n, p in enumerate(pairs)}


class TestOracleQueryBudget:
    """The DoD for opt.2: an ask resolves its metadata in <= 2 statements,
    down from two per retrieved chunk.
    """

    def test_metadata_lookups_are_two_statements(self, db):
        from grimore.cognition.oracle import Oracle

        note_ids = [_note(db, f"/v/n{i}.pdf", f"N{i}") for i in range(5)]
        similar = []
        for n, note_id in enumerate(note_ids):
            text = f"chunk body {n}"
            db.store_embedding(note_id, 0, text, _vec(), page=n + 1)
            similar.append({"note_id": note_id, "text": text, "score": 1.0 - n / 10})

        oracle = Oracle.__new__(Oracle)
        oracle.db = db

        statements: list[str] = []
        conn = db._thread_conn()
        conn.set_trace_callback(statements.append)
        try:
            titles = oracle._titles_for(similar)
            anchors = oracle._anchors_for(similar)
        finally:
            conn.set_trace_callback(None)

        assert len(titles) == 5
        assert len(anchors) == 5
        # Filter out SQLite's own internal statements (prefixed with "--") and
        # the transaction control the context manager emits.
        app_statements = [
            s for s in statements
            if not s.startswith("--") and s.strip().upper().startswith("SELECT")
        ]
        assert len(app_statements) == 2, app_statements
