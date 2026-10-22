"""Unified, read-only maintenance review inbox."""
from datetime import datetime, timedelta, timezone

from grimore.cognition.embedder import Embedder
from grimore.cognition.review_inbox import build_review_inbox
from grimore.session import Session
from grimore.utils.config import Config, MemoryConfig, OutputConfig, VaultConfig


def _session(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return Session(Config(
        vault=VaultConfig(path=str(vault), ignored_dirs=[]),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
    ))


def _note(db, path, title, content_hash, vector=None, text="chunk"):
    note_id = db.upsert_note(path, title, content_hash=content_hash)
    if vector is not None:
        db.store_embedding(
            note_id, 0, text, Embedder.serialize_vector(vector),
            chunk_hash=f"chunk-{note_id}",
            full_text_content=text,
        )
    return note_id


def test_inbox_aggregates_existing_signals_without_writes(tmp_path):
    session = _session(tmp_path)
    db = session.db

    # Exact duplicate group.
    _note(db, "/vault/copy-a.md", "Copy A", "same", [0.0, 1.0], "Shared exact text.")
    _note(db, "/vault/copy-b.md", "Copy B", "same", [0.0, 1.0], "Shared exact text.")

    # Semantic near-duplicate pair with different hashes.
    _note(db, "/vault/near-a.md", "Near A", "near-a", [1.0, 0.0], "Similar passage A.")
    _note(db, "/vault/near-b.md", "Near B", "near-b", [0.999, 0.001], "Similar passage B.")

    # Review-due note with an explicit human verification timestamp. A scan
    # timestamp alone must never be presented as verification.
    stale_id = _note(db, "/vault/old.md", "Old", "old", [0.0, 0.0, 1.0], "Old indexed passage.")
    db.set_note_category(stale_id, "tech/linux")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    db.upsert_freshness(
        "/vault/old.md", old, 90, verification_source="human:test",
    )

    # Persisted open contradiction from the existing Mirror store.
    _note(db, "/vault/claim-a.md", "Claim A", "claim-a")
    _note(db, "/vault/claim-b.md", "Claim B", "claim-b")
    db.replace_claims_for_note(
        "/vault/claim-a.md", [("A is true.", 0, 10, None)], old,
    )
    db.replace_claims_for_note(
        "/vault/claim-b.md", [("A is false.", 0, 11, None)], old,
    )
    claims = db.get_claim_extraction_state()
    assert len(claims) == 2
    with db._get_connection() as conn:
        claim_ids = [row[0] for row in conn.execute("SELECT id FROM claims ORDER BY id")]
    db.insert_contradiction(
        claim_ids[0], claim_ids[1], "high", "Opposite claims.", old,
    )

    before_freshness = db.list_freshness()
    payload = build_review_inbox(session)

    assert payload["status"] == "candidates_only"
    assert payload["counts"] == {
        "all": 4,
        "contradiction": 1,
        "duplicate_exact": 1,
        "duplicate_near": 1,
        "review_due": 1,
    }
    assert {item["kind"] for item in payload["items"]} == {
        "contradiction", "duplicate_exact", "duplicate_near", "review_due",
    }
    assert all(item["status"] == "candidate" for item in payload["items"])
    by_kind = {item["kind"]: item for item in payload["items"]}
    assert all(item["documents"] for item in payload["items"])
    assert all(item["passages"] for item in payload["items"])
    assert all(item["coverage"]["basis"] for item in payload["items"])
    assert all(item["coverage"]["limitations"] for item in payload["items"])
    assert by_kind["contradiction"]["passages"][0]["source"] == "extracted_claim"
    assert by_kind["contradiction"]["coverage"]["caps"] == {
        "max_sections_per_document": 8,
        "max_claims_per_document": 80,
        "max_characters_per_document": 64000,
    }
    assert by_kind["duplicate_exact"]["coverage"]["status"] == "complete"
    assert by_kind["duplicate_near"]["coverage"]["chunks_analyzed"] == 2
    assert by_kind["duplicate_near"]["coverage"]["caps"] == {"max_results": 30}
    assert by_kind["review_due"]["coverage"]["status"] == "partial"
    assert by_kind["review_due"]["passages"][0]["text"] == "Old indexed passage."
    assert by_kind["review_due"]["documents"][0]["classification"] == {
        "category": "tech/linux",
        "tags": [],
        "manual_override": False,
        "override_updated_at": None,
        "override_source": None,
    }
    assert by_kind["review_due"]["documents"][0]["timeline"]["verified"] == {
        "value": old,
        "source": "human:test",
    }
    assert db.list_freshness() == before_freshness == [("/vault/old.md", 90)]
    assert db.count_open_contradictions() == 1


def test_empty_inbox_has_stable_counts(tmp_path):
    payload = build_review_inbox(_session(tmp_path))

    assert payload["counts"] == {
        "all": 0,
        "contradiction": 0,
        "duplicate_exact": 0,
        "duplicate_near": 0,
        "review_due": 0,
    }
    assert payload["items"] == []


def test_near_duplicate_shows_passages_closest_to_other_document_mean(tmp_path):
    session = _session(tmp_path)
    session.config.cognition.dedupe_threshold = 0.6
    db = session.db
    a_id = _note(db, "/vault/a.md", "A", "a", [1.0, 0.0], "A matching passage.")
    db.store_embedding(
        a_id, 1, "A unrelated passage.", Embedder.serialize_vector([0.0, 1.0]),
        chunk_hash="a-1", full_text_content="A unrelated passage.",
    )
    b_id = _note(db, "/vault/b.md", "B", "b", [1.0, 0.0], "B matching passage.")
    db.store_embedding(
        b_id, 1, "B second passage.", Embedder.serialize_vector([1.0, 0.0]),
        chunk_hash="b-1", full_text_content="B second passage.",
    )

    payload = build_review_inbox(session)

    candidate = next(item for item in payload["items"] if item["kind"] == "duplicate_near")
    assert [passage["text"] for passage in candidate["passages"]] == [
        "A matching passage.", "B matching passage.",
    ]
    assert candidate["coverage"]["chunks_analyzed"] == 4
    assert candidate["coverage"]["passages"] == {"shown": 2, "available": 4}


def test_inbox_exposes_manual_classification_for_document_editing(tmp_path):
    session = _session(tmp_path)
    first_id = _note(session.db, "/vault/a.md", "A", "same")
    _note(session.db, "/vault/b.md", "B", "same")
    session.db.set_classification_override(
        note_id=first_id,
        category="human/reference",
        tags=["curated", "trusted"],
        updated_at="2026-09-25T12:00:00Z",
    )

    candidate = build_review_inbox(session)["items"][0]
    first = next(document for document in candidate["documents"] if document["note_id"] == first_id)

    assert first["classification"] == {
        "category": "human/reference",
        "tags": ["curated", "trusted"],
        "manual_override": True,
        "override_updated_at": "2026-09-25T12:00:00Z",
        "override_source": "web",
    }


def test_decisions_hide_candidates_but_keep_auditable_history(tmp_path):
    session = _session(tmp_path)
    _note(session.db, "/vault/a.md", "A", "same")
    _note(session.db, "/vault/b.md", "B", "same")
    candidate = build_review_inbox(session)["items"][0]
    session.db.record_review_decision(
        candidate_id=candidate["id"],
        kind=candidate["kind"],
        decision="confirm",
        decided_at="2026-09-25T10:00:00Z",
        postpone_until=None,
        snapshot={"documents": candidate["documents"], "reason": candidate["reason"]},
    )

    active = build_review_inbox(session)
    complete = build_review_inbox(session, include_inactive=True)

    assert active["items"] == []
    assert active["counts"]["all"] == 0
    assert active["history"][0]["decision"] == "confirm"
    assert complete["items"][0]["review_state"]["decision"] == "confirm"


def test_expired_postponement_returns_candidate_to_inbox(tmp_path):
    session = _session(tmp_path)
    _note(session.db, "/vault/a.md", "A", "same")
    _note(session.db, "/vault/b.md", "B", "same")
    candidate = build_review_inbox(session)["items"][0]
    session.db.record_review_decision(
        candidate_id=candidate["id"],
        kind=candidate["kind"],
        decision="postpone",
        decided_at="2026-09-25T10:00:00Z",
        postpone_until="2999-01-01T00:00:00Z",
        snapshot={"documents": candidate["documents"]},
    )
    assert build_review_inbox(session)["items"] == []

    session.db.record_review_decision(
        candidate_id=candidate["id"],
        kind=candidate["kind"],
        decision="postpone",
        decided_at="2000-01-01T00:00:00Z",
        postpone_until="2000-01-02T00:00:00Z",
        snapshot={"documents": candidate["documents"]},
    )

    visible = build_review_inbox(session)["items"]

    assert len(visible) == 1
    assert visible[0]["review_state"]["postpone_active"] is False
