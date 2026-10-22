"""Persistent human category/tag overrides."""
from grimore.memory.classification_overrides import apply_classification_override
from grimore.memory.db import Database


def test_manual_classification_updates_index_and_overrides_future_proposals(tmp_path):
    path = tmp_path / "grimore.db"
    db = Database(str(path))
    note_id = db.upsert_note("/vault/note.md", "Note", "hash")
    assert note_id is not None
    db.set_note_category(note_id, "model/old")
    db.upsert_tags(note_id, ["old", "generated"])

    event = db.set_classification_override(
        note_id=note_id,
        category="human/reference",
        tags=["curated", "trusted"],
        updated_at="2026-09-25T12:00:00Z",
    )
    effective = db.resolve_classification_override(
        "/vault/note.md",
        proposed_tags=["new-model-tag"],
        proposed_category="model/new",
    )
    visible = db.get_note_classifications([note_id])[note_id]

    assert event["previous_category"] == "model/old"
    assert event["previous_tags"] == ["generated", "old"]
    assert effective == {
        "tags": ["curated", "trusted"],
        "category": "human/reference",
        "overridden": True,
        "updated_at": "2026-09-25T12:00:00Z",
        "source": "web",
    }
    assert visible["category"] == "human/reference"
    assert visible["tags"] == ["curated", "trusted"]
    assert visible["manual_override"] is True
    assert apply_classification_override(
        db,
        "/vault/note.md",
        {"tags": ["model"], "category": "model/new", "summary": "Kept"},
    ) == {
        "tags": ["curated", "trusted"],
        "category": "human/reference",
        "summary": "Kept",
    }
    db.close()

    reopened = Database(str(path))
    assert reopened.resolve_classification_override(
        "/vault/note.md", proposed_tags=[], proposed_category=None,
    )["tags"] == ["curated", "trusted"]


def test_manual_classification_can_intentionally_clear_values_and_audits_changes(tmp_path):
    db = Database(str(tmp_path / "grimore.db"))
    note_id = db.upsert_note("/vault/note.md", "Note", "hash")
    assert note_id is not None
    db.upsert_tags(note_id, ["old"])
    db.set_note_category(note_id, "old/category")

    db.set_classification_override(
        note_id=note_id,
        category=None,
        tags=[],
        updated_at="2026-09-25T12:00:00Z",
    )

    classification = db.get_note_classifications([note_id])[note_id]
    with db._get_connection() as conn:
        events = conn.execute(
            "SELECT COUNT(*) FROM classification_override_log WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0]
    assert classification["category"] is None
    assert classification["tags"] == []
    assert classification["manual_override"] is True
    assert events == 1
