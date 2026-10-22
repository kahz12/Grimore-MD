"""Persistence and audit guarantees for maintenance review decisions."""
from grimore.memory.db import Database


def test_review_decision_state_and_log_survive_reopen(tmp_path):
    path = tmp_path / "grimore.db"
    db = Database(str(path))
    first = db.record_review_decision(
        candidate_id="duplicate-near:1:2",
        kind="duplicate_near",
        decision="postpone",
        decided_at="2026-09-25T10:00:00Z",
        postpone_until="2026-10-25T10:00:00Z",
        snapshot={"documents": [{"title": "A"}, {"title": "B"}]},
    )
    second = db.record_review_decision(
        candidate_id="duplicate-near:1:2",
        kind="duplicate_near",
        decision="confirm",
        decided_at="2026-10-26T10:00:00Z",
        postpone_until=None,
        snapshot={"documents": [{"title": "A"}, {"title": "B"}]},
    )

    assert first["previous_decision"] is None
    assert second["previous_decision"] == "postpone"
    db.close()

    reopened = Database(str(path))
    state = reopened.get_review_candidate_states(["duplicate-near:1:2"])
    history = reopened.list_review_decisions()

    assert state["duplicate-near:1:2"]["decision"] == "confirm"
    assert [entry["decision"] for entry in history] == ["confirm", "postpone"]
    assert history[0]["snapshot"]["documents"][0]["title"] == "A"


def test_review_decision_validation_rejects_invalid_postpone_shape(tmp_path):
    db = Database(str(tmp_path / "grimore.db"))

    try:
        db.record_review_decision(
            candidate_id="review-due:x",
            kind="review_due",
            decision="postpone",
            decided_at="2026-09-25T10:00:00Z",
            postpone_until=None,
            snapshot={},
        )
    except ValueError as error:
        assert "postpone_until" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("postpone without an expiry must fail")
