import os

import pytest

from grimore.utils.index_progress import (
    IndexProgressReporter,
    read_index_progress,
    recovery_actions,
)


@pytest.fixture
def progress_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "grimore.utils.paths.user_cache_dir",
        lambda _name: str(tmp_path / "cache"),
    )


def test_progress_roundtrip_counts_and_bounded_paths(tmp_path, progress_cache):
    reporter = IndexProgressReporter(tmp_path, "scan")
    reporter.start(total=3, dry_run=False)
    reporter.begin_item("notes/a.md")
    reporter.record("processed", "notes/a.md", chunks=4)
    reporter.record("skipped", "private.md", reason="privacy")
    reporter.record("failed", "broken.pdf", error="parse failed")
    reporter.finish()

    run = read_index_progress(tmp_path)[0]
    assert run["status"] == "completed_with_errors"
    assert run["counts"] == {
        "completed": 3,
        "processed": 1,
        "skipped": 1,
        "failed": 1,
        "previewed": 0,
        "removed": 0,
        "chunks": 4,
    }
    assert run["current_path"] is None
    assert [item["path"] for item in run["items"]] == [
        "notes/a.md", "private.md", "broken.pdf",
    ]
    assert {action["id"] for action in recovery_actions([run])} == {
        "retry-failed", "review-privacy",
    }


def test_dead_running_process_is_reported_interrupted(tmp_path, progress_cache):
    reporter = IndexProgressReporter(tmp_path, "scan")
    reporter.start(total=10, dry_run=False)
    reporter._state["pid"] = 999_999_999
    reporter._publish()

    run = read_index_progress(tmp_path)[0]
    assert run["status"] == "interrupted"
    assert recovery_actions([run])[0]["id"] == "resume-interrupted"


def test_preview_records_bounded_proposed_file_changes(tmp_path, progress_cache):
    reporter = IndexProgressReporter(tmp_path, "scan")
    reporter.start(total=1, dry_run=True)
    reporter.record("previewed", "note.md", changes={
        "target": "note.md",
        "action": "update",
        "fields": [{
            "name": "summary",
            "before": "old",
            "after": "x" * 400,
        }],
    })
    reporter.finish()

    item = read_index_progress(tmp_path)[0]["items"][0]
    assert item["changes"]["target"] == "note.md"
    assert item["changes"]["action"] == "update"
    assert item["changes"]["fields"][0]["before"] == "old"
    assert item["changes"]["fields"][0]["after"].endswith("…")
    assert len(item["changes"]["fields"][0]["after"]) == 301


def test_active_process_stays_running(tmp_path, progress_cache):
    reporter = IndexProgressReporter(tmp_path, "daemon")
    reporter.start(total=None, dry_run=True)

    run = read_index_progress(tmp_path)[0]
    assert run["pid"] == os.getpid()
    assert run["status"] == "running"
    assert run["total"] is None


def test_no_snapshot_suggests_safe_preview(tmp_path, progress_cache):
    assert read_index_progress(tmp_path) == []
    assert recovery_actions([])[0]["command"] == "grimore scan --dry-run"


def test_unavailable_cache_never_stops_indexing(tmp_path, monkeypatch):
    def unavailable(*_args):
        raise OSError("read-only cache")

    monkeypatch.setattr("grimore.utils.index_progress.index_progress_path", unavailable)
    reporter = IndexProgressReporter(tmp_path, "scan")

    reporter.start(total=1, dry_run=False)
    reporter.begin_item("note.md")
    reporter.record("processed", "note.md", chunks=1)
    reporter.finish()

    assert reporter._disabled is True
    assert read_index_progress(tmp_path) == []
