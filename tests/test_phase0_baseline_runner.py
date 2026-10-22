"""Plan/workspace/subprocess contracts of the Phase-0 runner; never contact Ollama."""
import hashlib
import json
from pathlib import Path

import pytest

from eval import run_phase0_baseline as runner


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A stand-in reference vault whose files match a pinned manifest."""
    source = tmp_path / "source"
    source.mkdir()
    pinned = {}
    for name in ("note.md", "report.pdf"):
        (source / name).write_bytes(f"content of {name}".encode())
        pinned[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "SOURCE_VAULT", source)
    monkeypatch.setattr(runner, "CORPUS_SHA256", pinned)
    return source


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "grimore.toml"
    path.write_text('[vault]\npath = "./test_vault"\n[memory]\ndb_path = "grimore.db"\n',
                    encoding="utf-8")
    return path


def _report(failures=0, notes=2):
    return {"aggregate": {"generation_failure_n": failures},
            "provenance": {"corpus": {"notes": notes}}}


def _fake_grimore(workspace, results):
    """subprocess.run stand-in: each eval writes the next queued report."""
    calls = []

    def run(argv, *, cwd, **_):
        calls.append(argv[3:])
        if "--export" in argv:
            target = Path(cwd) / argv[argv.index("--export") + 1]
            target.write_text(json.dumps(results.pop(0)), encoding="utf-8")
        return type("Done", (), {"returncode": 0})()

    return calls, run


def test_plan_writes_nothing_and_lists_isolated_steps(corpus, config, tmp_path):
    schedule = runner.plan(config, tmp_path / "ws")
    assert schedule["status"] == "planned-not-executed" and schedule["problems"] == []
    assert [s["id"] for s in schedule["steps"]] == list(runner.STEPS)
    full = next(s for s in schedule["steps"] if s["id"] == "evaluation_full")["argv"]
    assert "--checkpoint" in full and "--retrieval-only" not in full
    assert not (tmp_path / "ws").exists()


def test_changed_corpus_or_escaping_config_blocks_execution(corpus, config, tmp_path):
    (corpus / "note.md").write_text("edited", encoding="utf-8")
    config.write_text('[vault]\npath = "/home/someone/vault"\n', encoding="utf-8")
    schedule = runner.plan(config, tmp_path / "ws")
    assert schedule["status"] == "blocked"
    assert "changed: note.md" in schedule["problems"]
    assert any("[vault] path" in p for p in schedule["problems"])
    with pytest.raises(ValueError):
        runner.execute(schedule, tmp_path / "ws")
    assert not (tmp_path / "ws").exists()


def test_workspace_copies_the_pinned_corpus_and_refuses_foreign_inputs(corpus, config, tmp_path):
    workspace = tmp_path / "ws"
    runner.prepare(runner.plan(config, workspace), workspace)
    assert (workspace / "test_vault/test_vault/note.md").read_bytes() == (corpus / "note.md").read_bytes()
    runner.prepare(runner.plan(config, workspace), workspace)  # same inputs resume
    (workspace / "grimore.toml").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="other inputs"):
        runner.prepare(runner.plan(config, workspace), workspace)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.md").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="not a Phase-0 workspace"):
        runner.prepare(runner.plan(config, stray), stray)


def test_contaminated_report_is_kept_aside_and_the_step_retried(corpus, config, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    results = [_report(), _report(), _report(), _report(failures=1), _report()]
    calls, fake = _fake_grimore(workspace, results)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    summary = runner.execute(runner.plan(config, workspace), workspace, retries=1)
    assert summary["status"] == "completed-measurements"
    assert [c[0] for c in calls] == ["scan"] + ["eval"] * 5
    assert len(list((workspace / "reports").glob("evaluation_full.contaminated-*.json"))) == 1
    # A relaunch finds every step finished and starts no process.
    calls.clear()
    assert runner.execute(runner.plan(config, workspace), workspace)["status"] == "completed-measurements"
    assert calls == []


def test_retries_are_bounded_and_a_partial_index_is_an_error(corpus, config, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    calls, fake = _fake_grimore(workspace, [_report(), _report(), _report(failures=1)])
    monkeypatch.setattr(runner.subprocess, "run", fake)
    summary = runner.execute(runner.plan(config, workspace), workspace, retries=0)
    assert summary["status"] == "incomplete-measurements"
    assert summary["attempts"][-1] == {**summary["attempts"][-1], "id": "tuning_full", "status": "incomplete"}

    other = tmp_path / "other"
    _, fake = _fake_grimore(other, [_report(notes=1)])
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(ValueError, match="indexed notes"):
        runner.execute(runner.plan(config, other), other)
