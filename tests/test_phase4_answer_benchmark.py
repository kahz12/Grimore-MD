"""No model calls in CI: exercise preflight, isolation and input identity."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval import benchmark_answers as bench


def test_local_endpoint_restores_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://not-used.example")
    with bench.local_endpoint():
        assert os.environ["OLLAMA_HOST"] == bench.LOCAL_HOST
    assert os.environ["OLLAMA_HOST"] == "https://not-used.example"
    monkeypatch.delenv("OLLAMA_HOST")
    with bench.local_endpoint():
        assert os.environ["OLLAMA_HOST"] == bench.LOCAL_HOST
    assert "OLLAMA_HOST" not in os.environ


def test_only_selected_installed_model_identities_and_no_private_parent_path():
    http = MagicMock()
    http.get.return_value.json.return_value = {"models": [
        {"name": "embed:latest", "digest": "abc", "size": 5,
         "details": {"family": "test", "parent_model": "/private/path"}},
        {"name": "unrelated:latest", "digest": "xyz"},
    ]}
    identity = bench.installed_identity(http, ["embed"])
    assert list(identity) == ["embed"]
    assert identity["embed"]["digest"] == "abc"
    assert "private" not in json.dumps(identity) and "unrelated" not in json.dumps(identity)
    http.post.assert_not_called()
    with pytest.raises(ValueError, match="no download attempted"):
        bench.installed_identity(http, ["missing"])
    http.get.return_value.json.return_value = {"models": [{"name": "embed:latest"}]}
    with pytest.raises(ValueError, match="digest required"):
        bench.installed_identity(http, ["embed"])


def test_tuning_guard_does_not_load_held_out_dataset(monkeypatch):
    fake = MagicMock(split="evaluation", document_formats=["pdf"])
    monkeypatch.setattr(bench, "load_golden", lambda _: [fake])
    session = MagicMock()
    monkeypatch.setattr(bench, "Session", session)
    with pytest.raises(ValueError, match="never held-out"):
        bench.benchmark(bench.DEFAULT_CASES, bench.DEFAULT_PDF, Path("grimore.toml"))
    session.assert_not_called()


def test_preflight_failure_uses_only_owned_temp_paths_and_cleans_them(tmp_path, monkeypatch):
    config_path = tmp_path / "test.toml"
    config_path.write_text('[vault]\npath="/not-used"\n[memory]\ndb_path="/not-used/db"\n')
    original = bench.DEFAULT_PDF.read_bytes()
    captured = {}

    class FailingSession:
        def __init__(self, config):
            captured["config"] = config

        def __enter__(self):
            config = captured["config"]
            vault = Path(config.vault.path)
            captured["vault"] = vault
            assert vault.is_dir() and (vault / bench.DEFAULT_PDF.name).read_bytes() == original
            assert Path(config.memory.db_path).parent == vault.parent
            assert config.output.dry_run and not config.output.auto_commit
            assert config.cognition.allow_remote is False
            assert config.cognition.llm_backend == "ollama"
            assert os.environ["OLLAMA_HOST"] == bench.LOCAL_HOST
            raise RuntimeError("Controlled preflight failure")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench, "Session", FailingSession)
    monkeypatch.setenv("OLLAMA_HOST", "https://not-used.example")
    with pytest.raises(RuntimeError, match="preflight failure"):
        bench.benchmark(bench.DEFAULT_CASES, bench.DEFAULT_PDF, config_path, model="already-installed")
    assert not captured["vault"].parent.exists()
    assert bench.DEFAULT_PDF.read_bytes() == original
    assert os.environ["OLLAMA_HOST"] == "https://not-used.example"
    assert captured["config"].cognition.model_llm_local == "already-installed"


def test_fingerprints_include_models_pipeline_and_detect_changed_input(tmp_path):
    config = tmp_path / "test.toml"
    config.write_text("# first")
    first = bench.fingerprints(bench.DEFAULT_CASES, bench.DEFAULT_PDF, config)
    config.write_text("# second")
    second = bench.fingerprints(bench.DEFAULT_CASES, bench.DEFAULT_PDF, config)
    assert first["config_file_sha256"] != second["config_file_sha256"]
    assert first["pdf_sha256"] == second["pdf_sha256"]
    assert "grimore/cognition/prompts/oracle.txt" in first["code_sha256"]
    assert "grimore/ingest/pdf_layout.py" in first["code_sha256"]
    assert "eval/context_order_control.py" in first["code_sha256"]


def test_context_order_refused_before_models_and_cli_defaults_are_explicit(monkeypatch):
    session = MagicMock()
    monkeypatch.setattr(bench, "Session", session)
    with pytest.raises(ValueError, match="Unknown context-order"):
        bench.benchmark(bench.DEFAULT_CASES, bench.DEFAULT_PDF, Path("grimore.toml"),
                        context_order="select-expected")
    session.assert_not_called()


@pytest.mark.parametrize("top_k,retrieval_k", [(0, 10), (-1, 10), (5, 3)])
def test_invalid_context_limits_rejected_before_session(top_k, retrieval_k):
    with pytest.raises(ValueError, match="retrieval_k"):
        bench.benchmark(bench.DEFAULT_CASES, bench.DEFAULT_PDF, Path("grimore.toml"),
                        top_k=top_k, retrieval_k=retrieval_k)


def test_export_refuses_overwrite_before_model_calls(tmp_path, monkeypatch):
    report = tmp_path / "existing.json"
    report.write_text("preserve")
    run = MagicMock()
    monkeypatch.setattr(bench, "benchmark", run)
    monkeypatch.setattr("sys.argv", ["benchmark_answers", "--output", str(report)])
    assert bench.main() == 2
    assert report.read_text() == "preserve"
    run.assert_not_called()


def test_export_preserves_complete_results_and_stdout_is_not_truncated(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    result = {"turns": [{"answer": "Long response " * 10000}], "manual_review": [{"claim_support": "pending"}]}
    monkeypatch.setattr(bench, "benchmark", lambda *a, **kw: result)
    monkeypatch.setattr("sys.argv", ["benchmark_answers", "--output", str(report)])
    assert bench.main() == 0
    assert json.loads(report.read_text()) == result
    assert capsys.readouterr().out == ""
