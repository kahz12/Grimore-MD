"""Run the shipped Obsidian plugin checks without installing an editor."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


def test_obsidian_plugin_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js required for the Obsidian plugin's automated checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(root / "tests/obsidian_plugin.test.cjs")],
        cwd=root, capture_output=True, text=True, timeout=30,
        env={**os.environ, "GRIMORE_OBSIDIAN_TEST_PYTHON": sys.executable},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _cli_config(tmp_path, *, exists=True):
    vault = tmp_path / "vault"
    if exists:
        vault.mkdir()
    database = tmp_path / "index.db"
    (tmp_path / "grimore.toml").write_text(
        f"[vault]\npath = {json.dumps(str(vault))}\n"
        f"[memory]\ndb_path = {json.dumps(str(database))}\n"
        "[output]\nauto_commit = false\ndry_run = true\n",
        encoding="utf-8",
    )
    return database


def test_mcp_cli_metadata_starts_without_models_and_keeps_stdout_clean(tmp_path):
    database = _cli_config(tmp_path)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "grimore_library_info", "arguments": {},
        }},
    ]
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "GRIMORE_PROFILE": "",
           "OLLAMA_HOST": "http://127.0.0.1:1"}
    result = subprocess.run(
        [sys.executable, "-m", "grimore", "mcp"], cwd=tmp_path,
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True, text=True, timeout=15, env=env,
    )
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[0]["result"]["serverInfo"]["name"] == "grimore"
    metadata = json.loads(responses[1]["result"]["content"][0]["text"])
    assert metadata["contract_version"] == 1
    assert "mcp_server_started" in result.stderr
    assert not database.exists()  # Identity queries do not even initialise SQLite.


def test_mcp_cli_invalid_vault_fails_on_stderr_only(tmp_path):
    _cli_config(tmp_path, exists=False)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "grimore", "mcp"], cwd=tmp_path,
        input="", capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(root), "GRIMORE_PROFILE": ""},
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "not an accessible directory" in result.stderr
