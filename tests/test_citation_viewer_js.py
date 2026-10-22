"""Run the real browser controller's navigation checks with optional Node.js."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_citation_selection_controller():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is optional; run node --test tests/citation_viewer.test.cjs for UI checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, "--test", str(root / "tests/citation_viewer.test.cjs")],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
