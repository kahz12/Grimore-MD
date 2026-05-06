"""B-03 regression: .env is loaded from the cwd ONLY, never via upward walk."""
import os

import pytest

from grimore.utils.config import _load_project_env


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip GRIMORE_TEST_* keys before and after each test."""
    snapshot = {k: v for k, v in os.environ.items() if k.startswith("GRIMORE_TEST_")}
    for k in list(os.environ):
        if k.startswith("GRIMORE_TEST_"):
            monkeypatch.delenv(k, raising=False)
    yield
    for k in list(os.environ):
        if k.startswith("GRIMORE_TEST_"):
            monkeypatch.delenv(k, raising=False)
    for k, v in snapshot.items():
        monkeypatch.setenv(k, v)


def test_loads_env_from_cwd(tmp_path, monkeypatch, isolated_env):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("GRIMORE_TEST_OWN=one\n")

    monkeypatch.chdir(project)
    assert _load_project_env() is True
    assert os.environ.get("GRIMORE_TEST_OWN") == "one"


def test_does_not_walk_upward_to_parent_env(tmp_path, monkeypatch, isolated_env):
    """The whole point of B-03: a parent .env must NOT leak into the env."""
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".env").write_text("GRIMORE_TEST_PARENT=parent_value\n")

    child = parent / "child"
    child.mkdir()  # No .env here.

    monkeypatch.chdir(child)
    # Returns False because no .env in cwd; crucially it must NOT have
    # picked up the parent's value via an upward walk.
    assert _load_project_env() is False
    assert "GRIMORE_TEST_PARENT" not in os.environ


def test_existing_env_var_is_not_overridden(tmp_path, monkeypatch, isolated_env):
    project = tmp_path / "p"
    project.mkdir()
    (project / ".env").write_text("GRIMORE_TEST_KEEP=from_dotenv\n")

    monkeypatch.setenv("GRIMORE_TEST_KEEP", "from_shell")
    monkeypatch.chdir(project)
    _load_project_env()
    # override=False: shell wins.
    assert os.environ["GRIMORE_TEST_KEEP"] == "from_shell"
