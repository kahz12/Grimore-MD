"""B-03 regression: .env is loaded from the cwd ONLY, never via upward walk."""
import os

import pytest

from grimore.utils.config import (
    _load_project_env,
    _set_cognition_string,
    update_cognition_models,
)


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


# ── _set_cognition_string + update_cognition_models ────────────────────────


class TestSetCognitionString:
    """Regex-based, comment-preserving update of [cognition] keys."""

    def test_replaces_double_quoted_value(self):
        text = '[cognition]\nmodel_llm_local = "old"\n'
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert 'model_llm_local = "new"' in out

    def test_replaces_single_quoted_value(self):
        text = "[cognition]\nmodel_llm_local = 'old'\n"
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert 'model_llm_local = "new"' in out

    def test_preserves_inline_comment(self):
        text = '[cognition]\nmodel_llm_local = "old"  # a comment\n'
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert 'model_llm_local = "new"  # a comment' in out

    def test_preserves_other_keys_in_section(self):
        text = (
            '[cognition]\n'
            'model_llm_local = "old"\n'
            'allow_remote = false\n'
            'rrf_k = 60\n'
        )
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert "allow_remote = false" in out
        assert "rrf_k = 60" in out

    def test_does_not_touch_other_sections(self):
        text = (
            '[vault]\nmodel_llm_local = "do not touch"\n'
            '[cognition]\nmodel_llm_local = "old"\n'
            '[memory]\ndb_path = "x.db"\n'
        )
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert '[vault]\nmodel_llm_local = "do not touch"' in out
        assert 'db_path = "x.db"' in out
        # Only the cognition occurrence flips.
        assert out.count('model_llm_local = "new"') == 1

    def test_appends_when_key_absent_from_cognition(self):
        text = '[cognition]\nallow_remote = false\n[memory]\ndb_path = "x.db"\n'
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert 'model_llm_local = "new"' in out
        # Must land inside [cognition], i.e. before [memory].
        assert out.index('model_llm_local = "new"') < out.index("[memory]")

    def test_appends_section_when_no_cognition_block(self):
        text = '[vault]\npath = "./v"\n'
        out = _set_cognition_string(text, "model_llm_local", "new")
        assert "[cognition]" in out
        assert 'model_llm_local = "new"' in out


class TestUpdateCognitionModels:
    def test_returns_false_when_no_toml(self, tmp_path):
        path = tmp_path / "missing.toml"
        assert update_cognition_models(chat_model="x", config_path=str(path)) is False
        assert not path.exists()

    def test_rewrites_chat_only(self, tmp_path):
        path = tmp_path / "grimore.toml"
        path.write_text(
            '[cognition]\nmodel_llm_local = "a"\nmodel_embeddings_local = "b"\n',
            encoding="utf-8",
        )
        assert update_cognition_models(chat_model="z", config_path=str(path)) is True
        text = path.read_text(encoding="utf-8")
        assert 'model_llm_local = "z"' in text
        assert 'model_embeddings_local = "b"' in text

    def test_rewrites_both_when_supplied(self, tmp_path):
        path = tmp_path / "grimore.toml"
        path.write_text(
            '[cognition]\nmodel_llm_local = "a"\nmodel_embeddings_local = "b"\n',
            encoding="utf-8",
        )
        update_cognition_models(
            chat_model="zc", embedding_model="ze", config_path=str(path),
        )
        text = path.read_text(encoding="utf-8")
        assert 'model_llm_local = "zc"' in text
        assert 'model_embeddings_local = "ze"' in text
