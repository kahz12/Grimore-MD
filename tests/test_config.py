"""B-03 regression: .env is loaded from the cwd ONLY, never via upward walk."""
import os

import pytest

from grimore.utils.config import (
    _deep_merge,
    _load_project_env,
    _set_cognition_string,
    get_active_profile,
    load_config,
    set_active_profile,
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


# ── multi-vault profiles ───────────────────────────────────────────────────


@pytest.fixture
def reset_profile_state(monkeypatch):
    """Ensure each profile test starts with a clean module-level stash + env."""
    monkeypatch.delenv("GRIMORE_PROFILE", raising=False)
    previous = get_active_profile()
    set_active_profile(None)
    yield
    set_active_profile(previous)


class TestDeepMerge:
    def test_overlay_replaces_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_overlay_adds_keys(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overlay_merges_nested_dicts(self):
        merged = _deep_merge(
            {"vault": {"path": "/base", "ignored_dirs": [".git"]}},
            {"vault": {"path": "/over"}},
        )
        # path overridden, ignored_dirs inherited.
        assert merged == {"vault": {"path": "/over", "ignored_dirs": [".git"]}}

    def test_lists_are_replaced_not_concatenated(self):
        # Profile setting formats = ["md"] means "only Markdown", not
        # "Markdown also" — explicit choice in the plan.
        merged = _deep_merge({"vault": {"formats": ["md", "pdf"]}},
                             {"vault": {"formats": ["md"]}})
        assert merged == {"vault": {"formats": ["md"]}}


_PROFILE_TOML = """
[vault]
path = "./default-vault"
ignored_dirs = [".obsidian"]

[cognition]
model_llm_local = "qwen2.5:3b"

[profiles.work]
[profiles.work.vault]
path = "/home/me/work-notes"
display_name = "Work"

[profiles.work.cognition]
model_llm_local = "qwen2.5:14b"

[profiles.personal]
[profiles.personal.vault]
path = "/home/me/notes"
"""


class TestProfiles:
    def test_no_profile_returns_defaults(self, tmp_path, reset_profile_state):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        cfg = load_config(str(path))
        assert cfg.vault.path == "./default-vault"
        assert cfg.cognition.model_llm_local == "qwen2.5:3b"
        assert cfg.active_profile is None

    def test_profile_arg_deep_merges_over_defaults(self, tmp_path, reset_profile_state):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        cfg = load_config(str(path), profile="work")
        # Overridden by profile.
        assert cfg.vault.path == "/home/me/work-notes"
        assert cfg.vault.display_name == "Work"
        assert cfg.cognition.model_llm_local == "qwen2.5:14b"
        # Inherited from defaults.
        assert cfg.vault.ignored_dirs == [".obsidian"]
        assert cfg.active_profile == "work"

    def test_profile_env_var_picked_up(self, tmp_path, reset_profile_state, monkeypatch):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        monkeypatch.setenv("GRIMORE_PROFILE", "personal")
        cfg = load_config(str(path))
        assert cfg.vault.path == "/home/me/notes"
        assert cfg.active_profile == "personal"

    def test_cli_flag_beats_env_var(self, tmp_path, reset_profile_state, monkeypatch):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        monkeypatch.setenv("GRIMORE_PROFILE", "personal")
        set_active_profile("work")
        cfg = load_config(str(path))
        assert cfg.vault.path == "/home/me/work-notes"
        assert cfg.active_profile == "work"

    def test_explicit_arg_beats_cli_flag(self, tmp_path, reset_profile_state):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        set_active_profile("work")
        cfg = load_config(str(path), profile="personal")
        assert cfg.vault.path == "/home/me/notes"
        assert cfg.active_profile == "personal"

    def test_unknown_profile_raises_clean_error(self, tmp_path, reset_profile_state):
        path = tmp_path / "grimore.toml"
        path.write_text(_PROFILE_TOML, encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown profile"):
            load_config(str(path), profile="ghost")

    def test_missing_config_with_profile_raises(self, tmp_path, reset_profile_state):
        missing = tmp_path / "nope.toml"
        with pytest.raises(ValueError, match="Cannot apply profile"):
            load_config(str(missing), profile="work")

    def test_missing_config_without_profile_returns_defaults(self, tmp_path, reset_profile_state):
        missing = tmp_path / "nope.toml"
        cfg = load_config(str(missing))
        assert cfg.vault.path == "./vault"
        assert cfg.active_profile is None

    def test_profile_with_empty_section_does_not_crash(self, tmp_path, reset_profile_state):
        path = tmp_path / "grimore.toml"
        # Profile declared but empty — should yield base defaults marked
        # with the profile name.
        path.write_text("[profiles.bare]\n", encoding="utf-8")
        cfg = load_config(str(path), profile="bare")
        assert cfg.active_profile == "bare"
        assert cfg.vault.path == "./vault"
