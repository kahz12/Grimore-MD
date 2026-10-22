"""Suite-wide isolation shared by every test module."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_user_cache(tmp_path_factory, monkeypatch):
    """Keep every test out of the developer's real user cache.

    Grimore stores the daemon lock and log, shell history, indexing progress
    and private change backups there. Without this, each run left files —
    document backups included — in ``~/.cache/grimore``. The environment
    variable covers child processes (MCP and plugin tests); the patch covers
    platforms where platformdirs ignores ``XDG_CACHE_HOME``. A test that pins
    its own cache still wins, since its patch is applied afterwards.
    """
    root = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.setattr(
        "grimore.utils.paths.user_cache_dir",
        lambda appname=None, *args, **kwargs: str(root / (appname or "grimore")),
    )
