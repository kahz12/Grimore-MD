"""
Configuration Management.
This module defines the project's configuration schema using dataclasses
and handles loading settings from 'grimore.toml' and environment variables.
"""
import os
import re
from pathlib import Path
import tomllib
from dotenv import load_dotenv
from dataclasses import dataclass, field, fields

from grimore.utils.logger import get_logger

logger = get_logger(__name__)


def _load_project_env() -> bool:
    """
    Load Grimore's ``.env`` from the current working directory ONLY.

    python-dotenv's default ``load_dotenv()`` walks upward from the cwd
    until it finds a ``.env``. If Grimore happens to be run from a nested
    directory whose ancestor holds an unrelated project's ``.env``, that
    file's variables (potentially OLLAMA_HOST or other secrets) would
    silently leak into Grimore's environment. Pinning the path matches
    how ``grimore.toml`` is discovered — both anchored to the cwd — and
    closes B-03.
    """
    return load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


_load_project_env()

@dataclass
class VaultConfig:
    """Settings related to the Markdown vault location and filtering."""
    path: str = "./vault"
    ignored_dirs: list[str] = field(default_factory=lambda: [".obsidian", ".trash", ".git", "Templates"])


def is_ignored_path(file_path, ignored_dirs: list[str]) -> bool:
    """
    Whether ``file_path`` sits under one of the ``ignored_dirs`` *as a real
    path component*.

    Why not substring matching: ``".git" in str(path)`` also hits
    ``/vault/.gitignore-test/`` or any note whose title contains ".git". We
    want the segment to match a full directory name.
    """
    if not ignored_dirs:
        return False
    parts = Path(file_path).parts
    ignored = set(ignored_dirs)
    return any(part in ignored for part in parts)

@dataclass
class CognitionConfig:
    """Settings for the LLM and Embedding models (Ollama)."""
    model_llm_local: str = "qwen2.5:3b"
    model_embeddings_local: str = "nomic-embed-text"
    allow_remote: bool = False  # If False, only loopback addresses are allowed for Ollama
    # Retrieval: fuse BM25 (FTS5) and cosine similarity via Reciprocal Rank Fusion.
    hybrid_search: bool = True
    rrf_k: int = 60
    # Minimum cosine-similarity score for `connect` to propose a wikilink.
    # Below this the candidate is dropped; exposed so vaults that lean more
    # on RAG recall can relax it and vice versa.
    connect_threshold: float = 0.7
    # Per-call HTTP timeouts (seconds) for Ollama. Defaults suit qwen2.5:3b on
    # modest hardware; bigger models (e.g. ministral-3:14b) routinely need
    # request_timeout_s ≥ 180 on the first warm-up call.
    request_timeout_s: int = 60   # /api/generate, JSON path
    stream_timeout_s: int = 120   # /api/generate, streaming path
    embed_timeout_s: int = 30     # /api/embeddings

@dataclass
class MemoryConfig:
    """Settings for the persistence layer (SQLite)."""
    db_path: str = "grimore.db"

@dataclass
class OutputConfig:
    """Settings for how Grimore writes back to the vault."""
    auto_commit: bool = True  # Automatically commit changes to Git before writing
    dry_run: bool = True     # If True, no changes are actually written to disk

@dataclass
class MaintenanceConfig:
    """
    Periodic housekeeping performed by the daemon: VACUUM, WAL checkpoint,
    purge of unused tags. Runs once per ``interval_hours`` while the daemon
    is idle enough for a brief exclusive lock.
    """
    enabled: bool = True
    interval_hours: int = 24
    vacuum: bool = True
    purge_tags: bool = True
    wal_checkpoint: bool = True

@dataclass
class ChroniclerConfig:
    """
    Chronicler — temporal staleness tracking.

    ``windows`` maps a (case-insensitive, prefix-matched) category path
    to a freshness window in *days*. A value of ``0`` is the explicit
    "never stale" sentinel — chosen over ``None`` because TOML users can
    override 0 cleanly. Categories that don't match any rule are also
    exempt from staleness reporting.

    Defaults are the v2.1 plan's "Suggested" answer to Q3.
    """
    windows: dict[str, int] = field(default_factory=lambda: {
        "tech/": 90,
        "tools/": 90,
        "infra/": 90,
        "dev/": 180,
        "code-snippets/": 180,
        "concepts/": 0,
        "theory/": 0,
        "journal/": 0,
        "daily/": 0,
    })

@dataclass
class DaemonConfig:
    """
    Ambient watcher mode (Section 4 of v2.1 plan).

    Disabled by default — the daemon is opt-in. Setting ``enabled = true``
    in ``grimore.toml``'s ``[daemon]`` section signals to the CLI that the
    user has chosen to run it; ``grimore daemon start`` always works
    regardless, but tooling that auto-starts the daemon (e.g. an `init`
    hook) should respect this flag.

    ``log_events`` controls whether per-save one-liners are appended to
    ``daemon.log`` under the user cache dir. When false the daemon still
    runs structured logging through ``structlog``; this just skips the
    extra append-only file.
    """
    enabled: bool = False
    log_events: bool = True


@dataclass
class Config:
    """Main configuration container."""
    vault: VaultConfig = field(default_factory=VaultConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    chronicler: ChroniclerConfig = field(default_factory=ChroniclerConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

def _filter_known(cls, data: dict, section: str) -> dict:
    """
    Keep only keys the dataclass knows about.

    Why: TOML is edited by hand. An unknown key should warn, not crash the
    whole CLI with TypeError — especially for people copy-pasting snippets
    from an older README.
    """
    known = {f.name for f in fields(cls)}
    filtered = {}
    for key, value in data.items():
        if key in known:
            filtered[key] = value
        else:
            logger.warning("unknown_config_key", section=section, key=key)
    return filtered


def load_config(config_path: str = "grimore.toml") -> Config:
    """
    Loads configuration from a TOML file.
    Falls back to default values if the file is missing or partially defined.
    """
    path = Path(config_path)
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return Config(
        vault=VaultConfig(**_filter_known(VaultConfig, data.get("vault", {}), "vault")),
        cognition=CognitionConfig(**_filter_known(CognitionConfig, data.get("cognition", {}), "cognition")),
        memory=MemoryConfig(**_filter_known(MemoryConfig, data.get("memory", {}), "memory")),
        output=OutputConfig(**_filter_known(OutputConfig, data.get("output", {}), "output")),
        maintenance=MaintenanceConfig(**_filter_known(MaintenanceConfig, data.get("maintenance", {}), "maintenance")),
        chronicler=ChroniclerConfig(**_filter_known(ChroniclerConfig, data.get("chronicler", {}), "chronicler")),
        daemon=DaemonConfig(**_filter_known(DaemonConfig, data.get("daemon", {}), "daemon")),
    )


def _set_cognition_string(text: str, key: str, value: str) -> str:
    """Replace ``key = "..."`` inside the ``[cognition]`` section, or append it.

    Comment-preserving in-place edit. We deliberately avoid serializing
    the whole TOML — there's no stdlib writer, and the alternatives
    either drop comments (tomli-w) or add a dependency (tomlkit).

    Handles double- and single-quoted basic strings; unquoted/literal
    multi-line strings fall through to "append" (logged as a warning by
    the caller if the resulting file is malformed).
    """
    cog_match = re.search(r"^\[cognition\]\s*$", text, flags=re.MULTILINE)
    if not cog_match:
        sep = "" if text.endswith("\n") else "\n"
        return f'{text}{sep}\n[cognition]\n{key} = "{value}"\n'

    start = cog_match.end()
    next_section = re.search(r"^\[", text[start:], flags=re.MULTILINE)
    end = start + next_section.start() if next_section else len(text)
    section_body = text[start:end]

    pattern = re.compile(
        rf'^(?P<lead>\s*{re.escape(key)}\s*=\s*)(?:"[^"]*"|\'[^\']*\')(?P<trail>.*)$',
        flags=re.MULTILINE,
    )
    if pattern.search(section_body):
        new_body = pattern.sub(rf'\g<lead>"{value}"\g<trail>', section_body)
    else:
        trailing = "" if section_body.endswith("\n") else "\n"
        new_body = f'{section_body}{trailing}{key} = "{value}"\n'

    return text[:start] + new_body + text[end:]


def update_cognition_models(
    chat_model: str | None = None,
    embedding_model: str | None = None,
    config_path: str = "grimore.toml",
) -> bool:
    """Persist a model swap into ``[cognition]`` of ``grimore.toml``.

    Returns True if the file was rewritten. Returns False if no toml
    exists at ``config_path`` (in which case defaults will continue to
    apply on next start — caller decides whether to warn).
    """
    path = Path(config_path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if chat_model is not None:
        text = _set_cognition_string(text, "model_llm_local", chat_model)
    if embedding_model is not None:
        text = _set_cognition_string(text, "model_embeddings_local", embedding_model)
    path.write_text(text, encoding="utf-8")
    return True
