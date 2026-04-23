"""
Configuration Management.
This module defines the project's configuration schema using dataclasses
and handles loading settings from 'grimoire.toml' and environment variables.
"""
import os
from pathlib import Path
import tomllib
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load environment variables from .env file
load_dotenv()

@dataclass
class VaultConfig:
    """Settings related to the Markdown vault location and filtering."""
    path: str = "./vault"
    ignored_dirs: list[str] = field(default_factory=lambda: [".obsidian", ".trash", ".git", "Templates"])

@dataclass
class CognitionConfig:
    """Settings for the LLM and Embedding models (Ollama)."""
    model_llm_local: str = "qwen2.5:3b"
    model_embeddings_local: str = "nomic-embed-text"
    allow_remote: bool = False  # If False, only loopback addresses are allowed for Ollama
    # Retrieval: fuse BM25 (FTS5) and cosine similarity via Reciprocal Rank Fusion.
    hybrid_search: bool = True
    rrf_k: int = 60

@dataclass
class MemoryConfig:
    """Settings for the persistence layer (SQLite)."""
    db_path: str = "grimoire.db"

@dataclass
class OutputConfig:
    """Settings for how Grimoire writes back to the vault."""
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
class Config:
    """Main configuration container."""
    vault: VaultConfig = field(default_factory=VaultConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)

def load_config(config_path: str = "grimoire.toml") -> Config:
    """
    Loads configuration from a TOML file.
    Falls back to default values if the file is missing or partially defined.
    """
    path = Path(config_path)
    if not path.exists():
        return Config()
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    # Map TOML data to dataclasses
    vault_data = data.get("vault", {})
    cognition_data = data.get("cognition", {})
    memory_data = data.get("memory", {})
    output_data = data.get("output", {})
    maintenance_data = data.get("maintenance", {})

    return Config(
        vault=VaultConfig(**vault_data),
        cognition=CognitionConfig(**cognition_data),
        memory=MemoryConfig(**memory_data),
        output=OutputConfig(**output_data),
        maintenance=MaintenanceConfig(**maintenance_data),
    )
