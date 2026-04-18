import os
from pathlib import Path
import tomllib
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class VaultConfig:
    path: str = "./vault"
    ignored_dirs: list[str] = field(default_factory=lambda: [".obsidian", ".trash", ".git", "Templates"])

@dataclass
class CognitionConfig:
    model_llm_local: str = "qwen2.5:3b"
    model_embeddings_local: str = "nomic-embed-text"
    allow_remote: bool = False

@dataclass
class MemoryConfig:
    db_path: str = "grimoire.db"

@dataclass
class OutputConfig:
    auto_commit: bool = True
    dry_run: bool = True

@dataclass
class Config:
    vault: VaultConfig = field(default_factory=VaultConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

def load_config(config_path: str = "grimoire.toml") -> Config:
    path = Path(config_path)
    if not path.exists():
        return Config()
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    # Simple manual mapping since we're not using pydantic
    vault_data = data.get("vault", {})
    cognition_data = data.get("cognition", {})
    memory_data = data.get("memory", {})
    output_data = data.get("output", {})
    
    return Config(
        vault=VaultConfig(**vault_data),
        cognition=CognitionConfig(**cognition_data),
        memory=MemoryConfig(**memory_data),
        output=OutputConfig(**output_data)
    )
