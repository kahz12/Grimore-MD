<div align="center">

```text
 ________  ________  ___  _____ ______   ________  ___  ________  _______
|\   ____\|\   __  \|\  \|\   _ \  _   \|\   __  \|\  \|\   __  \|\  ___ \
\ \  \___|\ \  \|\  \ \  \ \  \\\__\ \  \ \  \|\  \ \  \ \  \|\  \ \   __/|
 \ \  \  __\ \   _  _\ \  \ \  \\|__| \  \ \  \\\  \ \  \ \   _  _\ \  \_|/__
  \ \  \|\  \ \  \\  \\ \  \ \  \    \ \  \ \  \\\  \ \  \ \  \\  \\ \  \_|\ \
   \ \_______\ \__\\ _\\ \__\ \__\    \ \__\ \_______\ \__\ \__\\ _\\ \_______\
    \|_______|\|__|\|__|\|__|\|__|     \|__|\|_______|\|__|\|__|\|__|\|_______|
```

**An automated knowledge engine for your Markdown vault**

[![Version](https://img.shields.io/badge/version-2.0-6B4BCB?style=for-the-badge)](#) [![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Local-First](https://img.shields.io/badge/Privacy-Local--First-2EA043?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/) [![License](https://img.shields.io/badge/License-MIT-E2B100?style=for-the-badge)](LICENSE)

*Sense the vault · surface connections · wake the Oracle*

</div>

---

Grimoire watches your Markdown vault, auto-tags every note, builds a semantic index, and lets you query your own knowledge base — entirely through local LLMs. Nothing leaves your machine, and no API keys are required.

## Principles

| | |
| :--- | :--- |
| **Sovereignty** | All inference runs through [Ollama](https://ollama.com). No third parties, no telemetry. |
| **Idempotent**  | Every note is SHA-256 hashed; unchanged notes cost zero cycles on re-scan. |
| **Reversible**  | Git Guard snapshots the vault before every write; `--dry-run` is the default. |
| **Non-intrusive** | Runs as a background daemon; surfaces insights only when they're worth your attention. |

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with `qwen2.5:3b` and `nomic-embed-text` pulled
- A git-initialised Markdown vault

## Installation

```bash
git clone https://github.com/kahz12/Grimore-MD.git
cd Grimore-MD

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

grimoire status
```

## Quick start

```bash
grimoire scan --vault-path /path/to/vault --no-dry-run   # first full pass
grimoire daemon start                                    # watch in background
grimoire ask "What threads through my notes on Heidegger's nihilism?"
```

## What it does

**Ingestion.** Watchdog-driven with a 45 s debounce so files are processed once you've stopped typing. Frontmatter, formatting and comments are preserved on write-back; only content-changed notes are reprocessed.

**Cognition.** A local LLM tags, summarises and files each note into one hierarchical category. Tags are normalised and reconciled against an optional controlled vocabulary. A five-failure circuit breaker opens a 120 s cooldown instead of thrashing Ollama.

**Memory.** SQLite in WAL mode with an FTS5 full-text index sitting beside per-chunk vector embeddings — cached by `sha256(model || chunk)` so swapping embedders invalidates cleanly.

**Retrieval.** Cosine similarity and BM25 are fused via Reciprocal Rank Fusion — exact terms and fuzzy concepts both land. Degrades to BM25-only if the embedder is down, or vector-only if FTS5 isn't available.

**Synthesis.** `connect` maintains an idempotent `## Suggested Connections` block of wikilinks in each note. `ask` runs RAG over the hybrid index and returns answers that cite back to your own notes.

## Architecture

```mermaid
graph TD
    A["Markdown Vault"] -->|watchdog| B["Ingest"]
    B -->|SHA-256| C{"Changed?"}
    C -->|no| Z["Skip"]
    C -->|yes| D["Git Guard snapshot"]
    D --> E["Cognition<br/>tags · category · summary<br/>chunks · vectors"]
    E --> F[("SQLite<br/>WAL + FTS5")]
    F --> G["Link Injector"]
    G --> A
    F --> H["Oracle RAG<br/>BM25 + cosine → RRF"]
    H --> I["CLI"]
```

## CLI

| Command | Purpose |
| :--- | :--- |
| `grimoire scan` | Walk the vault, tag changed notes, refresh embeddings. |
| `grimoire connect` | Discover semantic links and inject wikilinks. |
| `grimoire ask <q>` | Query the Oracle (RAG) with citations. |
| `grimoire tags` | Frequency table of tags currently in use. |
| `grimoire category <sub>` | `list` · `add` · `rm` · `notes`. |
| `grimoire daemon <sub>` | `run` · `start` · `stop` · `status`. |
| `grimoire maintenance run` | Ad-hoc VACUUM, WAL checkpoint, tag purge. |
| `grimoire prune` | Drop DB entries for notes gone from disk. |
| `grimoire preflight` | Validate config, Ollama and vault access. |
| `grimoire status` | Dashboard of vault, cognition and daemon. |

Run `grimoire <cmd> --help` for flags.

## Configuration

`grimoire.toml` lives at the project root.

| Section | Key | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `vault` | `path` | `./vault` | Root of your Markdown notes. |
| `vault` | `ignored_dirs` | `[".obsidian", ".trash", ".git", "Templates"]` | Skipped during scan and watch. |
| `cognition` | `model_llm_local` | `qwen2.5:3b` | Ollama model for tagging and the Oracle. |
| `cognition` | `model_embeddings_local` | `nomic-embed-text` | Ollama model for semantic vectors. |
| `cognition` | `allow_remote` | `false` | Required for non-loopback Ollama hosts. |
| `cognition` | `hybrid_search` | `true` | Fuse BM25 and cosine via RRF at query time. |
| `cognition` | `rrf_k` | `60` | RRF constant — higher flattens the rank-weight curve. |
| `cognition` | `connect_threshold` | `0.7` | Minimum cosine score for `connect` to propose a link. |
| `memory` | `db_path` | `grimoire.db` | SQLite file. |
| `output` | `auto_commit` | `true` | Git Guard pre-change snapshots. |
| `output` | `dry_run` | `true` | Prevents writes until explicitly disabled. |
| `maintenance` | `enabled` | `true` | Master switch for daemon housekeeping. |
| `maintenance` | `interval_hours` | `24` | How often housekeeping runs. |
| `maintenance` | `vacuum` | `true` | Reclaim free pages on disk. |
| `maintenance` | `purge_tags` | `true` | Drop tag rows no longer referenced by any note. |
| `maintenance` | `wal_checkpoint` | `true` | Fold the `-wal` sidecar into the main DB file. |

## Taxonomy

Drop a `taxonomy.yml` at the root of your vault to pin your tag vocabulary and category tree:

```yaml
vocabulary:
  - philosophy
  - classical-occultism
  - nihilism

categories:
  History:
    - Ancient
    - Modern
  Science:
    Physics:
      - Quantum
    - Biology
  Art: []
```

Tags are normalised (`"Classical Occultism"` → `classical-occultism`) and rewritten to the canonical form when known; unknown tags are kept verbatim. Categories are the hierarchical counterpart — one canonical path per note (`Science/Physics/Quantum`). The LLM picks from the live menu; unknown paths are rejected. Input resolution is accent- and case-insensitive (`"science / physics"` → `"Science/Physics"`).

Missing or malformed files fall back to sensible defaults — ingestion is never blocked.

## Privacy & safety

- **Per-note opt-out** — `privacy: never_process` in the frontmatter excludes a note from cognition entirely.
- **PII detection** — API keys, emails, IPs and SSH keys are flagged in logs before any LLM call.
- **Prompt-injection hardening** — role markers in note content are neutralised before reaching the LLM.
- **Git Guard** — every mutation is preceded by an auto-commit, so `git reflog` is always your undo.
- **Rolling backups** — the daemon snapshots the SQLite DB daily, keeping the last five under `backups/`.

### Backup hygiene

Backup files under `backups/` are **raw, unencrypted copies of your SQLite database**. The DB stores per-chunk `text_content` (the first 500 characters of every embedded chunk), so anything sensitive that ends up inside a note — passwords, tokens, private addresses — also ends up readable inside every backup, even if the daemon never sent that content to a remote LLM.

Defaults that already help:
- `backups/` is created `chmod 0o700` and each `grimoire_*.db` is written `chmod 0o600` (owner-only on POSIX).
- The `.gitignore` shipped with this repo excludes `backups/` from version control.

What you should still do if your vault holds secrets:
- **Encrypt the directory.** Mount `backups/` (or the whole project) on `gocryptfs` / `eCryptfs` / `age`-based wrappers. Grimoire makes no attempt to encrypt at rest — that's the operator's job.
- **Trim retention.** The default keeps the last 5 daily snapshots (`max_backups=5` in `grimoire/utils/backup.py`). Lower it if your threat model treats the DB as toxic waste.
- **Treat the device as the trust boundary.** Full-disk encryption (LUKS, FileVault, Termux's own data partition encryption) covers the at-rest case for the local-first / single-user threat model Grimoire targets.

## Stack

Python 3.11+ · Ollama · SQLite (WAL + FTS5) · Typer + Rich · watchdog · structlog · GitPython

## Roadmap

| Version | Feature | Status |
| :--- | :--- | :--- |
| **2.1** | The Black Mirror — automatic detection of contradictions across your notes | Planned |
| **2.2** | MCP server — query Grimoire from Claude and other MCP clients | Planned |
| **2.3** | Multi-format ingest — PDFs, EPUBs, and web clippings | Planned |

## License

Released under the [MIT License](LICENSE).

<div align="center">

*Build your digital cortex. Own your data. Automate your wisdom.*

</div>
