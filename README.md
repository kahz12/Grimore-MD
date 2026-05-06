<div align="center">

```text
 ____    ____    ______            _____   ____    ____
/\  _`\ /\  _`\ /\__  _\   /'\_/`\/\  __`\/\  _`\ /\  _`\
\ \ \L\_\ \ \L\ \/_/\ \/  /\      \ \ \/\ \ \ \L\ \ \ \L\_\
 \ \ \L_L\ \ ,  /  \ \ \  \ \ \__\ \ \ \ \ \ \ ,  /\ \  _\L
  \ \ \/, \ \ \\ \  \_\ \__\ \ \_/\ \ \ \_\ \ \ \\ \\ \ \L\ \
   \ \____/\ \_\ \_\/\_____\\ \_\\ \_\ \_____\ \_\ \_\ \____/
    \/___/  \/_/\/ /\/_____/ \/_/ \/_/\/_____/\/_/\/ /\/___/
```

**An automated knowledge engine for your Markdown vault**

[![Version](https://img.shields.io/badge/version-2.0-6B4BCB?style=for-the-badge)](#) [![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Local-First](https://img.shields.io/badge/Privacy-Local--First-2EA043?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/) [![License](https://img.shields.io/badge/License-MIT-E2B100?style=for-the-badge)](LICENSE)

</div>

---

Grimore watches your Markdown vault, auto-tags every note, builds a hybrid semantic index, and answers questions against it — entirely through local LLMs. Nothing leaves your machine, and no API keys are required.

## Quick start

Requires Python 3.11+, [Ollama](https://ollama.com) with `qwen2.5:3b` and `nomic-embed-text` pulled, and a git-initialised vault.

```bash
git clone https://github.com/kahz12/Grimore-MD.git
cd Grimore-MD
python -m venv .venv && source .venv/bin/activate
pip install -e .

grimore preflight                                       # validate config + Ollama
grimore scan --vault-path /path/to/vault --no-dry-run   # first full pass
grimore daemon start                                    # keep the index live
grimore ask "What threads through my notes on nihilism?"
```

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

- **Ingest** — watchdog observer with a 45 s debounce; SHA-256 idempotency means unchanged notes cost nothing on re-scan.
- **Cognition** — local LLM tags, summarises and files each note into one hierarchical category. A 5-failure circuit breaker opens a 120 s cooldown rather than thrashing Ollama.
- **Memory** — SQLite in WAL mode with FTS5 alongside per-chunk vectors keyed by `sha256(model ‖ chunk)`, so swapping embedders invalidates cleanly.
- **Retrieval** — BM25 and cosine fused via Reciprocal Rank Fusion. Degrades to either side alone if the other is unavailable.
- **Synthesis** — `connect` maintains an idempotent `## Suggested Connections` block of wikilinks; `ask` runs RAG over the hybrid index and cites back to your own notes.

## Commands

| Command | Purpose |
| :--- | :--- |
| `grimore scan` | Walk the vault, tag changed notes, refresh embeddings. |
| `grimore connect` | Discover semantic links and inject wikilinks. |
| `grimore ask <q>` | Query the Oracle with citations. |
| `grimore tags` | Frequency table of tags currently in use. |
| `grimore category <sub>` | `list` · `add` · `rm` · `notes`. |
| `grimore daemon <sub>` | `run` · `start` · `stop` · `status`. |
| `grimore maintenance run` | VACUUM, WAL checkpoint, tag purge. |
| `grimore prune` | Drop DB entries for notes gone from disk. |
| `grimore preflight` | Validate config, Ollama and vault access. |
| `grimore status` | Dashboard of vault, cognition and daemon. |

`grimore <cmd> --help` for flags.

## Configuration

Edit `grimore.toml` at the project root. The shipped defaults are safe — `dry_run = true` blocks writes until you opt in, and `allow_remote = false` pins inference to loopback Ollama. Notable knobs:

- `cognition.model_llm_local` / `model_embeddings_local` — Ollama model names.
- `cognition.hybrid_search` + `rrf_k` — toggle BM25/cosine fusion and tune the rank-weight curve (default 60).
- `cognition.connect_threshold` — minimum cosine score for `connect` to suggest a wikilink (default 0.7).
- `maintenance.interval_hours` — daemon housekeeping cadence (default 24).
- `vault.ignored_dirs` — directories skipped by both scan and watch.

## Taxonomy

Drop a `taxonomy.yml` at the root of your vault to pin a controlled vocabulary and category tree:

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
```

Tags are normalised (`"Classical Occultism"` → `classical-occultism`) and rewritten to canonical form when known; unknown tags pass through. Categories produce one canonical hierarchical path per note. Resolution is accent- and case-insensitive. Missing or malformed files fall back to defaults — ingest never blocks.

## Privacy & safety

- **Per-note opt-out** — `privacy: never_process` in frontmatter excludes a note from cognition entirely.
- **PII detection** — API keys, emails, IPs and SSH keys are flagged in logs before any LLM call.
- **Prompt-injection hardening** — role markers inside note content are neutralised before reaching the LLM.
- **Git Guard** — every mutation is preceded by an auto-commit; `git reflog` is your undo.
- **Daily backups** — the daemon snapshots SQLite and keeps the last five under `backups/` (`chmod 0700`/`0600`).

> `backups/` holds **raw, unencrypted SQLite copies** that include the first 500 characters of every embedded chunk. If your vault carries secrets, mount it on an encrypted FS (`gocryptfs`, `age`, LUKS). Grimore deliberately does not encrypt at rest — full-disk encryption is the right boundary for a single-user, local-first tool.

## Stack

Python 3.11+ · Ollama · SQLite (WAL + FTS5) · Typer + Rich · watchdog · structlog · GitPython · platformdirs

## License

Released under the [MIT License](LICENSE).
