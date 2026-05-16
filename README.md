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
grimore shell                                           # conversational mode
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
- **Synthesis** — `connect` maintains an idempotent `## Suggested Connections` block of wikilinks; `ask` runs RAG over the hybrid index and cites back to your own notes. `distill` fuses notes that share a tag or category into a single reference note under `_synthesis/`; `chronicler` flags notes past their freshness window; `mirror` (the Black Mirror) cross-checks claims across notes and surfaces contradictions.

## Shell — conversational mode

`grimore shell` opens an interactive REPL that treats every line as a question to the Oracle, with `/`-prefixed meta-commands and `@note` references for pinning vault context.

```text
❯ what threads through my notes on nihilism?
…streams the Oracle's answer in place…

❯ @camus-revolt how does this connect to absurdism?
…attaches the note "camus-revolt" as priority context for this question…

❯ /pin @nietzsche-gay-science      # rides on every future question
❯ /scan --no-dry-run
About to write frontmatter to every changed note. Continue? [y/N] _

❯ /again                           # re-run the previous question
❯ /why                             # re-print the last answer's sources
❯ /history 20                      # last 20 questions in this session
❯ /save                            # export the transcript as a vault note
❯ /models chat ministral-3:14b
```

- **Slash commands** mirror every CLI verb (`/scan`, `/connect`, `/prune`, `/status`, `/models`, `/category`, `/chronicler`, `/mirror`, `/distill`) plus shell-only helpers (`/again`, `/why`, `/pin`, `/unpin`, `/save`, `/history`). A mistype → "did you mean …?" suggestion via difflib.
- **`@`-mentions** autocomplete vault note titles via fuzzy match (rapidfuzz) and attach the full body as priority context for the next ask. Every resolution is re-validated through `SecurityGuard.resolve_within_vault`, so `@../escape` never leaves the vault.
- **Destructive commands prompt y/N** before running (`/scan --no-dry-run`, `/connect --no-dry-run`, `/prune --no-dry-run`, `/category rm`); pass `--yes` to bypass for scripting.
- **Bottom toolbar** shows live vault · chat model · embed model · dry-run badge · pin count, updating immediately after `/models chat foo`.
- **Multi-line composer**: plain `Enter` submits; `Esc-Enter` or `Alt-Enter` inserts a newline; trailing `\` continues the line.
- **Vi-mode** editing is available via `[shell] vi_mode = true` in `grimore.toml`.

The shell holds a warm `Session` so consecutive `ask`s skip Embedder + LLMRouter cold-start. `/refresh` drops cached services when the vault has changed from another process.

## Commands

| Command | Purpose |
| :--- | :--- |
| `grimore shell` | Conversational REPL (slash commands + `@` mentions). |
| `grimore scan` | Walk the vault, tag changed notes, refresh embeddings. |
| `grimore connect` | Discover semantic links and inject wikilinks. |
| `grimore ask <q>` | One-shot Oracle query with citations. |
| `grimore tags` | Frequency table of tags currently in use. |
| `grimore distill` | Synthesize notes sharing a tag or category into `_synthesis/`. |
| `grimore chronicler <sub>` | `list` · `check` · `verify` — track stale notes. |
| `grimore mirror <sub>` | `scan` · `show` · `dismiss` · `resolve` — surface contradictions. |
| `grimore category <sub>` | `list` · `add` · `rm` · `notes`. |
| `grimore daemon <sub>` | `run` · `start` · `stop` · `status`. |
| `grimore maintenance run` | VACUUM, WAL checkpoint, tag purge. |
| `grimore prune` | Drop DB entries for notes gone from disk. |
| `grimore preflight` | Validate config, Ollama and vault access. |
| `grimore status` | Dashboard of vault, cognition and daemon. |

`grimore <cmd> --help` for flags. Inside the shell, `/help` lists every slash command and `/help <cmd>` shows its usage.

## Configuration

Edit `grimore.toml` at the project root. The shipped defaults are safe — `dry_run = true` blocks writes until you opt in, and `allow_remote = false` pins inference to loopback Ollama. Notable knobs:

- `cognition.model_llm_local` / `model_embeddings_local` — Ollama model names.
- `cognition.request_timeout_s` / `stream_timeout_s` / `embed_timeout_s` — per-call HTTP timeouts; bump these for 14 B-class models on CPU.
- `cognition.hybrid_search` + `rrf_k` — toggle BM25/cosine fusion and tune the rank-weight curve (default 60).
- `cognition.connect_threshold` — minimum cosine score for `connect` to suggest a wikilink (default 0.7).
- `maintenance.interval_hours` — daemon housekeeping cadence (default 24).
- `vault.ignored_dirs` — directories skipped by both scan and watch.
- `vault.display_name` — optional human label shown in the shell's bottom toolbar (falls back to the directory name).
- `shell.vi_mode` — enable vi-style modal editing in the shell (default `false`).
- `shell.fuzzy_threshold` — minimum rapidfuzz score (0–100) for `@`-mention completions (default 55).

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

Python 3.11+ · Ollama · SQLite (WAL + FTS5) · Typer + Rich · prompt-toolkit · rapidfuzz · watchdog · structlog · GitPython · platformdirs

## License

Released under the [MIT License](LICENSE).
