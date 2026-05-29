# Grimore — User Guide (English)

> A local-first knowledge engine for document vaults — Markdown, PDF,
> EPUB, DOCX, ODT, RTF, HTML, TXT — all handled by the same pipeline.
> Everything runs against a loopback Ollama; no API keys, no telemetry, no cloud.

---

## Table of contents

1. [What Grimore is (and what it isn't)](#1-what-grimore-is-and-what-it-isnt)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Your first vault — the five-minute tour](#4-your-first-vault--the-five-minute-tour)
5. [The `grimore.toml` file](#5-the-grimoretoml-file)
   - [Supported formats](#supported-formats)
   - [Sidecars: how non-MD metadata is stored](#sidecars-how-non-md-metadata-is-stored)
   - [Opt-in engines: PDF backends, OCR, magic-byte sniffer](#opt-in-engines-pdf-backends-ocr-magic-byte-sniffer)
   - [Multi-vault profiles](#multi-vault-profiles)
6. [Day-to-day commands](#6-day-to-day-commands)
   - [`scan`](#scan)
   - [`connect`](#connect)
   - [`ask`](#ask)
   - [`eval`](#eval)
   - [`tags`](#tags)
   - [`prune`](#prune)
   - [`status`](#status)
   - [`preflight`](#preflight)
   - [`daemon`](#daemon)
   - [`maintenance run`](#maintenance-run)
   - [`migrate-embeddings`](#migrate-embeddings)
   - [`category`](#category)
   - [`chronicler`](#chronicler)
   - [`mirror`](#mirror)
   - [`distill`](#distill)
   - [`graph export`](#graph-export)
   - [`mcp`](#mcp)
   - [`serve`](#serve)
7. [The interactive shell — `grimore shell`](#7-the-interactive-shell--grimore-shell)
   - [Composing input](#composing-input)
   - [Slash commands](#slash-commands)
   - [`@`-mentions](#-mentions)
   - [Pinning notes](#pinning-notes)
   - [Approval prompts](#approval-prompts)
   - [Saving transcripts](#saving-transcripts)
   - [Conversation persistence — `/thread`, `/resume`, `/threads`](#conversation-persistence--thread-resume-threads)
   - [Conversation memory & `/forget`](#conversation-memory--forget)
   - [Switching models live](#switching-models-live)
   - [Bottom toolbar](#bottom-toolbar)
   - [Vi-mode](#vi-mode)
8. [Taxonomy: `taxonomy.yml`](#8-taxonomy-taxonomyyml)
9. [Frontmatter conventions](#9-frontmatter-conventions)
10. [Privacy & safety](#10-privacy--safety)
11. [Integrations — MCP, HTTP API, OpenAI-compatible backends](#11-integrations--mcp-http-api-openai-compatible-backends)
12. [Working with bigger models](#12-working-with-bigger-models)
13. [Troubleshooting](#13-troubleshooting)
14. [Glossary](#14-glossary)

---

## 1. What Grimore is (and what it isn't)

Grimore watches a directory of documents (your **vault**) and turns it
into a queryable knowledge base. Markdown is the first-class citizen, but
**PDF, EPUB, DOCX, ODT, RTF, HTML and TXT** are extracted by the same
pipeline. For every document it:

- extracts **tags** and a one-paragraph **summary** with a local LLM,
- files it under one **category** in a hierarchical tree,
- splits the body into **chunks**, embeds each chunk into a **vector**,
- proposes **wikilinks** between semantically related notes,
- answers questions in natural language with **citations back to your own notes**.

Everything sits on top of an **SQLite** database with WAL mode and an **FTS5**
index. Retrieval fuses **BM25** (full-text) and **cosine similarity** (vectors)
via **Reciprocal Rank Fusion** (RRF).

**What Grimore is not:**

- It is not a note editor. Use Obsidian / Logseq / your text editor.
- It is not a cloud sync service. Pair it with `git`, Syncthing, or whatever
  you already use.
- It is not an "AI agent" that edits your notes autonomously. Every
  destructive operation defaults to **dry-run**.

---

## 2. Requirements

| Component | Minimum | Notes |
| :--- | :--- | :--- |
| Python | 3.11 | Type-hint syntax used throughout. |
| Ollama | Latest stable | Listening on `127.0.0.1:11434`. |
| Disk | ~1.5 GB per small embedding model | Models live under `~/.ollama`. |
| RAM | 8 GB | For `qwen2.5:3b` + `nomic-embed-text`. 14B-class models want 16 GB+. |
| `git` | Any modern version | Required for the safety-snapshot net. |

Supported OSes: **Linux** and **Windows** are the primary targets. **Termux on
Android** is an alternative supported environment.

Recommended chat models (local):

- `qwen2.5:3b` — the documented default; fast and well-behaved on CPU.
- `qwen3.5:0.8b` — even faster, but emits a "thinking" phase the Oracle has
  to wait through before tokens appear.
- `ministral-3:14b` — slower but stronger reasoning. Bump
  `cognition.request_timeout_s` to 180+ before using.

Recommended embedding model:

- `nomic-embed-text` — the documented default.
- `embeddinggemma:latest` — an alternative if you want Google's encoder.

---

## 3. Installation

```bash
git clone https://github.com/kahz12/Grimore-MD.git
cd Grimore-MD
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Confirm everything is wired up:

```bash
grimore preflight
```

`preflight` checks four things, in order:

1. The config file is loadable.
2. Ollama is reachable.
3. The chat and embedding models you configured are actually pulled.
4. The vault path exists, and (unless `--no-check-git`) is a git repository.

Any blocking error exits with code 1 and a panel telling you exactly which
line to look at.

---

## 4. Your first vault — the five-minute tour

Assume your notes live under `~/notes`.

```bash
cd ~/notes
git init                                            # safety net
echo "vault.path = \"~/notes\"" > grimore.toml      # see §5 for the full file

grimore preflight                                   # green panel = ready
grimore scan --dry-run                              # preview what would be tagged
grimore scan --no-dry-run                           # actually write tags + index
grimore connect --no-dry-run                        # add wikilinks
grimore ask "what threads through my notes on memory?"
```

You now have a tagged, embedded, link-injected vault and you've asked your
first Oracle question. From here you can either keep using the one-shot
commands or open the **interactive shell** (§7) and stay inside it:

```bash
grimore shell
```

---

## 5. The `grimore.toml` file

Grimore looks for `grimore.toml` in the current working directory. The shipped
defaults are safe: nothing leaves your machine, and write-mode is opt-in. A
complete annotated example:

```toml
[vault]
path = "./vault"
ignored_dirs = [".obsidian", ".trash", ".git", "Templates"]
# Document formats Grimore will pick up, by lowercase extension. The
# default tracks every adapter shipped with Grimore that uses pure-Python
# or stdlib-only extraction. Add "doc" if you have antiword installed.
formats = ["md", "txt", "html", "htm", "docx", "pdf", "epub", "rtf", "odt"]
# Hidden root, relative to the vault, where Grimore mirrors a `.md`
# sidecar for every non-Markdown document. The original binaries are
# never mutated; tags, summaries and suggested-link blocks live here.
sidecar_dir   = ".grimore/sidecars"
# Set false to keep all non-Markdown metadata in the DB only (no
# sidecars on disk).
write_sidecars = true
# Optional label shown in the shell's bottom toolbar. Defaults to the
# directory name when unset.
display_name = "Library"

[cognition]
model_llm_local       = "qwen2.5:3b"
model_embeddings_local = "nomic-embed-text"
allow_remote          = false   # block non-loopback Ollama / OpenAI endpoints
hybrid_search         = true    # BM25 + cosine via RRF
rrf_k                 = 60      # RRF rank-weight; lower = steeper
connect_threshold     = 0.7     # cosine floor for a suggested wikilink
request_timeout_s     = 60      # /api/generate, JSON path
stream_timeout_s      = 120     # /api/generate, streaming path
embed_timeout_s       = 30      # /api/embeddings
# Second-stage re-rank after RRF fusion. Off by default — it costs a
# pass over `rerank_pool` candidates per query. The "llm" engine asks
# the local chat model to rate each candidate (no extra install,
# ~15–30 s/query); "cross-encoder" uses a sentence-transformers
# reranker (sub-second; needs the `reranker` extra).
rerank          = false
rerank_pool     = 20
rerank_engine   = "llm"          # "llm" | "cross-encoder"
rerank_model    = "BAAI/bge-reranker-base"
# Vector backend. "auto" (default) uses sqlite-vec when the extension
# is installed and the index dimension matches, else falls back to
# numpy matmul. Force "numpy" to pin the in-memory path; "sqlite-vec"
# requires the optional extra and the wheel to load cleanly.
vector_backend  = "auto"
# LLM backend dispatch. "ollama" (default) keeps v2.x behaviour. Set
# to "openai" to talk to any OpenAI-compatible server (llama.cpp,
# vLLM, LM Studio, OpenRouter, OpenAI). See §11 for full details.
llm_backend     = "ollama"
llm_base_url    = ""              # e.g. "http://localhost:8080/v1" for llama.cpp server
llm_api_key_env = "GRIMORE_LLM_API_KEY"   # env var holding the bearer token

[ingest]
# PDF engine. "pypdf" is the always-available default; "pdfplumber" and
# "pymupdf" are opt-in extras with their own install hints in preflight.
# PyMuPDF is AGPL — verify licence compatibility before switching.
pdf_engine    = "pypdf"
# OCR fallback for scanned PDFs (image-only pages). Off by default;
# requires the `ocr` extra and the `tesseract` binary on PATH.
ocr           = false
ocr_timeout_s = 30
# Magic-byte sniffer for misnamed / extension-less files. Off by
# default; requires the `sniff` extra (python-magic) and libmagic.
sniff_magic   = false
# Chunker engine. "markdown" (default) splits on headings + size cap,
# deterministic and free at scan time. "semantic" embeds each
# sentence and splits on topic shifts — better recall on long-form
# prose at the cost of one embedding per sentence during scan.
chunker            = "markdown"
semantic_threshold = 0.55     # cosine drop that triggers a new chunk
chunk_max_chars    = 1500     # hard cap regardless of engine

[memory]
db_path = "grimore.db"

[output]
auto_commit = true  # git snapshot before every write
dry_run     = true  # safe default — opt in with --no-dry-run

[maintenance]
enabled         = true
interval_hours  = 24
vacuum          = true
purge_tags      = true
wal_checkpoint  = true

[chronicler.windows]
"tech/"        = 90
"tools/"       = 90
"infra/"       = 90
"dev/"         = 180
"code-snippets/" = 180
"concepts/"    = 0    # 0 = never stale
"theory/"      = 0
"journal/"     = 0
"daily/"       = 0

[daemon]
enabled         = false
log_events      = true
debounce_seconds = 45    # batch rapid editor saves into one re-index
poll_fallback    = false # use polling instead of inotify (NFS, FUSE, Termux)
poll_interval_s  = 30.0

[shell]
vi_mode         = false  # set true for prompt_toolkit vi-mode
fuzzy_threshold = 55     # 0–100, rapidfuzz score floor for @-completions
# Where /thread save writes conversation transcripts. Relative paths
# anchor under your home directory, so the default lives at
# ~/.grimore/threads. Absolute paths are honoured as-is.
threads_dir     = ".grimore/threads"

# Optional: named vault profiles. Each [profiles.<name>] block
# deep-merges over the top-level sections when activated by
# `--profile <name>` or `GRIMORE_PROFILE=<name>`. See §5 → "Multi-vault profiles".
# [profiles.work]
# [profiles.work.vault]
# path = "/home/me/work-notes"
# [profiles.work.cognition]
# model_llm_local = "qwen2.5:14b"
```

Unknown keys are logged at WARNING and ignored — copy-pasting old config
snippets won't crash the CLI.

### Supported formats

| Extension | Adapter           | Notes                                                                    |
|-----------|-------------------|--------------------------------------------------------------------------|
| `md`      | Markdown          | First-class. Frontmatter, wikilinks and inline writeback.                |
| `txt`     | Plain text        | Stdlib only. First non-blank line becomes the title fallback.            |
| `html`    | HTML / XHTML      | `<title>`, then first `<h1>` / `<h2>`. Needs `beautifulsoup4`.           |
| `htm`     | HTML (alias)      | Same adapter as `.html`.                                                 |
| `docx`    | DOCX (OOXML)      | Pure-stdlib: `zipfile` + `xml.etree`. Headings → sections.               |
| `pdf`     | PDF               | Per-page sections; page anchors flow through to citations as `#p.N`.     |
| `epub`    | EPUB              | Spine-order chapters → sections; titles from OPF metadata then nav.      |
| `rtf`     | RTF               | `striprtf`; one anchor-free section.                                     |
| `odt`     | OpenDocument Text | Pure-stdlib zip+XML. Mirrors the DOCX flow for ODT outlines.             |
| `doc`     | Legacy `.doc`     | **Opt-in.** Add `"doc"` to `formats` and install `antiword` on PATH.     |

For paginated formats, citations gain an anchor automatically — an
answer drawn from page 137 of *Designing Data-Intensive Applications*
renders as `[[Designing Data-Intensive Applications#p.137]]`. For
structured formats, the anchor falls back to the enclosing heading.

### Sidecars: how non-MD metadata is stored

Grimore never mutates a PDF / EPUB / DOCX / ODT / RTF / HTML / TXT
source. Instead, the cognition layer writes a `.md` **sidecar** under
`<vault>/<sidecar_dir>/` (default: `.grimore/sidecars/`) that mirrors
the source path:

```
vault/
  Books/Designing-Data-Intensive-Applications.pdf
  .grimore/
    sidecars/
      Books/Designing-Data-Intensive-Applications.pdf.md
```

The sidecar carries frontmatter (`tags`, `summary`, `category`,
`last_tagged`) and — once `grimore connect` has run — a `## Suggested
Connections` block. The original `.pdf` extension is preserved in the
sidecar name so `Foo.pdf` and `Foo.epub` never collide on a single
`Foo.md`. Set `write_sidecars = false` to keep all of this DB-only.

`@`-mentions in the shell are sidecar-aware: typing `@my-book` resolves
to the binary source, but the attachment Grimore feeds the model comes
from the sidecar's clean extracted text.

### Opt-in engines: PDF backends, OCR, magic-byte sniffer

These knobs live under `[ingest]` and stay off until you set them.
Preflight (`grimore preflight`) only probes the ones you've enabled,
so a default install reports a clean ✓ row per format.

- **Alternative PDF engines.** `pdfplumber` handles columns and tables
  better than `pypdf`; `pymupdf` (AGPL — verify licence compatibility)
  has the best extraction quality. Install the matching extra:

  ```bash
  pip install 'grimore[pdf-plumber]'
  pip install 'grimore[pdf-mupdf]'   # AGPL-3.0
  ```

  Then set `pdf_engine` to `"pdfplumber"` or `"pymupdf"`. The change
  lands on the next scan without any restart.

- **OCR for scanned PDFs.** Pages with an empty text layer can be
  rasterised and OCR'd via `tesseract`. Two dependencies are needed:
  the Python wheels (`pip install 'grimore[ocr]'`) and the `tesseract`
  binary itself. Then set `ocr = true`. OCR sections are tagged with
  the synthetic heading `(ocr)` so reviewers can audit them.

- **Magic-byte sniffer.** When `sniff_magic = true`, files whose
  extension is missing or doesn't match `formats` are inspected for
  their actual content type (via `libmagic`) and routed to the right
  adapter if one exists. Useful for vaults that hold ad-hoc downloads.
  Install the `sniff` extra and a system `libmagic` (Linux:
  `apt install libmagic1`; Termux: `pkg install file`; Windows:
  `pip install python-magic-bin`).

### Multi-vault profiles

A single `grimore.toml` can carry several named **profiles**. Each
`[profiles.<name>]` block holds overrides that *deep-merge* over the
top-level sections — anything you don't override is inherited. Pick a
profile per invocation with `--profile <name>` (any command), or set
the `GRIMORE_PROFILE` environment variable to make it the default.

```toml
# Top-level keys still apply when no profile is selected.
[vault]
path = "./vault"
display_name = "Personal"

[cognition]
model_llm_local = "qwen2.5:3b"

# A heavier office machine + a different vault path.
[profiles.work]
[profiles.work.vault]
path = "/home/me/work-notes"
display_name = "Work"
[profiles.work.cognition]
model_llm_local = "qwen2.5:14b"
request_timeout_s = 180
```

```bash
grimore --profile work ask "what changed in last week's standups?"
GRIMORE_PROFILE=work grimore shell
```

Precedence: the `--profile`/`-P` flag beats `GRIMORE_PROFILE`, which
beats the unprofiled defaults. An unknown profile name fails fast
with a clear error message instead of silently falling back to the
defaults. `grimore status` shows the active profile name next to the
vault path.

Lists are replaced wholesale, not concatenated — `formats = ["md"]`
inside a profile means *only* Markdown, not "Markdown also".

---

## 6. Day-to-day commands

Run `grimore <cmd> --help` for the authoritative flag list. Below is a
detailed walk-through.

### `scan`

```bash
grimore scan [-p PATH] [--dry-run|--no-dry-run] [--json]
```

Walks the vault, tags new or changed documents in every configured
format, and refreshes the embedding index.

- `-p, --vault-path` — override the vault path for this run only.
- `--dry-run` / `--no-dry-run` — flip without touching the config.
- `--json` — emit JSON-formatted structured logs (handy in CI / monitoring).

**Idempotency.** Two-tier hashing: a cheap SHA-256 of the file bytes
(`file_hash`) gates the expensive extraction step; only documents
whose bytes have changed pay for re-extraction. Within that, an
unchanged content-hash skips the LLM call entirely. Re-scanning a
fully indexed vault is effectively free even for a 500 MB PDF library.

**What gets written.** When run with `--no-dry-run`:

- `tags`, `summary`, `category`, `last_tagged` are written into each note's
  YAML frontmatter,
- the SQLite row is upserted,
- per-chunk embeddings are stored under a key of `sha256(model ‖ chunk)`
  (changing embedding model invalidates the cache cleanly).

**Privacy escape hatch.** A note with `privacy: never_process` in its
frontmatter is skipped before the LLM is invoked.

### `connect`

```bash
grimore connect [--dry-run|--no-dry-run] [-t THRESHOLD]
```

Walks every note and finds semantically similar siblings via cosine
similarity. When run with `--no-dry-run`, it idempotently maintains a
`## Suggested Connections` block at the end of each note containing
wikilinks to its strongest matches.

- `-t, --threshold` — cosine floor, in `[0.0, 1.0]`. Default comes from
  `cognition.connect_threshold` (0.7). Lower = more (noisier) suggestions.

The block is regenerated, not appended-to — running `connect` repeatedly is
safe.

### `ask`

```bash
grimore ask "<question>" [-k N] [-e PATH]
```

A one-shot Retrieval-Augmented Generation query.

- `-k, --top-k` — how many context chunks to retrieve (default 5). More
  chunks = richer answer, but a slower model.
- `-e, --export PATH` — render the answer + sources as a markdown note at
  `PATH` instead of streaming to stdout.

The Oracle always cites the documents it pulled context from. Citations
appear at the end of the answer as `[[title]]` links — and for
paginated or structured formats they gain an anchor:
`[[Designing Data-Intensive Applications#p.137]]` for PDFs,
`[[Annual Report#Chapter 3]]` for DOCX, EPUB and ODT. Markdown and TXT
citations stay anchor-free.

Hallucinated citations (titles the model invents that weren't in the
retrieved context) are stripped from the rendered answer and logged at
`oracle_citation_hallucinated`. The returned source list always
reflects the *retrieved* notes, not whatever the model claimed.

### `eval`

```bash
grimore eval [-g PATH] [-k N] [--judge/--no-judge] [--export PATH] [--json]
```

Runs a golden Q&A set against the Oracle and reports retrieval and
answer-quality metrics. Defaults to `eval/grimore_golden.yaml`.

| Metric | What it measures |
|---|---|
| **recall@k** | Fraction of expected sources that appear in the top-k retrieved set. |
| **MRR** | Mean reciprocal rank of the first expected source hit. |
| **faithfulness** | 1 − dropped/total citations (1.0 = no hallucinated citations). |
| **keyword recall** | Fraction of expected keywords present in the answer body. |
| **answer relevance** | LLM-as-judge: local model rates the answer 0–10 vs the question. |
| **p50 / p95 latency** | Wall-clock per turn. |

- `-g, --golden` — path to the YAML golden set.
- `-k, --top-k` — passages retrieved per question (default 5).
- `--no-judge` — skip the LLM-as-judge pass (handy offline / fast CI).
- `--export` — dump the full report as JSON for downstream tooling.
- `--json` — JSON-formatted structured logs.

The golden set format is one entry per Q&A item with optional
`follow_ups` so conversation memory (`Oracle.history`) is exercised
too. Schema version is pinned; unknown keys raise so typos surface
immediately. See `eval/grimore_golden.yaml` for a working example.

### `tags`

```bash
grimore tags [-n N]
```

Frequency table of every tag in use. `-n` caps the rows shown (default 30).

### `prune`

```bash
grimore prune [-p PATH] [--dry-run|--no-dry-run]
```

Removes DB rows for notes that have disappeared from disk, then purges any
tag rows nobody references anymore. **Dry-run by default** — pass
`--no-dry-run` to actually delete.

### `status`

```bash
grimore status
```

A dashboard showing vault path, cognition models, DB size, daemon status,
last scan time, and similar at-a-glance data. Equivalent to `/status`
inside the shell.

### `preflight`

```bash
grimore preflight [--check-git|--no-check-git]
```

Validates config, Ollama connectivity (with model-presence checks), and
vault access. By default it requires a git repo iff `output.auto_commit`
is true — override with the flag.

### `daemon`

```bash
grimore daemon run         # foreground, Ctrl-C to stop
grimore daemon start       # background, PID + log under platform cache dir
grimore daemon stop
grimore daemon status      # shows up/down badge
```

The daemon uses `watchdog` to observe vault changes; a 45-second debounce
batches saves so a rapid editor flurry doesn't fire the LLM five times. It
also runs `[maintenance]` on the configured cadence.

The PID/log paths are picked via `platformdirs`, so foreground and
background invocations always agree on where state lives.

### `maintenance run`

```bash
grimore maintenance run [--skip-vacuum] [--skip-purge] [--skip-checkpoint]
```

Runs the housekeeping pipeline once, immediately. Reports how many tags
were purged, how many WAL frames were checkpointed, and how many bytes
VACUUM reclaimed. Each `--skip-*` flag turns the matching step off for
this one run only (config defaults are restored next time).

### `migrate-embeddings`

```bash
grimore migrate-embeddings <new-model> [--status] [--abort] [--write-config/--no-write-config]
```

Hot-swap the embedding model without taking the vault offline. The
command builds a shadow `embeddings_migration` table, re-embeds every
chunk against the new model, then atomically replaces the live table
in a single transaction. While the migration runs, search keeps
serving from the old vectors.

- `--status` prints the resume point of an in-flight migration.
- `--abort` clears the shadow table; the live vectors are untouched.
- `--write-config` (default) rewrites `[cognition].model_embeddings_local`
  on success so the next process picks up the new model.

Resumable: re-running the command with the same target picks up at
the row the worker last finished. A preflight refuses to start if
disk free space is less than 2 × current embeddings size.

### `category`

```bash
grimore category list
grimore category add <path>
grimore category rm  <path> [-f|--force]
grimore category notes <path> [--flat]
```

Maintains the hierarchical category tree in `taxonomy.yml`.

- `add` accepts a slash-separated path — missing ancestors are created.
- `rm` refuses to delete a category that still has assigned notes unless
  you pass `--force`. With `--force`, the category vanishes from the tree
  but the notes keep their (now dangling) `category:` field until the next
  scan rewrites it.
- `notes --flat` lists only the notes directly under the path; without it,
  descendants are included.

### `chronicler`

```bash
grimore chronicler list [--decay|--no-decay]
grimore chronicler check <path>
grimore chronicler verify <path>
```

Temporal staleness tracking — flags notes that are past their freshness
window (defined per category prefix in `[chronicler.windows]`).

- `list` shows everything past its window. `--decay` runs the cached LLM
  decay verdict on each row.
- `check <path>` runs the decay LLM against a single note now (slower, but
  current).
- `verify <path>` resets the freshness clock — useful when you've re-read
  a note and confirmed it's still accurate.

### `mirror`

```bash
grimore mirror                                 # list open contradictions
grimore mirror scan [-k N] [--full]            # extract + check
grimore mirror show <id>                       # render one in detail
grimore mirror dismiss <id>                    # mark as not-a-contradiction
grimore mirror resolve <id>                    # mark as resolved
```

The **Black Mirror** extracts atomic claims from each note, finds claim
pairs across notes, and asks the LLM whether they contradict each other.
The default `--top-k 5` looks at each claim's five nearest neighbours;
`--full` re-extracts every note from scratch (slow — cold rebuild).

### `distill`

```bash
grimore distill --tag <name>      [-p N] [--dry-run]
grimore distill --category <path> [-p N] [--dry-run]
```

Synthesises every note carrying the given tag (or filed under the given
category, recursively) into a single reference note under `_synthesis/`.
The output gets `grimore_generated: true` in its frontmatter so later
`distill` runs don't re-include themselves.

- `-p, --passages` — top-K passages per source note (default 3).
- `--dry-run` — build the synthesis but skip the file write.

> **Note:** Unlike `scan`/`connect`/`prune`, the CLI's `distill` writes by
> default. The shell's `/distill` mirrors that, but adds an interactive
> "y/N" approval before any write (see §7).

### `graph export`

```bash
grimore graph export <output> [-f json|dot|obsidian-canvas]
                              [--suggested/--no-suggested]
                              [--suggested-top N]
                              [--suggested-threshold X]
```

Crawls the vault's link graph and writes it in the chosen format.
Three edge kinds are produced:

| Kind | Source |
|---|---|
| `wikilink` | Explicit `[[Title]]` references in Markdown bodies. |
| `suggested` | Top-N cosine neighbours per note (mean-pooled chunk vectors). Filtered by `--suggested-threshold` (default 0.7). |
| `contradicts` | Open or resolved contradiction pairs from the **Black Mirror**, lifted to the note level. Dismissed pairs are excluded. |

Formats:

- **`json`** — `{version, nodes, edges}`. Stable schema, easy to feed
  into other tools.
- **`dot`** — Graphviz source. Render with `dot -Tsvg vault.dot -o vault.svg`.
- **`obsidian-canvas`** — drops the graph into a `.canvas` file that
  Obsidian opens directly, with nodes grouped by category on a grid.

The semantic-neighbour pass is the slow part — pass `--no-suggested`
for a quick survey based on wikilinks + contradictions alone.

### `mcp`

```bash
grimore mcp [--json/--no-json]
```

Spawns Grimore as a stdio Model Context Protocol server so any
MCP-aware client (Claude Desktop, Cursor, Zed, …) can call into the
vault as tools. Read-only by design: `grimore_ask`, `grimore_search`,
`grimore_get_note`, `grimore_connect`, `grimore_list_categories`.
Write operations stay on the CLI so a remote client can never trigger
a destructive op by accident.

See [`docs/mcp-setup.md`](mcp-setup.md) for copy-pasteable client
configs and the working-directory caveat.

### `serve`

```bash
grimore serve [-H HOST] [-P PORT] [--allow-lan] [--api-token TOKEN]
              [--cors-origin ORIGIN]
```

Boots a read-only HTTP API + minimal vanilla-JS browser UI on
`http://127.0.0.1:8000`. Routes:

| Method · Path | Description |
|---|---|
| `GET /api/health` | Version + preflight summary. |
| `POST /api/ask` | `{question, top_k?, stream?}`. SSE streaming when `stream: true`. |
| `POST /api/search` | `{query, top_k?}` → hybrid hits with snippets. |
| `GET /api/notes/{id}` | Note metadata + on-disk body. |
| `GET /api/categories` | Vault-wide counts. |
| `GET /` | The web UI. |

Security gates baked into the CLI (not just docs):

- Bind to loopback by default. Setting `--host 0.0.0.0` (or any
  non-loopback address) requires both `--allow-lan` and `--api-token`.
- `--api-token` (also read from `GRIMORE_API_TOKEN`) is checked on every
  POST. GETs stay open so the loopback browser flow doesn't have to
  thread credentials into every fetch.
- CORS is off by default. `--cors-origin <origin>` adds exactly one
  origin (no wildcards).

The `serve` extra ships only `starlette` + `uvicorn` so the install
stays Termux-safe (no pydantic-core wheel needed on ARM).

---

## 7. The interactive shell — `grimore shell`

The shell is the recommended day-to-day interface. It keeps a warm
`Session`, so consecutive `ask`s reuse the embedder and the LLM router
without paying cold-start costs.

Open it:

```bash
grimore shell
```

You land on a banner and a `❯ ` prompt. Type a question and press Enter —
no `/ask` prefix needed:

```text
❯ what threads through my notes on nihilism?
…streams the Oracle's answer in place…
[[camus-revolt]] · [[nietzsche-gay-science]] · [[cioran-bitter-cradle]]
```

### Composing input

- **Plain `Enter`** submits.
- **`Esc` then `Enter`** *or* **`Alt+Enter`** inserts a newline (so you can
  ask multi-paragraph questions or paste a block).
- Lines ending in a trailing `\` continue onto the next line and are
  joined into a single logical input before dispatch.
- **`Ctrl+C`** cancels the current input *or* the in-flight question
  without killing the loop.
- **`Ctrl+D`** (EOF) exits the shell cleanly.
- **`Ctrl+R`** (Emacs) opens reverse-history search.

### Slash commands

Every CLI verb has a slash twin, plus a handful of shell-only helpers.
Slash commands tab-complete from a popup as soon as you type `/`.

| Slash | What it does |
| :--- | :--- |
| `/ask` | Explicit ask — usually not needed (just type the question). |
| `/scan`, `/connect`, `/prune` | Mirror the CLI verbs, with approval prompts. |
| `/status`, `/tags`, `/preflight` | Read-only inspection. |
| `/category list \| add \| rm \| notes` | Same as the CLI. |
| `/chronicler list \| check \| verify` | Same as the CLI. |
| `/mirror`, `/mirror scan/show/dismiss/resolve` | Same as the CLI. |
| `/distill` | Same as the CLI, plus an approval prompt on writes. |
| `/again` | Re-ask the previous question with the same flags. |
| `/why` | Re-print the sources cited by the last answer. |
| `/pin @note […]` | Pin notes to every future ask (`/pin` alone lists pins). |
| `/unpin [@note]` | Remove one pin (or all, called bare). |
| `/save [path] [-f]` | Export the session transcript as a vault note. |
| `/history [N]` | Show the last N questions (default 10). |
| `/forget` | Drop conversation memory (turns + last_*). Pins stay. |
| `/thread save\|resume\|list` | Persist / reload conversations across shells. |
| `/resume <name>` | Shortcut for `/thread resume <name>`. |
| `/threads` | Shortcut for `/thread list`. |
| `/models [chat\|embed [name\|idx]]` | List backend models & switch live. |
| `/refresh` | Drop cached services + the @-mention index. |
| `/clear` | Clear the screen. |
| `/help [cmd]` | List commands, or detail one. |
| `/exit`, `/quit` | Leave. |

A mistype like `/scna` triggers a `Did you mean: /scan?` suggestion via
`difflib`.

### `@`-mentions

A token beginning with `@` is resolved against the vault index and the
matching note's full body is attached to the next ask as priority context:

```text
❯ @camus-revolt how does this connect to absurdism?
```

Resolution order:

1. exact path under the vault root (with or without `.md`),
2. exact title match (case-insensitive),
3. `rapidfuzz` best match above the configured threshold.

Every resolution is re-validated through `SecurityGuard.resolve_within_vault`,
so an attempt like `@../escape` is rejected and the literal `@token` stays
in the message text. Tokens that don't resolve trigger a muted "no note
matched" line and are left as plain text in the question.

Attachments are capped at 32,000 characters per note (the embedder's
`EMBED_MAX_CHARS`).

The completer offers fuzzy completion for `@` tokens; the meta-column on
the right shows the match score so you can dial `[shell] fuzzy_threshold`
to taste.

### Pinning notes

`@`-mentions are one-shot. To attach a note to *every* future question in
the session, pin it:

```text
❯ /pin @nietzsche-gay-science
Pinned: [[nietzsche-gay-science]]

❯ /pin                            # list current pins
❯ /unpin @nietzsche-gay-science   # remove one
❯ /unpin                          # drop all
```

The toolbar's `pins: N` segment updates immediately.

### Approval prompts

Any command that writes to the vault prompts before running:

```text
❯ /scan --no-dry-run
scan --no-dry-run will write frontmatter to every changed note.
Continue? [y/N] _
```

Pass `--yes` to bypass — useful for scripting:

```text
❯ /scan --no-dry-run --yes
```

Commands that prompt by default:

- `/scan --no-dry-run`
- `/connect --no-dry-run`
- `/prune --no-dry-run`
- `/category rm`
- `/distill` (when not `--dry-run`)

### Saving transcripts

```text
❯ /save                                 # default path: _transcripts/<ts>.md
❯ /save reflections/oracle-2026-05-20.md
❯ /save reflections/oracle-2026-05-20.md --force   # overwrite an existing file
```

The transcript contains one `Q1.` `Q2.` … heading per question plus the
full body of the most recent answer (with its sources). The path is
re-validated through `SecurityGuard.resolve_within_vault`, so paths that
escape the vault are refused. If the target file already exists, `/save`
refuses to clobber it unless you pass `-f` / `--force`.

### Conversation persistence — `/thread`, `/resume`, `/threads`

`/save` exports a one-shot markdown transcript into the vault. To
persist a *resumable* conversation thread instead, use the `/thread`
namespace. Threads survive shell restarts so a long research session
can be picked up later.

```text
❯ /thread save                                # auto-slug from the first question
❯ /thread save research-on-stoicism           # explicit slug
❯ /thread resume research-on-stoicism         # load it back
❯ /thread list                                # show every saved thread
❯ /threads                                    # shortcut for /thread list
❯ /resume research-on-stoicism                # shortcut for /thread resume …
```

What's stored:

- One JSONL file per thread under the configured `shell.threads_dir`
  (default `~/.grimore/threads/`).
- Each line is one turn: `{ts, q, a, sources}`.
- Atomic writes — the file is written to `<slug>.jsonl.tmp` then
  renamed, so a crash mid-write can't leave a corrupted thread.

Resuming pre-fills `last_question` and `last_answer` from the final
turn so `/again` and `/why` work immediately. The rolling
conversation memory (last 3 turns, see below) is restored from the
file, so follow-ups like "expand on the last point" resolve cleanly.

The slug for a no-arg `/thread save` is derived from the first six
word-runs of the first question; non-ASCII characters drop to ASCII
and punctuation is stripped so the filename stays portable.

### Conversation memory & `/forget`

Inside one shell session, Grimore keeps a rolling memory of the last
three turns (`{q, a, sources}` each). It feeds two places:

- **Query rewrite for retrieval.** Pronouns and references resolve
  against earlier turns ("expand on that" finds the same notes).
- **Answer coherence.** A short "Recent conversation" block precedes
  the retrieved context, so the model stays on topic across turns.

The one-shot CLI's `grimore ask` never populates this memory — its
behaviour is byte-identical to v2.0. Inside the shell, `/forget`
drops the memory plus `last_question` / `last_answer` / the question
log without touching pins or the on-disk shell history file:

```text
❯ /forget
Conversation forgotten — starting fresh.
```

`/refresh` (which rebuilds caches after a scan from another terminal)
also clears the conversation memory implicitly.

### Switching models live

```text
❯ /models                       # list installed Ollama models + current pick
❯ /models chat ministral-3:14b  # switch the chat model (live + persisted)
❯ /models embed embeddinggemma  # switch the embedding model
```

Model swaps persist back into `[cognition]` of `grimore.toml`. The chat
swap is instant; an embedding swap drops the cached embedder so the next
query rebuilds it under the new model.

You can also pass an index instead of a name:

```text
❯ /models chat 2
```

### Bottom toolbar

The persistent footer shows live session state:

```text
vault: Library  •  chat: qwen2.5:3b  •  embed: nomic-embed-text  •  dry-run  •  pins: 2
```

It re-renders every keystroke — `/models` and `/pin` are reflected
immediately.

### Vi-mode

Set `[shell] vi_mode = true` in `grimore.toml` to enable prompt_toolkit's
vi-style modal editing. The prompt glyph shifts from `❯ ` to `∙ ` when
you're in normal mode.

---

## 8. Taxonomy: `taxonomy.yml`

Drop a `taxonomy.yml` at the root of your vault to pin a controlled
vocabulary and a category tree:

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

- **Tag normalisation** — `"Classical Occultism"` → `classical-occultism`.
  Known tags are rewritten to their canonical form; unknown tags pass
  through (the tagger can introduce new ones, but you can also pre-seed
  the vocabulary to bias it).
- **Categories** are mutually exclusive: every note ends up filed under
  exactly one canonical hierarchical path.
- **Resolution is accent- and case-insensitive**: `"física"` and
  `"Fisica"` both resolve to `Science/Física` once that path exists.
- A missing or malformed `taxonomy.yml` falls back to defaults — ingest
  never blocks.

---

## 9. Frontmatter conventions

Grimore reads (and selectively writes) the following YAML frontmatter
fields. Anything you add manually is preserved on round-trip.

```yaml
---
title: "Camus and Revolt"
tags: [philosophy, nihilism, absurdism]
summary: |
  Short paragraph rewritten on every scan.
category: Philosophy/Existentialism
last_tagged: "2026-05-20T10:14:22Z"
privacy: never_process     # optional — excludes from cognition entirely
grimore_generated: true    # written by `distill` outputs; protects them from re-distillation
---
```

- `last_tagged` is a UTC ISO timestamp; the **Chronicler** uses it to
  compute staleness.
- `privacy: never_process` is the only field that's checked *before* any
  LLM call, so it really does keep the note off the wire.
- For **non-Markdown documents** (PDF, EPUB, DOCX, …) the same fields
  live in the sidecar `.md` under `<vault>/<sidecar_dir>/` instead of
  the source file. The original binary is never touched. To set
  `privacy: never_process` for a PDF, drop the frontmatter into its
  sidecar — Grimore preserves manual edits across re-scans.

---

## 10. Privacy & safety

**Local by construction.** With `cognition.allow_remote = false` (the
default), every Ollama call is rejected unless the endpoint resolves to a
loopback address. Flip the flag only if you're knowingly routing to a
trusted LAN box.

**Per-note opt-out.** `privacy: never_process` excludes a note from
cognition entirely (see §9).

**PII detection.** Before any LLM call, the content is scanned for API
keys, emails, IPs, and SSH keys; matches are flagged at WARNING in the
structured log (the call still proceeds — Grimore won't silently drop your
work, but you'll know).

**Prompt-injection hardening.** Role markers (`### System:` and similar)
embedded inside note content are neutralised before reaching the LLM.

**Git Guard.** Every mutation is preceded by an auto-commit. `git reflog`
is your undo.

**Daily backups.** The daemon snapshots SQLite and keeps the last five
under `backups/` with `chmod 0700`/`0600`.

> `backups/` are **raw, unencrypted SQLite copies** that include the first
> 500 characters of every embedded chunk. If your vault contains secrets,
> store it on an encrypted FS (`gocryptfs`, `age`, LUKS). Grimore
> deliberately does not encrypt at rest — full-disk encryption is the
> right boundary for a single-user, local-first tool.

**Path containment.** Every file Grimore touches (notes, transcripts,
`@`-mention attachments) is re-resolved through
`SecurityGuard.resolve_within_vault`. A token like `@../escape` cannot
exfiltrate a file outside the vault, even via symlink.

---

## 11. Integrations — MCP, HTTP API, OpenAI-compatible backends

Three surfaces let Grimore be used outside the terminal.

### MCP server

`grimore mcp` exposes the vault as tools to any Model Context Protocol
client (Claude Desktop, Cursor, Zed). It's a stdio server — your
client spawns `grimore mcp` and talks JSON-RPC on its stdin/stdout.

The MCP server is **read-only**: `grimore_ask`, `grimore_search`,
`grimore_get_note`, `grimore_connect`, `grimore_list_categories`.
Scans and migrations stay on the CLI.

Copy-pasteable client configs and the working-directory caveat (the
server reads `grimore.toml` from `cwd`, which an MCP client launches
from elsewhere) are in [`docs/mcp-setup.md`](mcp-setup.md).

### HTTP API + web UI

`grimore serve` boots a Starlette ASGI server on `127.0.0.1:8000` with
a minimal vanilla-JS UI. Loopback-only by default; non-loopback binds
require both `--allow-lan` and `--api-token`.

```bash
# Loopback only — open in your browser
grimore serve

# Expose on the LAN with a bearer token
GRIMORE_API_TOKEN="$(openssl rand -hex 32)" \
  grimore serve --host 0.0.0.0 --port 8000 \
                --allow-lan --api-token "$GRIMORE_API_TOKEN"
```

Streaming answers use Server-Sent Events on `POST /api/ask` when the
body has `stream: true`. CORS is off by default; pass
`--cors-origin <origin>` to enable exactly one origin.

The `serve` extra is required: `pip install 'grimore[serve]'`. On
Linux/Windows that pulls FastAPI as the upgrade path; the shipped
implementation uses only Starlette + uvicorn so a Termux install
stays viable (pydantic-core has no prebuilt wheel for Termux/ARM).

### OpenAI-compatible LLM backends

Grimore can talk to any server that speaks the OpenAI
`POST /v1/chat/completions` shape — llama.cpp server, vLLM, LM Studio,
OpenRouter, OpenAI proper. Configure it in `[cognition]`:

```toml
[cognition]
llm_backend     = "openai"
llm_base_url    = "http://localhost:8080/v1"   # llama.cpp server's default
llm_api_key_env = "GRIMORE_LLM_API_KEY"        # env var holding the token
model_llm_local = "your-model-name"            # whatever the server reports
allow_remote    = false                        # leave false to keep loopback-only
```

Then in the shell:

```bash
export GRIMORE_LLM_API_KEY="sk-…"   # leave empty for unauthenticated servers
grimore preflight
grimore ask "warm-up question"
```

Notes:

- The same `SecurityGuard.validate_llm_host` loopback gate enforces
  `allow_remote = false` against your `llm_base_url`. To point at a
  remote host you must explicitly set `allow_remote = true` *and*
  `llm_base_url` must use `https://`.
- The bearer token is read **lazily** from the env var on every call,
  so a rotated key takes effect without restarting Grimore.
- JSON mode uses `response_format: {"type": "json_object"}`; servers
  that don't honour it fall back to the router's "extract first
  `{…}`" regex — same fallback path used in the Ollama branch.
- Embeddings still go through Ollama. The `llm_backend` switch only
  affects the chat-completion path.

---

## 12. Working with bigger models

Bigger models (e.g. `ministral-3:14b`) often need a longer warm-up window
before the first token arrives. Bump the relevant timeouts in
`grimore.toml`:

```toml
[cognition]
model_llm_local   = "ministral-3:14b"
request_timeout_s = 180
stream_timeout_s  = 240
```

Some reasoning models (`qwen3.5:0.8b`, `deepseek-r1` family, etc.) emit a
"thinking" phase: Ollama streams chunks with an empty `response=""` and
the actual content lives in `thinking="…"` for up to a minute before the
first user-visible token. The shell shows a spinner during this phase and
will not look frozen.

`Ctrl+C` during the wait cancels cleanly and returns you to the prompt.

---

## 13. Troubleshooting

**`preflight` says Ollama is unreachable.**
Ollama isn't running, or `allow_remote = false` is rejecting your
hostname. Run `ollama serve` in another terminal and re-try.

**`preflight` says a model isn't pulled.**
`ollama pull qwen2.5:3b` (or whatever your config names). Then re-run
`grimore preflight`.

**`ask` returns "no sources" answers.**
The vault hasn't been embedded yet. Run `grimore scan --no-dry-run` first.
For freshly added notes, the daemon's 45-second debounce hasn't fired —
either wait or run `scan` again manually.

**`/save` reports a vault-traversal rejection.**
The path resolves outside the vault root. Use a path relative to the
vault (e.g. `_transcripts/foo.md`), not an absolute path.

**`@<title>` doesn't autocomplete.**
The fuzzy threshold may be too high. Lower `[shell] fuzzy_threshold` (the
default is 55; try 35–45). `/refresh` rebuilds the cached vault index in
case the title was added after the shell started.

**The shell looks frozen after I asked a question.**
The model is in its thinking phase. The spinner should be visible; if it
isn't, your terminal might be swallowing escape codes. Bigger models can
take 30–90 seconds before the first token.

**`scan` errors with "git not initialised".**
Either run `git init` inside the vault, or set
`output.auto_commit = false` in `grimore.toml` (you lose the safety net).

---

## 14. Glossary

- **Vault** — the root directory containing your Markdown notes.
- **Ingest** — the watchdog observer that detects changes.
- **Cognition** — the LLM-driven tag, category, summary, embedding pass.
- **Memory** — the SQLite database (WAL + FTS5 + vector columns).
- **Oracle** — the RAG layer; answers questions with citations.
- **Synthesis** — `connect`, `distill`, the suggested-connections block.
- **Chronicler** — temporal staleness tracker.
- **Black Mirror** — contradiction detector across notes.
- **RRF** — Reciprocal Rank Fusion, the BM25 + cosine fusion strategy.
- **Dry-run** — preview mode; no writes to disk.
- **Pin** — a note attached to every ask in the current shell session.
- **`@`-mention** — a note attached one-shot to the next ask.
- **Thread** — a JSONL transcript of a shell conversation stored
  under `shell.threads_dir`, resumable across sessions.
- **Profile** — a `[profiles.<name>]` block in `grimore.toml` that
  deep-merges over the defaults when activated via `--profile` or
  `GRIMORE_PROFILE`.
- **MCP** — Model Context Protocol; the stdio JSON-RPC contract by
  which Grimore plugs into Claude Desktop / Cursor / Zed.
- **Backend** — the chat-completion server Grimore talks to.
  `[cognition].llm_backend = "ollama" | "openai"` dispatches between
  Ollama's `/api/generate` and the OpenAI-compatible `/v1/chat/completions`.

---

Released under the MIT License.
