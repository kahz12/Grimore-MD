# Grimore — User Guide (English)

> A local-first knowledge engine for Markdown vaults.
> Everything runs against a loopback Ollama; no API keys, no telemetry, no cloud.

---

## Table of contents

1. [What Grimore is (and what it isn't)](#1-what-grimore-is-and-what-it-isnt)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Your first vault — the five-minute tour](#4-your-first-vault--the-five-minute-tour)
5. [The `grimore.toml` file](#5-the-grimoretoml-file)
6. [Day-to-day commands](#6-day-to-day-commands)
   - [`scan`](#scan)
   - [`connect`](#connect)
   - [`ask`](#ask)
   - [`tags`](#tags)
   - [`prune`](#prune)
   - [`status`](#status)
   - [`preflight`](#preflight)
   - [`daemon`](#daemon)
   - [`maintenance run`](#maintenance-run)
   - [`category`](#category)
   - [`chronicler`](#chronicler)
   - [`mirror`](#mirror)
   - [`distill`](#distill)
7. [The interactive shell — `grimore shell`](#7-the-interactive-shell--grimore-shell)
   - [Composing input](#composing-input)
   - [Slash commands](#slash-commands)
   - [`@`-mentions](#-mentions)
   - [Pinning notes](#pinning-notes)
   - [Approval prompts](#approval-prompts)
   - [Saving transcripts](#saving-transcripts)
   - [Switching models live](#switching-models-live)
   - [Bottom toolbar](#bottom-toolbar)
   - [Vi-mode](#vi-mode)
8. [Taxonomy: `taxonomy.yml`](#8-taxonomy-taxonomyyml)
9. [Frontmatter conventions](#9-frontmatter-conventions)
10. [Privacy & safety](#10-privacy--safety)
11. [Working with bigger models](#11-working-with-bigger-models)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)

---

## 1. What Grimore is (and what it isn't)

Grimore watches a directory of plain Markdown files (your **vault**) and turns it
into a queryable knowledge base. For every note it:

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
# Optional label shown in the shell's bottom toolbar. Defaults to the
# directory name when unset.
display_name = "Library"

[cognition]
model_llm_local       = "qwen2.5:3b"
model_embeddings_local = "nomic-embed-text"
allow_remote          = false   # block non-loopback Ollama endpoints
hybrid_search         = true    # BM25 + cosine via RRF
rrf_k                 = 60      # RRF rank-weight; lower = steeper
connect_threshold     = 0.7     # cosine floor for a suggested wikilink
request_timeout_s     = 60      # /api/generate, JSON path
stream_timeout_s      = 120     # /api/generate, streaming path
embed_timeout_s       = 30      # /api/embeddings

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
enabled    = false
log_events = true

[shell]
vi_mode         = false  # set true for prompt_toolkit vi-mode
fuzzy_threshold = 55     # 0–100, rapidfuzz score floor for @-completions
```

Unknown keys are logged at WARNING and ignored — copy-pasting old config
snippets won't crash the CLI.

---

## 6. Day-to-day commands

Run `grimore <cmd> --help` for the authoritative flag list. Below is a
detailed walk-through.

### `scan`

```bash
grimore scan [-p PATH] [--dry-run|--no-dry-run] [--json]
```

Walks the vault, tags new or changed notes, and refreshes the embedding index.

- `-p, --vault-path` — override the vault path for this run only.
- `--dry-run` / `--no-dry-run` — flip without touching the config.
- `--json` — emit JSON-formatted structured logs (handy in CI / monitoring).

**Idempotency.** Notes are hashed with SHA-256; an unchanged note never
calls the LLM. Re-scanning a fully indexed vault is effectively free.

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

The Oracle always cites the notes it pulled context from. Citations appear
at the end of the answer as `[[note-title]]` links.

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
| `/models [chat\|embed [name\|idx]]` | List Ollama models & switch live. |
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

## 11. Working with bigger models

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

## 12. Troubleshooting

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

## 13. Glossary

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

---

Released under the MIT License.
