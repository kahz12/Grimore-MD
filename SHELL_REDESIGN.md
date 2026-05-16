# Grimore Shell — Redesign Proposal

A redesign of `grimore shell` modelled after the conversational-first UX of
gemini-cli and Claude Code. The shell becomes an **Oracle-first conversation
surface** with slash commands for everything else.

This document is a proposal, not a commitment. The "Open Questions" section at
the end is what I need from you before writing any code. Please answer in-line
under each `Answer: ---------` block.

---

## 1. Vision

> *"The vault is a conversation. Everything else is a slash away."*

The current shell is a command REPL — every line is a verb (`ask`, `scan`,
`models chat …`). Useful, but it puts a layer of grammar between the user and
the Oracle. The redesign flips the default:

- **Plain text → ask the Oracle.** Streaming answer renders inline.
- **`/` → meta-commands** (scan, models, status, …). A popup picker appears the
  moment you type `/`, the way Claude Code's slash menu and gemini-cli's
  `/`-commands work.
- **`@` → vault references.** Type `@jung-arch` and an autocomplete menu of
  matching notes appears; pick one and it's attached as context for the next
  ask (and rendered as `[[Note Title]]` in the prompt buffer).

Everything else (history, streaming, sources panel) stays — the surface around
the input box changes, not the engines.

---

## 2. Input Model

```
┌──────────────────────────────────────────────────────────────────────────┐
│  test_vault                                                              │
│                                                                          │
│  ❯ what do my notes say about jungian shadow work?_                      │
│                                                                          │
│  vault: test_vault  •  chat: qwen2.5:3b  •  embed: nomic  •  [dry-run]   │
└──────────────────────────────────────────────────────────────────────────┘
```

| Input starts with | Treated as | Example |
|---|---|---|
| (letter / unicode) | Question for the Oracle | `what is the bayesian view of priors?` |
| `/` | Slash command (with popup menu) | `/scan --no-dry-run` |
| `@` | Vault reference (popup of matching notes) | `@bayes_prior` |
| `\` at end of line | Continue to next line | `summarize \⏎the last week of journal entries` |
| (empty) | No-op (does **not** call LLM) | `⏎` |

### Why this fits Grimore specifically

The whole product is *"ask my vault"*. Forcing users to type `ask` every time
is friction the product doesn't want. Slash commands carry the maintenance
ops (`/scan`, `/connect`, `/prune`) which are infrequent.

---

## 3. Slash Command Catalogue

A direct port of the current verbs, plus a few additions:

```
/ask <text>             ── Ask with explicit flags (--no-stream, --top-k, --export)
/status                 ── Dashboard (vault, cognition, daemon)
/tags [--limit N]       ── Top tags
/scan [--no-dry-run]    ── Index the vault
/connect [--threshold]  ── Propose wikilinks
/prune                  ── Remove orphans
/category list|add|rm|notes
/chronicler list|check|verify
/mirror list|scan|show|dismiss|resolve
/distill [--tag …] [--category …]
/models [chat|embed] [name|index]
/refresh                ── Drop cached services
/help [cmd]             ── This list, or details for one command
/clear                  ── Clear screen
/exit                   ── Leave the shell

── New ──
/again                  ── Re-run the last question
/why                    ── Expand the sources of the last answer
/pin @note [@note …]    ── Pin notes to every future ask in this session
/unpin [@note]          ── Drop one or all pins
/save [path]            ── Export this session's transcript
/theme <name>           ── Switch the Rich theme (if multi-theme is adopted)
```

Anything ambiguous gets a "did you mean…?" suggestion via difflib instead of
silently failing.

---

## 4. UI Elements

### 4.1 Bottom Toolbar (persistent)

Replaces the noisy `grimore(test_vault)[dry-run]> ` prefix. Always reflects
live state — after `/models chat foo` it updates immediately.

```
vault: test_vault  •  chat: qwen2.5:3b  •  embed: nomic-embed  •  dry-run
```

Implemented via prompt_toolkit's `bottom_toolbar` callback so it re-renders on
state changes and terminal resize.

### 4.2 Prompt Glyph

`❯ ` for normal mode, `… ` for line-continuation, `▌` while streaming an
answer. The glyph itself communicates state — no extra text needed.

### 4.3 Streaming Answer Block

Already implemented in `_do_ask`. Wrap it in a Rich `Live` with a soft border
so the answer block is visually distinct from the prompt:

```
╭─ Oracle ─────────────────────────────────────────────╮
│ The Jungian shadow refers to the unconscious aspects │
│ of personality that the conscious ego does not       │
│ identify in itself…                                  │
╰──────────────────────────────────────────────────────╯
Sources: [[jung-archetypes]] · [[shadow-work-journal]]
```

### 4.4 Approval Prompts (security-critical)

Destructive actions (`/scan --no-dry-run`, `/connect --no-dry-run`,
`/prune --no-dry-run`, `/category rm`) ask before doing:

```
About to write frontmatter to 47 notes in test_vault.
Continue? [y/N] _
```

A `--yes` flag bypasses the prompt for scripting.

### 4.5 Multi-line Composer

Toggle with `Esc+Enter` or trailing `\`. Enter alone submits. Prompt_toolkit's
`multiline=True` with custom keybindings.

### 4.6 Spinner During Retrieval

Between submit and first streamed token, show a spinner with a state label —
`embedding question…`, `searching vault…`, `waiting on llm…`. Helps when
Ollama is cold-starting a large model.

---

## 5. `@`-Mentions

Type `@`, and a popup completer surfaces note titles matching the typed
fragment. Selecting one:

1. Inserts `[[Note Title]]` into the input buffer.
2. Attaches the note's content to the next question's context (in addition
   to whatever retrieval surfaces).
3. The attached content is wrapped via `SecurityGuard.wrap_untrusted` — same
   defense as Oracle's RAG path.

`/pin` does the same but persists across asks until `/unpin`.

### Security on `@`-mentions

- Resolved paths must live inside the vault root (use `SecurityGuard
  .resolve_within_vault`); reject `..` traversal, absolute paths, symlinks
  out of the vault.
- File size cap (e.g., 32 KB per attached note) — same as the embedder's
  `EMBED_MAX_CHARS`.
- Attached content counts against `_ORACLE_CONTEXT_MAX_CHARS`; whole-source
  truncation already implemented in Oracle handles this gracefully.

---

## 6. Keybindings

| Keys | Action |
|---|---|
| `Enter` | Submit |
| `Esc Enter` *or* trailing `\` | Newline in buffer |
| `Tab` | Trigger completion menu |
| `Ctrl+C` (running) | Cancel current stream / command; stay in shell |
| `Ctrl+C` (empty) | Clear current input line |
| `Ctrl+D` (empty) | Exit shell |
| `Ctrl+R` | Reverse history search (prompt_toolkit built-in) |
| `Ctrl+L` | Clear screen (alias for `/clear`) |
| `Up / Down` | History |

Note: `Esc Enter` is more portable than `Shift+Enter`; Termux keyboards
often can't emit Shift+Enter at all.

---

## 7. Security Considerations

1. **Empty input never reaches the LLM.** A bare Enter is a no-op.
2. **`@`-paths are vault-scoped** and re-validated through
   `SecurityGuard.resolve_within_vault` on every reference.
3. **Destructive commands prompt y/N** by default — `--yes` for scripting.
4. **Prompt-injection defence is unchanged.** All retrieved + attached note
   content keeps going through `wrap_untrusted` / `sanitize_prompt`.
5. **History file** stays under the platform user-cache dir (current
   `shell_history_path`), per-vault. No secrets persisted — questions only.
6. **Typo guard.** Input starting with `/` that doesn't match a known slash
   command shows a "did you mean `/category`?" suggestion **without**
   silently falling through to `ask`. This protects against a user typing
   `/scna` and accidentally sending it to the LLM.
7. **Rate guard.** Identical consecutive questions inside 2 seconds are
   collapsed into one Ollama call (cheap protection against double-Enter).

---

## 8. Migration Plan

- **Phase 1 — Add, don't remove.** Ship slash commands alongside the existing
  word commands. Bare-text → ask is the new default; the old `ask <text>`
  form still works (just becomes redundant).
- **Phase 2 — Deprecate.** Word commands print a one-line nudge:
  `(tip: try /scan)`. Slash commands stay primary.
- **Phase 3 — Remove word commands** in a later release.

This keeps the existing 344 tests green during the transition (they exercise
`dispatch()` with word-form inputs).

---

## 9. Files Touched (rough scope)

| File | Change |
|---|---|
| `grimore/shell.py` | New input parser, slash dispatch, `@` completer, bottom toolbar, approval prompts. The bulk of work. |
| `grimore/operations.py` | Add `_do_pin`, `_do_unpin`, `_do_save_transcript`, `_do_repeat_last`. |
| `grimore/session.py` | Hold `pinned_notes: list[str]`, `last_question: str | None`, `last_answer: dict | None`. |
| `grimore/cognition/oracle.py` | `ask_stream(question, pinned_attachments=[…])` accepts pre-resolved attachments. |
| `tests/test_shell.py` | New suite covering slash parsing, `@`-resolution, approval flow, typo suggestions. |

No DB schema change. No new dependencies (prompt_toolkit + Rich already cover
everything).

---

## 10. Open Questions

Please answer below each. Anything you skip I'll fill in with the
"recommended" choice noted in the question.

### Q1. Migration strategy

Hard-cut to slash-only on day one, or run word + slash side-by-side for a
release?

**Recommended:** side-by-side (Phase 1 in §8). Lower regression risk; the
existing tests stay valid.

Answer: ---------
Side-by-Side.
---

### Q2. `@`-mention semantics

When `@note` is in the input, should the **full note body** be attached as
context, or only the note **title** (so the existing RAG retrieval surfaces
the chunks)?

**Recommended:** full body, wrapped via `SecurityGuard.wrap_untrusted`. The
RAG path already handles oversize content via whole-source truncation.

Answer: ---------
Full body.
---

### Q3. Newline keybinding

`Esc Enter` (portable, works in Termux) or `Alt+Enter` (more discoverable,
breaks on Android keyboards)?

**Recommended:** support both, document `Esc Enter` as primary.

Answer: ---------
Both.
---

### Q4. Bottom toolbar — always on?

Always visible (gemini-cli style) or toggleable via `/togglebar`?

**Recommended:** always on. The state it shows (model, dry-run) is exactly
the state you need before hitting Enter on a destructive command.

Answer: ---------
Always on.
---

### Q5. Themes

Single existing Rich theme, or add `/theme dark|light|nord`?

**Recommended:** single theme for v1. Adding themes is a separate, low-value
follow-up.

Answer: ---------
Single Theme.
---

### Q6. Unknown slash command behaviour

`/scna` → "did you mean `/scan`?" suggestion (difflib), or hard "unknown
command" error?

**Recommended:** suggestion. The whole point of the `/` namespace is
discoverability.

Answer: ---------
Suggestion.
---

### Q7. Approval prompts for destructive commands

Always prompt for `--no-dry-run` actions, or only when invoked without an
explicit `--yes`?

**Recommended:** always prompt unless `--yes` is supplied. Matches Claude
Code's permission-prompt UX and Grimore's existing dry-run-first ethos.

Answer: ---------
Always promp.
---

### Q8. Multi-turn conversation memory

Currently each `ask` is independent. Should the new shell carry prior Q&A
into the next ask as additional system context (so you can say "elaborate on
the second point")?

**Recommended:** **no** for v1 — token cost grows quickly, and Grimore's
strength is grounded-in-vault answers rather than free-form chat memory. Add
a `/again` command for the common "repeat the last question" case and call
multi-turn a v2 feature.

Answer: ---------
For now, NO.
---

### Q9. `/save` transcript format

Markdown (human-readable, drops into the vault as a note), JSON (machine-
readable), or both?

**Recommended:** markdown. The output naturally fits as a vault note — and
it can be re-ingested by `/scan` later, closing the loop.

Answer: ---------
Markdown.
---

### Q10. Vault label in the toolbar

Use the directory name (`test_vault`), or read an optional
`vault.display_name` field from `grimore.toml` so users can call their vault
"Library" or "Grimoire"?

**Recommended:** add the optional `display_name` field; fall back to the
directory name when unset.

Answer: ---------
Set the optional `display_name` field.
---

### Q11. Anything I'm missing?

Things you want that aren't in this proposal — vim-mode editing, a side panel
of recent files, fuzzy-find across the vault, anything else inspired by
gemini-cli or Claude Code that I haven't pulled in.

Answer: ---------
Three concrete additions identified:

---

#### A1. Fuzzy search on `@`-mentions

**What it is:**
The current `@` completer uses prefix match — `@jung` only surfaces notes
whose title starts with "jung". Replacing it with fuzzy match means `@jung`
also finds `archetypes-jung`, `carl-jung-notes`, or any title containing the
fragment anywhere in the string.

**Implementation:**
- Use `rapidfuzz.process.extract(fragment, note_titles, scorer=WRatio, limit=8)`
  as the backend for the prompt_toolkit `Completer`.
- Install `rapidfuzz` (available via pip, no heavy dependencies).
- Minimum score threshold should be configurable in `grimore.toml` under
  `[shell] fuzzy_threshold = 60` (range 0–100). Recommended default: 55.
- Results in the popup must be sorted by score descending. Optionally show
  the score as a visual hint if the theme allows:
  `@jung-archetypes  92`.
- Path resolution and `SecurityGuard.resolve_within_vault` apply exactly as
  in the current prefix match — the change is only in how suggestions are
  ranked, not in how paths are resolved.

**Why in v1:**
Notes with compound or inverted names (`shadow-work-jung` instead of
`jung-shadow-work`) are real cases. A completer that silently fails on these
forces the user to remember the exact prefix, which breaks the conversational
UX the redesign aims for.

---

#### A2. Vim-mode in the input buffer

**What it is:**
Enable vi editing mode in prompt_toolkit, allowing modal navigation and
editing (`hjkl`, `w`, `b`, `cw`, `dd`, `$`, `0`, etc.) inside the shell
input buffer.

**Implementation:**
- In the `PromptSession` constructor, add `vi_mode=True`. That is literally
  all that is needed for the base functionality.
- Add `vi_mode = false` (default OFF) under `[shell]` in `grimore.toml` so
  new users are not surprised.
- The shell reads that field at startup and passes
  `vi_mode=config.shell.vi_mode` to the `PromptSession`.
- The prompt glyph should reflect the current vi mode:
  - `❯ ` → insert mode (normal typing behaviour)
  - `∙ ` → normal mode (command mode)
  prompt_toolkit exposes `get_app().vi_state.input_mode` to detect this;
  use it in the `bottom_toolbar` callback or in the prompt string.
- Document in `/help` and in the README that vi-mode is enabled via
  `vi_mode = true` in `grimore.toml`.

**Why in v1:**
One line of code, zero new dependencies, feature flag OFF by default.
There is no reason to defer it. For users with a physical keyboard (or
Termux on a tablet with hardware keyboard) it is a significant quality-of-life
improvement when editing long questions.

---

#### A3. `/history [N]` command

**What it is:**
Display the last N questions from the current session inside the shell,
without exiting or manually opening the history file.

**Implementation:**
- `session.py`: the `Session` object already has `last_question: str | None`.
  Expand it to `question_log: list[str] = []`. Every successful `ask` call
  appends: `session.question_log.append(question)`.
- `operations.py`: add `_do_history(n: int = 10)` that prints the last N
  items numbered using Rich:
---
