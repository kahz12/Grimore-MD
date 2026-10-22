# MCP setup

Grimore ships a Model Context Protocol (MCP) server that exposes the
vault's RAG surface as tools to any MCP-aware client (Claude Desktop,
Cursor, Zed, etc.). It runs over stdio — your client spawns
`grimore mcp` and talks JSON-RPC on its stdin/stdout.

## What you get

| Tool | Inputs | Returns |
|---|---|---|
| `grimore_library_info` | — | Local vault-path fingerprint and model execution mode; no path, content or model call |
| `grimore_ask` | `question`, `top_k?` | Cited answer + evidence status, model execution mode, exact indexed passages, locators, source-version state and indexed PDF extraction coverage |
| `grimore_search` | `query`, `top_k?` | Top-k hybrid hits with snippets, exact `source_id` and indexed file hash (or null for unresolved legacy hits) |
| `grimore_get_note` | `note_id` *or* `title` | Note metadata, on-disk body and PDF extraction status |
| `grimore_get_source` | `source_id`, `expected_indexed_file_hash?`, `obsidian_vault?` | Stored indexed passage, locator, file-version state, extraction coverage and optional Markdown open target |
| `grimore_review_inbox` | `kind?`, `limit?`, `offset?`, `obsidian_vault?` | Active maintenance candidates with bounded evidence, coverage, counts and delivery limits; no decisions or history |
| `grimore_connect` | `note_id`, `top_k?` | Related notes via cosine on chunk vectors |
| `grimore_list_categories` | — | Vault-wide category counts |

The server is **read-only**. Scan and migrate operations stay on the
CLI so an LLM client can't trigger destructive ops by accident.

## Data boundaries: exactly what leaves Grimore

There are two distinct receivers to consider:

1. **The MCP client** receives each tool result over local stdio. The client
   may then send that result to its own local or hosted model; that second hop
   is controlled by the client, not by Grimore.
2. **Grimore's configured model services** receive only the inputs listed in
   the table below. Run `grimore status` before connecting a client. It shows
   `Execution: local`, `mixed`, or `remote`, plus separate chat and embedding
   routes. `grimore_ask` also returns the same `model_execution` object.

| Tool | MCP client receives | Grimore model services receive |
|---|---|---|
| `grimore_library_info` | Contract version, deterministic fingerprint of the local canonical vault path, and model execution mode. The fingerprint is an identity hint, not authentication or anonymization. | Nothing; no database is opened and no model service is probed. |
| `grimore_ask` | Answer, citation labels, evidence status, and one `source_details` item for every retrieved context chunk. Each item includes the **exact complete indexed passage**, relative document path, page/heading when known, content hash and current/modified/missing state. PDF items also include bounded `indexed_extraction` coverage; `partial` names missing pages, while `unknown` marks a legacy index without a recorded diagnosis. | The embedding service receives the question. The chat service receives the question, the selected indexed passages, their source labels and the Oracle instructions, including a coverage warning for partial or unknown PDF extraction. It does not receive unrelated notes or the live full file. |
| `grimore_search` | At most 200 characters of each matching indexed chunk (plus a truncation marker), note id, title, score, `source_id` and `indexed_file_hash`. No full passages or live file hashes. | The embedding service receives the search query. There is no chat completion. BM25 matching remains in-process. |
| `grimore_get_note` | The selected note's metadata, stored path (currently absolute), **currently readable body**, and `extraction` diagnostics. A PDF can be `complete`, `recovered`, or `partial` by page; a file that cannot currently be read reports `failed`. This body can be larger or newer than its indexed chunks. | Nothing; the file is parsed locally and returned directly to the MCP client. |
| `grimore_get_source` | One stored passage, relative path, locator, indexed/current hashes, file state and extraction coverage. Legacy rows without full text remain explicit excerpts. Optional Obsidian targets expose the requested vault name/ID and relative Markdown path, never an absolute path. | Nothing; reads SQLite and hashes the contained current file locally. No legacy re-chunking, embeddings or OCR. |
| `grimore_review_inbox` | Active candidate ids/types/priorities, document metadata and relative paths, reasons, selected indexed passages or extracted claim pairs, coverage, latest review state and delivery limits. Omits decision history. Passage text is capped at 3,000 characters; `text_truncated` marks delivery truncation. | Nothing; aggregates stored signals and vectors. Does not run claim extraction, contradiction generation or verification. |
| `grimore_connect` | Related note ids, titles and similarity scores. | Nothing; it compares vectors already stored in SQLite. |
| `grimore_list_categories` | Category names and note counts. | Nothing; it reads SQLite only. |

`source_details.passage.text` is not the short search preview. It is the full
passage that was indexed and supplied to the answer model (subject to the
document chunk-size limits). The adjacent `passage.completeness` and
`source_state` fields say whether that evidence is complete and still
matches the file.

The MCP tool schemas and descriptions are also visible to the client during
discovery. Grimore never sends the entire vault automatically, but a client
can request complete individual notes with `grimore_get_note`; only enable
that tool in clients you trust with those files.

`grimore_ask` with the optional sufficiency control also returns its audit
(`sufficiency`), including candidate passages, bounded checker/answer-model
response snapshots, proof validation and timing. The client receives that
audit even when the answer abstains. With the control off, this field is absent.

## First editor target: Obsidian

Obsidian was selected explicitly by the user for phase 5. The current block
provides **MCP queries plus non-executing open links**. The next delivered
block adds a desktop-only Obsidian plugin, shipped
locally but not installed or manually validated.

Supply `obsidian_vault` explicitly as the vault name or ID; it is not inferred
from the folder name. The generated `obsidian://open` targets use only `vault`
and `file`, with percent-encoded values. They never create/append/overwrite a
note or launch an application. A trusted client/user must choose to open them.
Only Markdown is supported in this block. Heading navigation is provided only
for a hash-matching source; repeated headings and exact chunk offsets are not
resolved. Modified or unverified files open at document level. Missing files,
out-of-vault paths and ambiguous note paths have no link. This follows the
[official Obsidian URI contract](https://help.obsidian.md/Extending+Obsidian/Obsidian+URI).

## Local, mixed, and remote model execution

- `local`: chat and embeddings are constrained to loopback endpoints.
- `mixed`: one service is explicit loopback and the other uses a remote URL.
- `remote`: both services use remote URLs.

The default `allow_remote = false` blocks non-loopback model endpoints. Mode
inspection performs no DNS lookup or HTTP request. A non-loopback hostname is
reported conservatively as remote when the remote gate is enabled. This mode
describes Grimore's own model calls; it does not describe what the MCP client
does with tool results.

## Quick start

Have an accessible vault and `grimore` on PATH. Source/review queries need
an existing index, but startup and library identity do not require model
services. Search and answers use their configured services when requested.
MCP startup validates the local directory only; it no longer runs the full
model/adapter preflight before discovery. Use `grimore preflight` separately
when you want that broader diagnostic. Logs and incidental output go to
stderr; stdout is reserved for JSON-RPC frames.

```bash
# Test the server independently first
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | grimore mcp
```

You should see a JSON object on stdout with `serverInfo.name` set to
`grimore`.

## Claude Desktop

Edit your `claude_desktop_config.json` (on macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`;
on Linux: `~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "grimore": {
      "command": "grimore",
      "args": ["mcp"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

Restart Claude Desktop. The Grimore tools appear in the tool picker
once it reconnects.

## Cursor

In Cursor's settings → MCP, add:

```json
{
  "grimore": {
    "command": "grimore",
    "args": ["mcp"]
  }
}
```

## Zed

In Zed's `settings.json`:

```json
{
  "experimental.context_servers": {
    "grimore": {
      "command": { "path": "grimore", "args": ["mcp"] }
    }
  }
}
```

## Working directory matters

Grimore reads `grimore.toml` and the vault from the **current working
directory**. MCP clients launch the server from their own working
directory, so either:

* set `cwd` in the client config (Claude Desktop and Cursor support
  this), pointing it at your vault root, **or**
* call the absolute path of a wrapper script that `cd`s first.

Example wrapper:

```bash
#!/usr/bin/env bash
cd /home/me/vault && exec grimore mcp
```

## Logs

`grimore mcp` defaults to JSON logs on stderr. Clients usually capture
that into their own log file; on Claude Desktop you can find it under
`~/Library/Logs/Claude/mcp-server-grimore.log` (macOS) or the
equivalent path on Linux/Windows.

Disable JSON logging with `grimore mcp --no-json` if you want to read
stderr directly.
