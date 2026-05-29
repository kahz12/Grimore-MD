# MCP setup

Grimore ships a Model Context Protocol (MCP) server that exposes the
vault's RAG surface as tools to any MCP-aware client (Claude Desktop,
Cursor, Zed, etc.). It runs over stdio — your client spawns
`grimore mcp` and talks JSON-RPC on its stdin/stdout.

## What you get

| Tool | Inputs | Returns |
|---|---|---|
| `grimore_ask` | `question`, `top_k?` | Cited answer + sources, drops hallucinated citations |
| `grimore_search` | `query`, `top_k?` | Top-k hybrid (BM25 + cosine) hits with snippets |
| `grimore_get_note` | `note_id` *or* `title` | Note metadata + on-disk body |
| `grimore_connect` | `note_id`, `top_k?` | Related notes via cosine on chunk vectors |
| `grimore_list_categories` | — | Vault-wide category counts |

The server is **read-only**. Scan and migrate operations stay on the
CLI so an LLM client can't trigger destructive ops by accident.

## Quick start

Pre-flight: have a scanned vault, Ollama running (or your configured
backend up), and `grimore` on PATH.

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
