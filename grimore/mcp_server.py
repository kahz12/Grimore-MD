"""Model Context Protocol (MCP) server — read-only Grimore RAG surface.

Exposes the warm ``Session.oracle`` + ``Connector`` behind an MCP
stdio server so Claude Desktop, Cursor, Zed and other MCP-aware
clients can use Grimore as a tool.

This module implements the JSON-RPC 2.0 framing and the MCP
``initialize`` / ``tools/list`` / ``tools/call`` handshake by hand —
the wire shape is simple enough that pulling in the official ``mcp``
SDK as a hard dependency would be more weight than it's worth. The
``mcp`` extra in ``pyproject.toml`` is still declared for forward
compatibility in case a future phase wants the SDK's transport helpers.

Tools exposed:

* ``grimore_ask``           — RAG-backed answer with cited sources.
* ``grimore_search``        — Hybrid (BM25 + cosine) retrieval, no LLM.
* ``grimore_get_note``      — Note body + frontmatter by id or title.
* ``grimore_connect``       — "Notes related to this one" via cosine.
* ``grimore_list_categories`` — Vault-wide category counts.

Read-only by design. ``scan`` / ``migrate-embeddings`` stay on the CLI
so an LLM client can't trigger destructive ops by mistake.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from typing import IO, Any, Callable, Optional

from grimore.session import Session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# MCP protocol version we advertise in the initialize response. Picked
# from the spec revision Claude Desktop and Cursor implement as of late
# 2025; the field is informational, mismatching client/server versions
# still work as long as both speak JSON-RPC 2.0.
_MCP_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "grimore", "version": "2.4.0"}


# ── JSON-RPC framing ───────────────────────────────────────────────────


def _rpc_result(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id, code: int, message: str, data=None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# Subset of the JSON-RPC error codes we actually use. Anything else
# (transport errors, framing problems) we surface as -32603 internal.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


# ── Tool wrapping ──────────────────────────────────────────────────────


@dataclass
class Tool:
    """A single MCP tool: name + JSON schema + Python handler.

    The handler receives the parsed ``arguments`` dict and returns a
    JSON-serialisable value. The wire layer wraps that value into MCP's
    ``content`` array so callers don't have to format text/JSON output
    twice.
    """
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]


def _content_text(value: Any) -> list[dict]:
    """Format a tool's return value as MCP ``content`` entries.

    MCP requires every tool response to be a list of typed content
    blocks. We always emit one ``text`` block whose body is the JSON
    serialisation of the handler's return — easy to parse client-side
    and works on every MCP client we know of.
    """
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [{"type": "text", "text": payload}]


# ── Server ─────────────────────────────────────────────────────────────


class MCPServer:
    """Stdio MCP server wrapping a warm :class:`grimore.session.Session`.

    Construction is cheap; the warm ``Session`` is built once and reused
    across every ``tools/call`` so the embedder + router don't pay
    cold-start cost per query.

    Concurrency model: stdin is read sequentially in :meth:`serve` —
    Claude Desktop, Cursor and Zed all serialise requests over a single
    transport, so a single-threaded loop is correct. We still take a
    lock around the Session in case a future client multiplexes.
    """

    def __init__(self, session: Optional[Session] = None):
        self.session = session or Session()
        self._lock = threading.Lock()
        self._tools = self._build_tools()

    # ── tool table ────────────────────────────────────────────────────

    def _build_tools(self) -> dict[str, Tool]:
        """Define the read-only tool surface.

        Schemas are JSON-Schema draft-07 (the dialect every MCP client
        we tested accepts). They double as the client-visible
        documentation, so the ``description`` fields are written for an
        LLM reader, not just a human one.
        """
        return {
            "grimore_ask": Tool(
                name="grimore_ask",
                description=(
                    "Answer a question against the Grimore vault with retrieval-"
                    "augmented generation. Returns the answer plus the wikilink "
                    "labels of the sources that informed it. Citations the model "
                    "fabricated (not in the retrieved context) are stripped."
                ),
                input_schema={
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {"type": "string", "description": "Question to ask the vault."},
                        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
                    },
                },
                handler=self._tool_ask,
            ),
            "grimore_search": Tool(
                name="grimore_search",
                description=(
                    "Hybrid (BM25 + cosine) search across the vault. Returns the "
                    "top-k matching chunks with note id, title, snippet, and a "
                    "fused relevance score. No LLM call — fast."
                ),
                input_schema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    },
                },
                handler=self._tool_search,
            ),
            "grimore_get_note": Tool(
                name="grimore_get_note",
                description=(
                    "Fetch a note's metadata and body. Provide either note_id "
                    "(preferred) or title (resolved via exact match)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "integer"},
                        "title": {"type": "string"},
                    },
                    # JSON-Schema can express 'one of these two' via oneOf;
                    # keep the schema lenient and validate in the handler so
                    # MCP clients that under-support oneOf still work.
                },
                handler=self._tool_get_note,
            ),
            "grimore_connect": Tool(
                name="grimore_connect",
                description=(
                    "Find notes related to a given note via cosine similarity "
                    "on the embeddings. Useful for 'see also' suggestions."
                ),
                input_schema={
                    "type": "object",
                    "required": ["note_id"],
                    "properties": {
                        "note_id": {"type": "integer"},
                        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
                    },
                },
                handler=self._tool_connect,
            ),
            "grimore_list_categories": Tool(
                name="grimore_list_categories",
                description=(
                    "List vault categories with note counts. Useful as a vault "
                    "table-of-contents."
                ),
                input_schema={"type": "object", "properties": {}},
                handler=self._tool_list_categories,
            ),
        }

    # ── tool handlers ─────────────────────────────────────────────────

    def _tool_ask(self, args: dict) -> dict:
        question = args.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is required and must be a non-empty string")
        # ValueError on a non-numeric value propagates to the dispatcher,
        # which maps it to _INVALID_PARAMS. maximum mirrors the advertised
        # input-schema bound so the handler actually enforces it (audit L2).
        top_k = SecurityGuard.coerce_top_k(args.get("top_k"), default=5, maximum=25)
        result = self.session.oracle.ask(question, top_k=top_k)
        return {
            "answer": result.get("answer", ""),
            "sources": list(result.get("sources") or []),
            "dropped_citations": int(result.get("dropped_citations") or 0),
        }

    def _tool_search(self, args: dict) -> dict:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required and must be a non-empty string")
        top_k = SecurityGuard.coerce_top_k(args.get("top_k"), default=10, maximum=50)
        query_vector = self.session.embedder.embed(query)
        use_hybrid = (
            getattr(self.session.config.cognition, "hybrid_search", True)
            and self.session.db.fts_available
        )
        if use_hybrid:
            hits = self.session.oracle.connector.find_hybrid(
                query_text=query,
                query_vector=query_vector,
                top_k=top_k,
            )
        elif query_vector:
            hits = self.session.oracle.connector.find_similar_notes(query_vector, top_k=top_k)
        else:
            hits = []

        rows: list[dict] = []
        for hit in hits:
            note_id = hit.get("note_id")
            title = self.session.db.get_note_title(note_id) or ""
            text = hit.get("text") or ""
            snippet = text[:200] + ("…" if len(text) > 200 else "")
            rows.append({
                "note_id": int(note_id),
                "title": title,
                "snippet": snippet,
                "score": float(hit.get("score") or 0.0),
            })
        return {"hits": rows}

    def _tool_get_note(self, args: dict) -> dict:
        note_id = args.get("note_id")
        title = args.get("title")
        row = None
        if note_id is not None:
            loc = self.session.db.get_note_location(int(note_id))
            if loc is not None:
                path, resolved_title = loc
                row = {"note_id": int(note_id), "path": path, "title": resolved_title}
        elif isinstance(title, str) and title.strip():
            # Title lookup goes through the existing path-keyed table:
            # iterate cached titles and match. For now we delegate to a
            # direct query against ``notes`` so we don't have to add a
            # new DB method just for the MCP path.
            with self.session.db._get_connection() as conn:
                hit = conn.execute(
                    "SELECT id, path, title FROM notes WHERE title = ? LIMIT 1",
                    (title.strip(),),
                ).fetchone()
            if hit is not None:
                row = {"note_id": int(hit[0]), "path": hit[1], "title": hit[2]}
        else:
            raise ValueError("Provide either note_id or title")

        if row is None:
            return {"found": False}

        # Read body off disk so callers see the live file (incremental
        # re-embed makes the DB a partial mirror of the chunked content,
        # not a full-text mirror).
        from pathlib import Path

        body = ""
        try:
            body = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("mcp_note_read_failed", path=row["path"], error=str(exc))
        return {
            "found": True,
            "note_id": row["note_id"],
            "title": row["title"],
            "path": row["path"],
            "body": body,
        }

    def _tool_connect(self, args: dict) -> dict:
        note_id = args.get("note_id")
        if note_id is None:
            raise ValueError("note_id is required")
        note_id = int(note_id)
        top_k = SecurityGuard.coerce_top_k(args.get("top_k"), default=5, maximum=25)

        # Pull the note's average embedding as the query vector: average
        # all its chunks so we score against the whole note's "topic"
        # rather than an arbitrary first chunk.
        with self.session.db._get_connection() as conn:
            rows = conn.execute(
                "SELECT vector FROM embeddings WHERE note_id = ?",
                (note_id,),
            ).fetchall()
        if not rows:
            return {"hits": []}
        from grimore.cognition.embedder import Embedder

        vectors = [Embedder.deserialize_vector(r[0]) for r in rows if r[0]]
        if not vectors:
            return {"hits": []}
        dim = len(vectors[0])
        avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        normed = Embedder.normalize(avg)

        hits = self.session.oracle.connector.find_similar_notes(
            normed, top_k=top_k, exclude_note_id=note_id, dedupe_by_note=True,
        )
        return {
            "hits": [
                {
                    "note_id": int(h["note_id"]),
                    "title": self.session.db.get_note_title(h["note_id"]) or "",
                    "score": float(h.get("score") or 0.0),
                }
                for h in hits
            ]
        }

    def _tool_list_categories(self, args: dict) -> dict:
        rows = self.session.db.get_category_frequency()
        return {
            "categories": [
                {"category": cat, "count": int(n)} for cat, n in rows
            ]
        }

    # ── dispatch ──────────────────────────────────────────────────────

    def handle_request(self, request: dict) -> Optional[dict]:
        """Dispatch one JSON-RPC request to its handler.

        Returns the response dict, or ``None`` for notifications (where
        the spec says no response is emitted).
        """
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        # Notifications: spec says no response. The "initialized"
        # notification fires after initialize completes.
        if req_id is None:
            if method == "notifications/initialized":
                return None
            logger.info("mcp_notification_ignored", method=method)
            return None

        if method == "initialize":
            return _rpc_result(req_id, {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _SERVER_INFO,
            })

        if method == "tools/list":
            return _rpc_result(req_id, {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in self._tools.values()
                ],
            })

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool = self._tools.get(tool_name)
            if tool is None:
                return _rpc_error(req_id, _METHOD_NOT_FOUND, f"unknown tool: {tool_name}")
            arguments = params.get("arguments") or {}
            try:
                with self._lock:
                    value = tool.handler(arguments)
            except ValueError as exc:
                return _rpc_error(req_id, _INVALID_PARAMS, str(exc))
            except Exception as exc:  # surface to the client without crashing the loop
                logger.error("mcp_tool_failed", tool=tool_name, error=str(exc))
                return _rpc_result(req_id, {
                    "content": _content_text({"error": str(exc)}),
                    "isError": True,
                })
            return _rpc_result(req_id, {
                "content": _content_text(value),
                "isError": False,
            })

        if method == "ping":
            return _rpc_result(req_id, {})

        return _rpc_error(req_id, _METHOD_NOT_FOUND, f"unknown method: {method}")

    # ── transport ─────────────────────────────────────────────────────

    def serve(self, *, stdin: IO[str] = None, stdout: IO[str] = None) -> None:
        """Run the stdio loop until stdin closes.

        Wire format follows the MCP convention used by Claude Desktop:
        newline-delimited JSON objects, one request or notification per
        line. The official spec also defines a ``Content-Length`` framed
        variant; every client we've tested defaults to NDJSON over
        stdio, so we ship the simpler form.
        """
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        logger.info("mcp_server_started")

        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write(stdout, _rpc_error(None, _PARSE_ERROR, str(exc)))
                continue
            if not isinstance(request, dict):
                self._write(stdout, _rpc_error(None, _INVALID_REQUEST, "request must be a JSON object"))
                continue
            response = self.handle_request(request)
            if response is not None:
                self._write(stdout, response)

        logger.info("mcp_server_stopped")

    @staticmethod
    def _write(stream: IO[str], payload: dict) -> None:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def run_stdio(session: Optional[Session] = None) -> None:
    """CLI entry point — build a server and run its stdio loop forever."""
    server = MCPServer(session=session)
    server.serve()
