"""Model Context Protocol (MCP) server — read-only Grimore RAG surface.

Exposes the warm ``Session.oracle`` + ``Connector`` behind an MCP
stdio server so Claude Desktop, Cursor, Zed and other MCP-aware
clients can use Grimore as a tool.

This module implements the JSON-RPC 2.0 framing and the MCP
``initialize`` / ``tools/list`` / ``tools/call`` handshake by hand —
the wire shape is simple enough that pulling in the official ``mcp``
SDK as a dependency would be more weight than it's worth.

Tools exposed:

* ``grimore_ask``           — RAG-backed answer with cited sources.
* ``grimore_search``        — Hybrid (BM25 + cosine) retrieval, no LLM.
* ``grimore_get_note``      — Note body + frontmatter by id or title.
* ``grimore_get_source``    — Exact indexed passage by structured source id.
* ``grimore_review_inbox``  — Active maintenance candidates, no decisions.
* ``grimore_library_info``  — Local vault fingerprint and model execution mode.
* ``grimore_connect``       — "Notes related to this one" via cosine.
* ``grimore_list_categories`` — Vault-wide category counts.

Read-only by design. ``scan`` / ``migrate-embeddings`` stay on the CLI
so an LLM client can't trigger destructive ops by mistake.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from contextlib import redirect_stdout
from hashlib import sha256
from typing import IO, Any, Callable, Optional

from grimore.session import Session
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.cognition.oracle import _format_source_label
from grimore.cognition.review_inbox import build_review_inbox
from grimore.integrations.obsidian import open_target
from grimore.utils.connectivity import cognition_connectivity
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

from grimore import __version__ as _pkg_version

logger = get_logger(__name__)

# MCP protocol version we advertise in the initialize response. Picked
# from the spec revision Claude Desktop and Cursor implement as of late
# 2025; the field is informational, mismatching client/server versions
# still work as long as both speak JSON-RPC 2.0.
_MCP_PROTOCOL_VERSION = "2024-11-05"
# Version comes from the package; hard-coding it here let it drift to 2.4.0
# while the package shipped 3.2.0, and an MCP client reads this field.
_SERVER_INFO = {"name": "grimore", "version": _pkg_version}
_SOURCE_ID = re.compile(r"^note:([1-9][0-9]{0,18}):chunk:([0-9]{1,19}):embedding:([1-9][0-9]{0,18})$")
_REVIEW_KINDS = ("contradiction", "duplicate_exact", "duplicate_near", "review_due")
_OBSIDIAN_SCHEMA = {
    "type": "string", "minLength": 1, "maxLength": 160,
    "description": "Explicit Obsidian vault name or ID. Returns open links; never launches Obsidian.",
}


# ── JSON-RPC framing ───────────────────────────────────────────────────


def _rpc_result(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id, code: int, message: str, data=None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# Subset of the JSON-RPC failure codes we actually use. Anything else
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
            "grimore_library_info": Tool(
                name="grimore_library_info",
                description=(
                    "Read a local-path fingerprint of the configured vault and the "
                    "model execution mode. Editor clients use this to reject a "
                    "different library before requesting passages. No absolute path, "
                    "credentials, note content, database or model service is read. "
                    "The fingerprint is an identity hint, not authentication."
                ),
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._tool_library_info,
            ),
            "grimore_ask": Tool(
                name="grimore_ask",
                description=(
                    "Answer a question against the Grimore vault with retrieval-"
                    "augmented generation. Returns the answer, its evidence status, "
                    "model execution mode, wikilink labels and structured source details: "
                    "exact indexed passages, "
                    "page or heading, and whether the file is current, modified, "
                    "missing, or unverified. PDF details include indexed_extraction "
                    "coverage, which can be partial even when a cited passage "
                    "supports a specific claim. Citations the model fabricated (not "
                    "in the retrieved context) are stripped."
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
                    "fused relevance score. A source_id identifies the exact indexed "
                    "chunk for grimore_get_source; it is null for unresolved hits. "
                    "No chat completion is used; the query "
                    "is still sent to the configured embedding service."
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
                    "Fetch a note's metadata and complete current on-disk body. "
                    "Provide either note_id (preferred) or title (resolved via "
                    "exact match). No model service is called."
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
            "grimore_get_source": Tool(
                name="grimore_get_source",
                description=(
                    "Read one exact indexed passage using source_id from search, "
                    "ask's source_details.id, or an indexed review passage. Never "
                    "falls back to a title or similar excerpt. Returns locators, "
                    "extraction coverage and current/modified/missing file state. "
                    "An optional expected_indexed_file_hash rejects a reindexed "
                    "version. Legacy rows remain excerpts; no model or OCR service "
                    "is called. Optional Obsidian links open the current Markdown "
                    "file, not the indexed snapshot; other formats have no URI."
                ),
                input_schema={
                    "type": "object", "required": ["source_id"],
                    "additionalProperties": False,
                    "properties": {
                        "source_id": {"type": "string", "pattern": _SOURCE_ID.pattern},
                        "expected_indexed_file_hash": {
                            "type": "string", "pattern": "^[0-9a-f]{64}$",
                        },
                        "obsidian_vault": _OBSIDIAN_SCHEMA,
                    },
                },
                handler=self._tool_get_source,
            ),
            "grimore_review_inbox": Tool(
                name="grimore_review_inbox",
                description=(
                    "Read active maintenance candidates: contradictions, exact/near "
                    "duplicates and notes due for review. Returns evidence, coverage, "
                    "counts and explicit delivery limits. Candidates are not factual "
                    "verdicts. Does not confirm, dismiss, postpone or modify anything, "
                    "and does not call model services. Near-duplicate discovery is "
                    "bounded to 30 pairs; pagination is over a rebuilt snapshot, "
                    "not an exhaustive or frozen audit. Optional Obsidian links "
                    "open current Markdown documents only. No decision history."
                ),
                input_schema={
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": list(_REVIEW_KINDS)},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
                        "offset": {"type": "integer", "default": 0, "minimum": 0, "maximum": 1000000},
                        "obsidian_vault": _OBSIDIAN_SCHEMA,
                    },
                },
                handler=self._tool_review_inbox,
            ),
            "grimore_connect": Tool(
                name="grimore_connect",
                description=(
                    "Find notes related to a given note via cosine similarity "
                    "on vectors already stored in SQLite. No model service is "
                    "called. Useful for 'see also' suggestions."
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
                    "table-of-contents. No model service is called."
                ),
                input_schema={"type": "object", "properties": {}},
                handler=self._tool_list_categories,
            ),
        }

    # ── tool handlers ─────────────────────────────────────────────────

    def _tool_library_info(self, args: dict) -> dict:
        self._readonly_args(args, set())
        canonical = os.path.normcase(str(self.session.vault_root.resolve()))
        fingerprint = sha256(("grimore-vault-v1\0" + canonical).encode("utf-8")).hexdigest()
        return {
            "contract_version": 1,
            "vault_identity": {
                "algorithm": "sha256-local-path-v1", "fingerprint": fingerprint,
            },
            "model_execution": cognition_connectivity(self.session.config),
        }

    def _tool_ask(self, args: dict) -> dict:
        question = args.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is required and must be a non-empty string")
        # ValueError on a non-numeric value propagates to the dispatcher,
        # which maps it to _INVALID_PARAMS. The runtime limit matches the
        # maximum advertised by the input schema.
        top_k = SecurityGuard.coerce_top_k(args.get("top_k"), default=5, maximum=25)
        result = self.session.oracle.ask(question, top_k=top_k)
        return {
            "answer": result.get("answer", ""),
            "sources": list(result.get("sources") or []),
            "source_details": list(result.get("source_details") or []),
            "citation_selections": result.get("citation_selections"),
            **({"sufficiency": result["sufficiency"]}
               if isinstance(result.get("sufficiency"), dict) else {}),
            "answer_status": result.get("answer_status") or "uncertain",
            "model_execution": cognition_connectivity(self.session.config),
            "dropped_citations": int(result.get("dropped_citations") or 0),
            # Only when the model gave no answer: a client must not read a
            # timeout as insufficient evidence.
            **({"generation_error": result["generation_error"]}
               if result.get("generation_error") else {}),
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
                include_embedding_id=True,
            )
        elif query_vector:
            hits = self.session.oracle.connector.find_similar_notes(
                query_vector, top_k=top_k, include_embedding_id=True,
            )
        else:
            hits = []

        titles = self.session.db.get_note_titles(
            hit.get("note_id") for hit in hits
            if hit.get("note_id") is not None
        )
        rows: list[dict] = []
        ids = [hit["embedding_id"] for hit in hits if hit.get("embedding_id") is not None]
        records = self.session.db.get_evidence_records(ids) if ids else {}
        for hit in hits:
            note_id = hit.get("note_id")
            title = titles.get(int(note_id), "")
            text = hit.get("text") or ""
            snippet = text[:200] + ("…" if len(text) > 200 else "")
            record = records.get(hit.get("embedding_id"))
            if record and record.get("note_id") != int(note_id):
                record = None
            rows.append({
                "note_id": int(note_id),
                "title": title,
                "snippet": snippet,
                "score": float(hit.get("score") or 0.0),
                "source_id": self._record_source_id(record) if record else None,
                "indexed_file_hash": record.get("file_hash") if record else None,
            })
        return {"hits": rows}

    @staticmethod
    def _record_source_id(record: dict) -> str:
        return (f"note:{record['note_id']}:chunk:{record['chunk_index']}:"
                f"embedding:{record['embedding_id']}")

    @staticmethod
    def _readonly_args(args: dict, allowed: set[str]) -> str | None:
        if not isinstance(args, dict) or set(args) - allowed:
            raise ValueError("arguments must be an object with only the documented fields")
        vault = args.get("obsidian_vault")
        if "obsidian_vault" in args:
            if (not isinstance(vault, str) or not vault.strip() or len(vault) > 160
                    or any(ord(ch) < 32 for ch in vault)):
                raise ValueError("obsidian_vault must be a non-empty vault name or ID (max 160 characters)")
        return vault

    def _tool_get_source(self, args: dict) -> dict:
        vault = self._readonly_args(args, {
            "source_id", "expected_indexed_file_hash", "obsidian_vault",
        })
        source_id = args.get("source_id")
        match = _SOURCE_ID.fullmatch(source_id) if isinstance(source_id, str) else None
        if match is None or any(int(n) > 2**63 - 1 for n in match.groups()):
            raise ValueError("source_id must be an exact indexed source ID from Grimore")
        expected = args.get("expected_indexed_file_hash")
        if "expected_indexed_file_hash" in args and (
            not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("expected_indexed_file_hash must be a lowercase SHA-256 hash")
        embedding_id = int(match[3])
        record = self.session.db.get_evidence_records([embedding_id]).get(embedding_id)
        if record is None or self._record_source_id(record) != source_id:
            return {"found": False, "reason": "source_not_found"}
        if expected is not None and record.get("file_hash") != expected:
            return {"found": False, "reason": "source_version_changed"}
        # No legacy reconstruction: semantic chunking could call an embedder.
        # Stored evidence remains readable even when its original file is gone.
        detail = self.session.oracle._source_detail(
            {"note_id": record["note_id"]}, title=record["title"],
            label=_format_source_label(record["title"], record.get("page"), record.get("heading")),
            page=record.get("page"), heading=record.get("heading"), record=record,
            recover_legacy=False,
        )
        result = {"found": True, "source": detail}
        if vault is not None:
            result["open_target"] = open_target(
                vault=vault, path=detail["document"]["path"],
                format=detail["document"]["format"], heading=detail["locator"]["heading"],
                source_state=detail["source_state"],
            )
        return result

    def _review_document(self, document: dict, vault: str | None) -> dict:
        result = dict(document)
        raw_path = document.get("path")
        state = "unavailable"
        relative = None
        if raw_path:
            try:
                root = self.session.vault_root.resolve()
                resolved = SecurityGuard.resolve_within_vault(raw_path, root)
                relative = resolved.relative_to(root).as_posix()
                state = "unverified" if resolved.is_file() else "missing"
            except (OSError, ValueError):
                pass
        result["path"] = relative
        if vault is not None:
            result["open_target"] = open_target(
                vault=vault, path=relative,
                format="md" if relative and relative.lower().endswith(".md") else "unsupported",
                source_state=state,
            )
        return result

    def _tool_review_inbox(self, args: dict) -> dict:
        vault = self._readonly_args(args, {"kind", "limit", "offset", "obsidian_vault"})
        kind = args.get("kind")
        if "kind" in args and kind not in _REVIEW_KINDS:
            raise ValueError("kind must be one of the documented review kinds")
        limit, offset = args.get("limit", 20), args.get("offset", 0)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer between 1 and 50")
        if type(offset) is not int or not 0 <= offset <= 1000000:
            raise ValueError("offset must be an integer between 0 and 1000000")
        payload = build_review_inbox(self.session, duplicate_limit=30)
        matching = [item for item in payload["items"] if kind is None or item["kind"] == kind]
        items = []
        for item in matching[offset:offset + limit]:
            delivered = dict(item)
            delivered["documents"] = [
                self._review_document(doc, vault) for doc in item["documents"][:20]
            ]
            passages = []
            for passage in item["passages"][:20]:
                shown = self._review_document(passage, vault)
                text = passage.get("text") or ""
                shown["text"] = text[:3000]
                shown["text_truncated"] = len(text) > 3000
                if shown["text_truncated"]:
                    shown["complete"] = False
                passages.append(shown)
            delivered["passages"] = passages
            reason = item.get("reason") or ""
            delivered["reason"] = reason[:2000]
            delivered["delivery"] = {
                "documents_truncated": len(item["documents"]) > 20,
                "passages_truncated": len(item["passages"]) > 20,
                "reason_truncated": len(reason) > 2000,
            }
            items.append(delivered)
        has_more = offset + len(items) < len(matching)
        return {
            "generated_at": payload["generated_at"], "status": payload["status"],
            "counts": payload["counts"], "items": items,
            "delivery": {
                "kind": kind, "offset": offset, "limit": limit,
                "matched": len(matching), "returned": len(items), "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
                "snapshot": "rebuilt_per_call", "near_duplicate_limit": 30,
                "max_documents_per_item": 20, "max_passages_per_item": 20,
                "max_passage_characters": 3000, "max_reason_characters": 2000,
                "history_included": False,
            },
        }

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
        # not a full-text mirror). Re-assert vault containment first: this
        # is the one tool that turns a stored path into file bytes for the
        # client, so a tampered row or a symlink swapped after indexing
        # must not become an arbitrary-file read. Fail as a plain miss.
        try:
            resolved = SecurityGuard.resolve_within_vault(
                row["path"], self.session.vault_root,
            )
        except ValueError:
            logger.warning(
                "mcp_note_path_escapes_vault",
                note_id=row["note_id"], path=row["path"],
            )
            return {"found": False}

        body = ""
        extraction = None
        try:
            note = MarkdownParser().parse_file(
                resolved,
                vault_root=self.session.vault_root,
                config=self.session.config,
            )
            body = note.content
            extraction = extraction_diagnostics(note)
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("mcp_note_read_failed", path=row["path"], error=str(exc))
            extraction = {
                "status": "failed",
                "issues": [{"code": "document_read_failed", "page": None}],
            }
        return {
            "found": True,
            "note_id": row["note_id"],
            "title": row["title"],
            "path": row["path"],
            "body": body,
            "extraction": extraction,
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
            connectivity = cognition_connectivity(self.session.config)
            return _rpc_result(req_id, {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _SERVER_INFO,
                "instructions": (
                    f"Grimore is read-only. Model execution is {connectivity['mode']}; "
                    "tool results may contain vault passages or complete note bodies."
                ),
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
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                return _rpc_error(req_id, _INVALID_PARAMS, "arguments must be an object")
            try:
                with self._lock:
                    value = tool.handler(arguments)
            except ValueError as exc:
                return _rpc_error(req_id, _INVALID_PARAMS, str(exc))
            except Exception as exc:  # surface to the client without crashing the loop
                logger.error("mcp_tool_failed", tool=tool_name, error=str(exc))
                message = (
                    "Read-only query failed; see local logs."
                    if tool_name in {
                        "grimore_get_source", "grimore_review_inbox", "grimore_library_info",
                    } else str(exc)
                )
                return _rpc_result(req_id, {
                    "content": _content_text({"error": message}),
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
    wire = sys.stdout
    # Adapters and optional libraries may print diagnostics. Reserve stdout
    # for protocol frames and direct incidental output to the drained stderr.
    with redirect_stdout(sys.stderr):
        server = MCPServer(session=session)
        server.serve(stdout=wire)
