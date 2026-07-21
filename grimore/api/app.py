"""Grimore HTTP API + minimal web UI.

Read-only ASGI app exposing the same surface as the MCP server. Built
on Starlette directly so the install footprint stays Termux-friendly
(no pydantic-core required).

Security defaults:

* Bind loopback only — the CLI's ``serve`` command enforces this
  before reaching ASGI.
* When ``api_token`` is set, every ``/api/*`` request from a
  non-loopback client requires ``Authorization: Bearer <token>`` — GET
  included, so note bodies can't be read without the token. Loopback
  clients (the local browser UI, which sends no token) stay open, as do
  the ``/`` UI shell and static assets. Enforced centrally in
  ``_TokenAuthMiddleware`` so no route can forget the check; the token
  comparison is constant-time, and repeated failures from one peer are
  throttled (429). ``strict_token`` drops the loopback exemption for
  hosts where localhost isn't a trust boundary (Android/Termux).
* CORS off by default. ``cors_origin`` (single origin, no wildcards)
  is the only escape valve. Plays well with the typical "serve on
  loopback, point a browser at it" flow.
"""
from __future__ import annotations

import ipaddress
import json
import secrets
import time
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from grimore.session import Session
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Discovered relative to this file so editable installs and packaged
# installs both pick the right asset path.
_HERE = Path(__file__).resolve().parent
_STATIC_DIR = _HERE / "static"
_TEMPLATES_DIR = _HERE / "templates"
_INDEX_HTML = _TEMPLATES_DIR / "index.html"

_API_VERSION = "2.4.0"


# ── auth ──────────────────────────────────────────────────────────────


def _client_is_loopback(scope: Scope) -> bool:
    """Whether the peer that opened this connection is loopback.

    Reads the real transport peer from ``scope['client']`` (uvicorn sets
    it to the actual socket address). We deliberately ignore
    ``X-Forwarded-For`` / ``Forwarded`` headers — trusting a
    client-supplied header here would re-open the very hole this gate
    closes. Unknown or non-IP peers fail closed (treated as remote).
    """
    client = scope.get("client")
    if not client:
        return False
    try:
        ip = ipaddress.ip_address(client[0])
    except (ValueError, IndexError, TypeError):
        return False
    if ip.is_loopback:
        return True
    # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) when bound dual-stack.
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def _request_has_valid_bearer(scope: Scope, api_token: str) -> bool:
    """Constant-time check of the ``Authorization: Bearer`` header.

    The comparison is done on raw bytes, never decoded text:
    ``secrets.compare_digest`` rejects non-ASCII ``str`` with a
    ``TypeError``, so a header carrying high bytes (a malformed or
    probing token) would otherwise raise and surface as a 500 instead of
    a clean 401. Bytes side-step that — a bad token simply fails to match.
    """
    token = api_token.encode("utf-8")
    for key, value in scope.get("headers") or []:
        if key == b"authorization":
            if not isinstance(value, (bytes, bytearray)) or not value.startswith(b"Bearer "):
                return False
            presented = bytes(value[len(b"Bearer "):]).strip()
            return secrets.compare_digest(presented, token)
    return False


async def _send_401(send: Send, message: str) -> None:
    body = json.dumps({"error": message}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": body})


async def _send_429(send: Send, retry_after: int) -> None:
    body = json.dumps({
        "error": "too many failed authentication attempts; retry later",
    }).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 429,
        "headers": [
            (b"content-type", b"application/json"),
            (b"retry-after", str(retry_after).encode("ascii")),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": body})


class _TokenAuthMiddleware:
    """Gate every ``/api/*`` route behind the bearer token.

    Closes the broken-access-control hole (audit H1): the previous
    per-handler check ran only on POSTs, leaving the GET note / category
    / health routes — i.e. the actual data — open to any host that could
    reach the port. Enforcing it here, in one place, means a route added
    later can't forget the check. Loopback callers are exempt so the
    local browser UI (which sends no token) keeps working; remote callers
    need the token for *all* methods, GET included.

    Only attached when a token is configured, so the no-token loopback
    deployment is byte-for-byte unchanged.

    ``exempt_loopback=False`` (the ``--strict-token`` serve flag) drops
    the loopback exemption: on Android/Termux any app on the device can
    reach localhost ports, so "loopback" is not "same trust domain" there.

    Failed attempts are throttled per peer: after ``MAX_FAILURES`` bad
    tokens inside ``WINDOW_SECONDS``, further attempts from that address
    get a 429 until the window expires. The constant-time compare already
    blunts timing attacks; this bounds the online guess *rate* on a LAN
    bind. State is in-memory and per-process — matching the single-process
    uvicorn deployment ``serve`` runs.
    """

    MAX_FAILURES = 10
    WINDOW_SECONDS = 60.0
    # Bound the throttle map so a spoofed-address sweep can't grow it
    # without limit.
    _MAX_TRACKED_PEERS = 1024

    def __init__(
        self, app: ASGIApp, *, api_token: str, exempt_loopback: bool = True,
    ) -> None:
        self.app = app
        self.api_token = api_token
        self.exempt_loopback = exempt_loopback
        self._failures: dict[str, tuple[float, int]] = {}

    @staticmethod
    def _peer_key(scope: Scope) -> str:
        client = scope.get("client")
        try:
            return str(client[0])
        except (TypeError, IndexError):
            return "?"

    def _is_throttled(self, peer: str, now: float) -> bool:
        entry = self._failures.get(peer)
        if entry is None:
            return False
        start, count = entry
        if now - start >= self.WINDOW_SECONDS:
            del self._failures[peer]
            return False
        return count >= self.MAX_FAILURES

    def _record_failure(self, peer: str, now: float) -> None:
        start, count = self._failures.get(peer, (now, 0))
        if now - start >= self.WINDOW_SECONDS:
            start, count = now, 0
        self._failures[peer] = (start, count + 1)
        if len(self._failures) > self._MAX_TRACKED_PEERS:
            expired = [
                k for k, (s, _) in self._failures.items()
                if now - s >= self.WINDOW_SECONDS
            ]
            for k in expired:
                del self._failures[k]
            # Everything still live: evict oldest windows so the newest
            # (most relevant) offenders stay tracked.
            while len(self._failures) > self._MAX_TRACKED_PEERS:
                oldest = min(self._failures, key=lambda k: self._failures[k][0])
                del self._failures[oldest]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        # Non-/api/ paths (UI shell, static assets) stay open. OPTIONS is
        # an unauthenticated CORS preflight carrying no credentials.
        if not path.startswith("/api/") or method == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if self.exempt_loopback and _client_is_loopback(scope):
            await self.app(scope, receive, send)
            return
        peer = self._peer_key(scope)
        now = time.monotonic()
        if self._is_throttled(peer, now):
            await _send_429(send, retry_after=int(self.WINDOW_SECONDS))
            return
        if _request_has_valid_bearer(scope, self.api_token):
            self._failures.pop(peer, None)
            await self.app(scope, receive, send)
            return
        self._record_failure(peer, now)
        await _send_401(send, "valid bearer token required")


# ── handlers ──────────────────────────────────────────────────────────


def _build_routes(session: Session) -> list:
    """Wire each Starlette route to a closure that captures the warm
    Session. Defined inline so test setups can swap a MagicMock Session
    per app build. Auth is handled upstream by ``_TokenAuthMiddleware``,
    not in these handlers.
    """

    async def health(request: Request) -> Response:
        return JSONResponse({
            "ok": True,
            "version": _API_VERSION,
            "vault": str(session.vault_root),
            "fts": bool(session.db.fts_available),
            "vec": bool(session.db.vec_available),
        })

    async def ask(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)

        question = (body or {}).get("question", "")
        if not isinstance(question, str) or not question.strip():
            return JSONResponse(
                {"error": "question is required"}, status_code=400,
            )
        try:
            top_k = SecurityGuard.coerce_top_k(body.get("top_k"), default=5)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        history = body.get("history") or None
        stream = bool(body.get("stream"))

        if stream:
            async def event_stream():
                # Oracle.ask_stream is a sync generator; wrap it via
                # Starlette's iterate_in_threadpool so a long Ollama
                # token wait doesn't block the event loop and stall
                # other in-flight requests.
                gen = session.oracle.ask_stream(
                    question, top_k=top_k, history=history,
                )
                async for event in iterate_in_threadpool(gen):
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"data: {payload}\n\n".encode("utf-8")

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = session.oracle.ask(question, top_k=top_k, history=history)
        return JSONResponse({
            "answer": result.get("answer", ""),
            "sources": list(result.get("sources") or []),
            "dropped_citations": int(result.get("dropped_citations") or 0),
        })

    async def search(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)

        query = (body or {}).get("query", "")
        if not isinstance(query, str) or not query.strip():
            return JSONResponse({"error": "query is required"}, status_code=400)
        try:
            top_k = SecurityGuard.coerce_top_k(body.get("top_k"), default=10)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        query_vector = session.embedder.embed(query)
        use_hybrid = (
            getattr(session.config.cognition, "hybrid_search", True)
            and session.db.fts_available
        )
        if use_hybrid:
            hits = session.oracle.connector.find_hybrid(
                query_text=query, query_vector=query_vector, top_k=top_k,
            )
        elif query_vector:
            hits = session.oracle.connector.find_similar_notes(query_vector, top_k=top_k)
        else:
            hits = []

        rows = []
        for hit in hits:
            note_id = hit.get("note_id")
            title = session.db.get_note_title(note_id) or ""
            text = hit.get("text") or ""
            snippet = text[:200] + ("…" if len(text) > 200 else "")
            rows.append({
                "note_id": int(note_id),
                "title": title,
                "snippet": snippet,
                "score": float(hit.get("score") or 0.0),
            })
        return JSONResponse({"hits": rows})

    async def get_note(request: Request) -> Response:
        # When a token is set, _TokenAuthMiddleware has already required
        # it for non-loopback callers before we reach this handler.
        try:
            note_id = int(request.path_params["note_id"])
        except (ValueError, KeyError):
            return JSONResponse({"error": "note_id must be an integer"}, status_code=400)

        loc = session.db.get_note_location(note_id)
        if loc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path, title = loc
        # The DB is normally trustworthy, but this is the one route that
        # turns a stored path into file bytes for a (possibly remote)
        # caller — so re-assert vault containment, catching a tampered
        # row or a symlink swapped after indexing. Fail as a plain 404.
        try:
            resolved = SecurityGuard.resolve_within_vault(path, session.vault_root)
        except ValueError:
            logger.warning("api_note_path_escapes_vault", note_id=note_id, path=path)
            return JSONResponse({"error": "not found"}, status_code=404)
        body_text = ""
        try:
            body_text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("api_note_read_failed", path=path, error=str(exc))
        return JSONResponse({
            "note_id": note_id,
            "title": title,
            "path": path,
            "body": body_text,
        })

    async def categories(request: Request) -> Response:
        rows = session.db.get_category_frequency()
        return JSONResponse({
            "categories": [
                {"category": cat, "count": int(n)} for cat, n in rows
            ],
        })

    async def index(request: Request) -> Response:
        # Serve the SPA shell. Template is a flat HTML file (no Jinja
        # variables) so we just stream the bytes.
        if not _INDEX_HTML.exists():
            return PlainTextResponse(
                "Grimore web UI assets missing. Reinstall the serve extra.",
                status_code=500,
            )
        return FileResponse(_INDEX_HTML, media_type="text/html; charset=utf-8")

    routes = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/notes/{note_id:int}", get_note, methods=["GET"]),
        Route("/api/categories", categories, methods=["GET"]),
        Route("/", index, methods=["GET"]),
    ]
    if _STATIC_DIR.exists():
        routes.append(Mount("/static", app=StaticFiles(directory=_STATIC_DIR), name="static"))
    return routes


def build_app(
    session: Session,
    *,
    api_token: Optional[str] = None,
    cors_origin: Optional[str] = None,
    strict_token: bool = False,
) -> Starlette:
    """Construct the ASGI app.

    Called by ``grimore serve`` and by tests that drive the routes via
    ``starlette.testclient.TestClient``. The session is captured by
    closure so requests reuse the warm embedder + router.

    ``strict_token`` drops the loopback exemption so the token is required
    from local clients too (Android/Termux: localhost is reachable by any
    app on the device). It only makes sense with a token, so it raises
    without one rather than silently serving unauthenticated.
    """
    if strict_token and not api_token:
        raise ValueError("strict_token requires api_token")
    # Order matters: CORS is listed first so it's the outermost layer and
    # answers preflight OPTIONS before auth runs; the token gate sits just
    # inside it, in front of the routes.
    middleware: list[Middleware] = []
    if cors_origin:
        middleware.append(Middleware(
            CORSMiddleware,
            allow_origins=[cors_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["authorization", "content-type"],
            allow_credentials=False,
        ))
    if api_token:
        middleware.append(Middleware(
            _TokenAuthMiddleware,
            api_token=api_token,
            exempt_loopback=not strict_token,
        ))

    return Starlette(
        debug=False,
        routes=_build_routes(session),
        middleware=middleware,
    )
