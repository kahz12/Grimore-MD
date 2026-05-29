"""Grimore HTTP API + minimal web UI.

Read-only ASGI app exposing the same surface as the MCP server. Built
on Starlette directly so the install footprint stays Termux-friendly
(no pydantic-core required).

Security defaults:

* Bind loopback only — the CLI's ``serve`` command enforces this
  before reaching ASGI.
* When ``api_token`` is set, every POST requires
  ``Authorization: Bearer <token>``. GETs and the UI stay open so the
  loopback browser flow is friction-free.
* CORS off by default. ``cors_origin`` (single origin, no wildcards)
  is the only escape valve. Plays well with the typical "serve on
  loopback, point a browser at it" flow.
"""
from __future__ import annotations

import json
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

from grimore.session import Session
from grimore.utils.logger import get_logger

logger = get_logger(__name__)

# Discovered relative to this file so editable installs and packaged
# installs both pick the right asset path.
_HERE = Path(__file__).resolve().parent
_STATIC_DIR = _HERE / "static"
_TEMPLATES_DIR = _HERE / "templates"
_INDEX_HTML = _TEMPLATES_DIR / "index.html"

_API_VERSION = "2.4.0"


# ── auth ──────────────────────────────────────────────────────────────


def _check_auth(request: Request, api_token: Optional[str]) -> Optional[Response]:
    """Return a 401 ``Response`` if the request lacks a valid bearer.

    None means "auth passed (or not required)". Called from each POST
    handler so we don't gate GETs / the UI.
    """
    if not api_token:
        return None
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return JSONResponse({"error": "missing bearer token"}, status_code=401)
    presented = header[len("Bearer "):].strip()
    if presented != api_token:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return None


# ── handlers ──────────────────────────────────────────────────────────


def _build_routes(session: Session, *, api_token: Optional[str]) -> list:
    """Wire each Starlette route to a closure that captures the warm
    Session and the configured token. Defined inline so test setups can
    swap a MagicMock Session per app build.
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
        unauthorized = _check_auth(request, api_token)
        if unauthorized is not None:
            return unauthorized

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)

        question = (body or {}).get("question", "")
        if not isinstance(question, str) or not question.strip():
            return JSONResponse(
                {"error": "question is required"}, status_code=400,
            )
        top_k = int(body.get("top_k", 5) or 5)
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
        unauthorized = _check_auth(request, api_token)
        if unauthorized is not None:
            return unauthorized

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)

        query = (body or {}).get("query", "")
        if not isinstance(query, str) or not query.strip():
            return JSONResponse({"error": "query is required"}, status_code=400)
        top_k = int(body.get("top_k", 10) or 10)

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
        # GETs are always open — they're read-only against vault metadata.
        try:
            note_id = int(request.path_params["note_id"])
        except (ValueError, KeyError):
            return JSONResponse({"error": "note_id must be an integer"}, status_code=400)

        loc = session.db.get_note_location(note_id)
        if loc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path, title = loc
        body_text = ""
        try:
            body_text = Path(path).read_text(encoding="utf-8", errors="replace")
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
) -> Starlette:
    """Construct the ASGI app.

    Called by ``grimore serve`` and by tests that drive the routes via
    ``starlette.testclient.TestClient``. The session is captured by
    closure so requests reuse the warm embedder + router.
    """
    middleware: list[Middleware] = []
    if cors_origin:
        middleware.append(Middleware(
            CORSMiddleware,
            allow_origins=[cors_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["authorization", "content-type"],
            allow_credentials=False,
        ))

    return Starlette(
        debug=False,
        routes=_build_routes(session, api_token=api_token),
        middleware=middleware,
    )
