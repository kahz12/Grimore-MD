"""Grimore HTTP API + minimal web UI.

ASGI app exposing read operations plus explicitly scoped writes for setup,
indexing, review decisions, and document classification corrections. Built on Starlette directly so the
install footprint stays Termux-friendly (no pydantic-core required).

Security defaults:

* Bind loopback only — the CLI's ``serve`` command enforces this
  before reaching ASGI.
* When ``api_token`` is set, every ``/api/*`` request from a
  non-loopback client requires ``Authorization: Bearer <token>`` — GET
  included, so note bodies can't be read without the token. Loopback
  read requests stay open by default, as do
  the ``/`` UI shell and static assets. Enforced centrally in
  ``_TokenAuthMiddleware`` so no route can forget the check; the token
  comparison is constant-time, and repeated failures from one peer are
  throttled (429). ``strict_token`` drops the loopback exemption for
  hosts where localhost isn't a trust boundary (Android/Termux).
* CORS off by default. ``cors_origin`` (single origin, no wildcards)
  is the only escape valve. Plays well with the typical "serve on
  loopback, point a browser at it" flow.
* Mutating routes are disabled unless ``api_token`` is configured and always
  require that bearer token, including from loopback clients.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import queue
import secrets
import sys
import threading
import time
import tomllib
from datetime import datetime, timedelta, timezone
from functools import partial
from contextlib import aclosing
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.cognition.review_inbox import build_review_inbox
from grimore.cognition.synthesizer import Synthesizer
from grimore.output.change_journal import ChangeConflictError, ChangeJournal
from grimore.session import Session
from grimore.utils.logger import get_logger
from grimore.utils.connectivity import cognition_connectivity
from grimore.utils.config import update_setup_config
from grimore.utils.index_progress import read_index_progress, recovery_actions
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Starlette's AnyIO bridge has had compatibility gaps on newly released
# Python versions. Bounded, short-lived workers keep model/SQLite work off the
# event loop without relying on an executor whose idle threads outlive loops.
_API_WORKER_SLOTS = threading.BoundedSemaphore(8)


async def _acquire_worker_slot() -> None:
    while not _API_WORKER_SLOTS.acquire(blocking=False):
        await asyncio.sleep(0.005)


async def _run_blocking(func, *args, **kwargs):
    await _acquire_worker_slot()
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put((True, partial(func, *args, **kwargs)()))
        except BaseException as exc:
            result_queue.put((False, exc))
        finally:
            _API_WORKER_SLOTS.release()

    threading.Thread(
        target=_worker, name="grimore-api-worker", daemon=True,
    ).start()
    while True:
        try:
            ok, result = result_queue.get_nowait()
            break
        except queue.Empty:
            await asyncio.sleep(0.005)
    if ok:
        return result
    raise result


async def _iterate_blocking(iterator):
    """Advance only on consumer demand; close the iterator on its own thread.

    Cancellation cannot interrupt a blocking next() already in progress, but
    it prevents every subsequent advance and releases the slot when that call
    returns (model I/O has its own timeout).
    """
    await _acquire_worker_slot()
    event_queue: queue.Queue = queue.Queue(maxsize=1)
    done = object()
    demand = threading.Event()
    stopped = threading.Event()

    def _producer() -> None:
        try:
            while True:
                demand.wait()
                demand.clear()
                if stopped.is_set():
                    break
                try:
                    item = next(iterator)
                except StopIteration:
                    event_queue.put((True, done))
                    break
                if stopped.is_set():
                    break
                event_queue.put((True, item))
        except BaseException as exc:
            event_queue.put((False, exc))
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            finally:
                _API_WORKER_SLOTS.release()

    try:
        threading.Thread(
            target=_producer, name="grimore-api-stream", daemon=True,
        ).start()
    except BaseException:
        _API_WORKER_SLOTS.release()
        raise
    try:
        while True:
            demand.set()
            while True:
                try:
                    ok, item = event_queue.get_nowait()
                    break
                except queue.Empty:
                    await asyncio.sleep(0.005)
            if not ok:
                raise item
            if item is done:
                break
            yield item
    finally:
        stopped.set()
        demand.set()

# Discovered relative to this file so editable installs and packaged
# installs both pick the right asset path.
_HERE = Path(__file__).resolve().parent
_STATIC_DIR = _HERE / "static"
_TEMPLATES_DIR = _HERE / "templates"
_INDEX_HTML = _TEMPLATES_DIR / "index.html"

# Read from the package rather than restated here: this constant had drifted
# to 2.4.0 while the package was at 3.2.0, and a client reading the API version
# has no way to notice.
from grimore import __version__ as _API_VERSION  # noqa: E402


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

    Authentication lives in middleware so every current and future API route
    follows the same access policy. Loopback callers are exempt for the read
    surface by default; remote callers need the token for *all* methods, GET
    included. A separate inner gate always protects the write routes.

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


class _WriteTokenAuthMiddleware(_TokenAuthMiddleware):
    """Require the bearer token for every mutating route, even locally."""

    WRITE_ROUTES = {"/api/setup/apply", "/api/indexing/start"}

    @classmethod
    def _is_write_route(cls, path: str, method: str) -> bool:
        return (
            path in cls.WRITE_ROUTES
            or (
                method == "POST"
                and path.startswith("/api/review-inbox/")
                and path.endswith("/decision")
            )
            or (
                method == "POST"
                and path.startswith("/api/notes/")
                and path.endswith("/classification")
            )
            or (
                method == "POST"
                and path.startswith("/api/changes/")
                and path.endswith("/restore")
            )
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") == "OPTIONS"
            or not self._is_write_route(
                scope.get("path", ""), scope.get("method", "GET"),
            )
        ):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


# ── handlers ──────────────────────────────────────────────────────────


def _build_routes(
    session: Session,
    *,
    write_enabled: bool = False,
    config_path: str | Path = "grimore.toml",
    scan_launcher=None,
) -> list:
    """Wire each Starlette route to a closure that captures the warm
    Session. Defined inline so test setups can swap a MagicMock Session
    per app build. Mutating handlers remain unavailable without a configured
    token; middleware authenticates them even for loopback clients.
    """
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = Path.cwd() / resolved_config_path
    configured_vault = getattr(getattr(session, "config", None), "vault", None)
    fixed_vault_root = Path(
        getattr(session, "vault_root", None)
        or getattr(configured_vault, "path", None)
        or Path.cwd()
    ).expanduser().resolve()
    scan_operation: dict = {
        "process": None,
        "task": None,
        "mode": None,
        "started_at": None,
        "exit_code": None,
    }
    config_operation = {"restart_required": False}

    async def _default_scan_launcher(*, dry_run: bool):
        flag = "--dry-run" if dry_run else "--no-dry-run"
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "grimore.cli",
            "scan",
            "--vault-path",
            str(fixed_vault_root),
            flag,
            cwd=str(Path.cwd()),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    launch_scan = scan_launcher or _default_scan_launcher

    async def _read_write_body(request: Request):
        """Read a small JSON object without permitting unbounded write bodies."""
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 16_384:
                return None, JSONResponse({
                    "error": "request body is too large",
                }, status_code=413)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return None, JSONResponse({
                "error": "body must be a JSON object",
            }, status_code=400)
        return body, None

    async def health(request: Request) -> Response:
        return JSONResponse({
            "ok": True,
            "version": _API_VERSION,
            "vault": str(session.vault_root),
            "fts": bool(session.db.fts_available),
            "vec": bool(session.db.vec_available),
            "cognition": cognition_connectivity(session.config),
        })

    def _model_available(selected: str, available: list[str]) -> bool:
        if selected in available:
            return True
        if selected and ":" not in selected:
            prefix = selected + ":"
            return any(name.startswith(prefix) for name in available)
        return False

    async def setup_summary(request: Request) -> Response:
        """Read-only setup assistant data for the bundled web UI.

        This endpoint validates only the configured vault.  It deliberately
        accepts no candidate path and performs no writes, preventing the
        onboarding surface from becoming an arbitrary filesystem probe or a
        premature configuration-mutation API.
        """
        config = session.config
        configured_path = str(config.vault.path)

        def _inspect_vault() -> dict:
            path = Path(configured_path).expanduser()
            exists = path.exists()
            is_directory = exists and path.is_dir()
            readable = is_directory and os.access(path, os.R_OK)
            return {
                "path": configured_path,
                "exists": exists,
                "is_directory": is_directory,
                "readable": readable,
                "valid": bool(exists and is_directory and readable),
                "formats": list(config.vault.formats),
            }

        vault = await _run_blocking(_inspect_vault)

        def _list_models() -> tuple[list[dict], list[dict]]:
            try:
                chat = list(session.router.list_installed_models() or [])
            except Exception as exc:
                logger.warning("setup_chat_models_unavailable", error=str(exc))
                chat = []

            backend = (
                getattr(config.cognition, "llm_backend", "ollama") or "ollama"
            ).lower()
            if backend == "ollama":
                embedding = list(chat)
            else:
                try:
                    embedding = list(session.embedder.list_installed_models() or [])
                except Exception as exc:
                    logger.warning("setup_embedding_models_unavailable", error=str(exc))
                    embedding = []
            return chat, embedding

        chat_rows, embedding_rows = await _run_blocking(_list_models)
        chat_names = [str(row.get("name") or "") for row in chat_rows if row.get("name")]
        embedding_names = [
            str(row.get("name") or "") for row in embedding_rows if row.get("name")
        ]
        selected_chat = str(config.cognition.model_llm_local)
        selected_embedding = str(config.cognition.model_embeddings_local)
        chat_ok = _model_available(selected_chat, chat_names)
        embedding_ok = _model_available(selected_embedding, embedding_names)

        checks = [
            {
                "id": "vault",
                "status": "pass" if vault["valid"] else "error",
                "message": (
                    "Configured folder is readable."
                    if vault["valid"] else
                    "Configured folder is missing, unreadable, or not a directory."
                ),
                "fix": "Update [vault].path, restart Grimore, and validate again.",
            },
            {
                "id": "chat_model",
                "status": "pass" if chat_ok else "error",
                "message": (
                    f"Chat model {selected_chat!r} is available."
                    if chat_ok else
                    f"Chat model {selected_chat!r} was not reported by the backend."
                ),
                "fix": "Select a reported chat model or install it in the configured backend.",
            },
            {
                "id": "embedding_model",
                "status": "pass" if embedding_ok else "error",
                "message": (
                    f"Embedding model {selected_embedding!r} is available."
                    if embedding_ok else
                    f"Embedding model {selected_embedding!r} was not reported by Ollama."
                ),
                "fix": "Select a reported embedding model or pull it with Ollama.",
            },
        ]
        config_present = (Path.cwd() / "grimore.toml").is_file()
        if not config_present:
            checks.append({
                "id": "config_file",
                "status": "warning",
                "message": "No grimore.toml exists in the server working directory.",
                "fix": "Create grimore.toml there, then restart Grimore.",
            })

        return JSONResponse({
            "read_only": True,
            "ready": vault["valid"] and chat_ok and embedding_ok,
            "config_present": config_present,
            "vault": vault,
            "models": {
                "chat": {
                    "selected": selected_chat,
                    "available": chat_names,
                    "valid": chat_ok,
                },
                "embeddings": {
                    "selected": selected_embedding,
                    "available": embedding_names,
                    "valid": embedding_ok,
                },
            },
            "checks": checks,
            "scan_effects": {
                "preview": [
                    "Reads changed documents and sends their extracted text to the chat model.",
                    "Shows proposed tags, summary, and category.",
                    "Does not rewrite documents, sidecars, or indexed note records.",
                ],
                "apply": [
                    "Writes generated frontmatter into changed Markdown files.",
                    "Creates or refreshes managed sidecars for non-Markdown files when enabled.",
                    "Stores note metadata, tags, categories, chunks, and embeddings in SQLite.",
                    "Creates a Git safety snapshot before document changes when enabled and available.",
                ],
                "never": [
                    "Never modifies the original PDF, EPUB, DOCX, ODT, RTF, HTML, or TXT file.",
                    "Skips documents marked privacy: never_process.",
                ],
            },
            "operations": {
                "write_enabled": write_enabled,
                "requires_bearer": True,
                "restart_required": config_operation["restart_required"],
                "config_fields": [
                    "vault_path", "chat_model", "embedding_model",
                ],
                "index_modes": ["preview", "apply"],
                "review_decisions": ["confirm", "dismiss", "postpone"],
                "classification_fields": ["category", "tags"],
                "document_restore": True,
            },
        })

    async def apply_setup(request: Request) -> Response:
        """Persist only the three setup fields; every other key is rejected."""
        if not write_enabled:
            return JSONResponse({
                "error": "write operations are disabled; restart serve with an API token",
            }, status_code=503)
        body, error_response = await _read_write_body(request)
        if error_response is not None:
            return error_response
        allowed = {"vault_path", "chat_model", "embedding_model", "confirm"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return JSONResponse({
                "error": f"unsupported configuration fields: {', '.join(unknown)}",
            }, status_code=400)
        if body.get("confirm") != "apply-config":
            return JSONResponse({
                "error": "confirm must be 'apply-config'",
            }, status_code=400)

        values = {}
        limits = {"vault_path": 4096, "chat_model": 200, "embedding_model": 200}
        for key, limit in limits.items():
            value = body.get(key)
            if not isinstance(value, str) or not value.strip():
                return JSONResponse({"error": f"{key} is required"}, status_code=400)
            value = value.strip()
            if len(value) > limit or any(ord(char) < 32 for char in value):
                return JSONResponse({"error": f"{key} is invalid"}, status_code=400)
            values[key] = value

        try:
            candidate_vault = await _run_blocking(
                lambda: Path(values["vault_path"]).expanduser().resolve(strict=True),
            )
        except (OSError, RuntimeError):
            return JSONResponse({
                "error": "vault_path must name an existing directory",
            }, status_code=400)
        if not candidate_vault.is_dir() or not os.access(candidate_vault, os.R_OK):
            return JSONResponse({
                "error": "vault_path must be a readable directory",
            }, status_code=400)

        try:
            chat_rows = await _run_blocking(session.router.list_installed_models)
            backend = (
                getattr(session.config.cognition, "llm_backend", "ollama") or "ollama"
            ).lower()
            embedding_rows = chat_rows if backend == "ollama" else await _run_blocking(
                session.embedder.list_installed_models,
            )
        except Exception as exc:
            logger.warning("api_setup_models_unavailable", error=str(exc))
            return JSONResponse({
                "error": "model inventory is unavailable; configuration was not changed",
            }, status_code=503)

        def _names(rows):
            return {str(row.get("name")) for row in (rows or []) if row.get("name")}

        allowed_chat = _names(chat_rows) | {
            str(session.config.cognition.model_llm_local),
        }
        allowed_embedding = _names(embedding_rows) | {
            str(session.config.cognition.model_embeddings_local),
        }
        if values["chat_model"] not in allowed_chat:
            return JSONResponse({"error": "chat_model was not reported"}, status_code=400)
        if values["embedding_model"] not in allowed_embedding:
            return JSONResponse({
                "error": "embedding_model was not reported",
            }, status_code=400)

        try:
            changed = await _run_blocking(
                update_setup_config,
                vault_path=str(candidate_vault),
                chat_model=values["chat_model"],
                embedding_model=values["embedding_model"],
                config_path=resolved_config_path,
            )
        except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as exc:
            logger.warning("api_setup_write_failed", error=str(exc))
            return JSONResponse({
                "error": f"configuration was not changed: {exc}",
            }, status_code=409)
        config_operation["restart_required"] = (
            config_operation["restart_required"] or bool(changed)
        )
        return JSONResponse({
            "changed": bool(changed),
            "restart_required": config_operation["restart_required"],
            "config_path": resolved_config_path.name,
            "values": {
                "vault_path": str(candidate_vault),
                "chat_model": values["chat_model"],
                "embedding_model": values["embedding_model"],
            },
        })

    def _public_scan_operation() -> dict:
        process = scan_operation["process"]
        return {
            "write_enabled": write_enabled,
            "active": bool(process is not None and process.returncode is None),
            "mode": scan_operation["mode"],
            "started_at": scan_operation["started_at"],
            "exit_code": scan_operation["exit_code"],
        }

    async def _watch_scan(process) -> None:
        try:
            exit_code = await process.wait()
        except Exception as exc:
            logger.error("api_index_process_wait_failed", error=str(exc))
            exit_code = -1
        if scan_operation["process"] is process:
            scan_operation["exit_code"] = int(exit_code)

    async def start_indexing(request: Request) -> Response:
        """Start one fixed CLI scan against the already loaded vault."""
        if not write_enabled:
            return JSONResponse({
                "error": "write operations are disabled; restart serve with an API token",
            }, status_code=503)
        body, error_response = await _read_write_body(request)
        if error_response is not None:
            return error_response
        if set(body) - {"mode", "confirm"}:
            return JSONResponse({"error": "unsupported indexing fields"}, status_code=400)
        mode = body.get("mode")
        if mode not in {"preview", "apply"}:
            return JSONResponse({
                "error": "mode must be 'preview' or 'apply'",
            }, status_code=400)
        if mode == "apply" and body.get("confirm") != "apply-indexing":
            return JSONResponse({
                "error": "confirm must be 'apply-indexing' for apply mode",
            }, status_code=400)
        if config_operation["restart_required"]:
            return JSONResponse({
                "error": "restart Grimore before indexing with the new configuration",
            }, status_code=409)

        process = scan_operation["process"]
        if process is not None and process.returncode is None:
            return JSONResponse({
                "error": "an indexing operation is already running",
            }, status_code=409)
        runs = await _run_blocking(read_index_progress, fixed_vault_root)
        if any(run.get("status") == "running" for run in runs):
            return JSONResponse({
                "error": "an indexing operation is already running",
            }, status_code=409)
        try:
            process = await launch_scan(dry_run=mode == "preview")
        except (OSError, RuntimeError) as exc:
            logger.error("api_index_start_failed", error=str(exc))
            return JSONResponse({
                "error": f"could not start indexing: {exc}",
            }, status_code=500)
        scan_operation.update({
            "process": process,
            "mode": mode,
            "started_at": time.time(),
            "exit_code": None,
        })
        scan_operation["task"] = asyncio.create_task(_watch_scan(process))
        return JSONResponse({
            "accepted": True,
            "operation": _public_scan_operation(),
        }, status_code=202)

    async def indexing_status(request: Request) -> Response:
        """Expose durable CLI/daemon progress without starting a scan."""
        runs = await _run_blocking(read_index_progress, session.vault_root)
        try:
            changes = await _run_blocking(
                ChangeJournal(fixed_vault_root).list_change_sets,
            )
        except OSError:
            changes = []
        public_runs = []
        for run in runs:
            current = dict(run)
            total = current.get("total")
            completed = int((current.get("counts") or {}).get("completed", 0))
            current["progress"] = (
                min(1.0, completed / total) if isinstance(total, int) and total > 0
                else (1.0 if total == 0 and current.get("status") != "running" else None)
            )
            public_runs.append(current)
        operation = _public_scan_operation()
        return JSONResponse({
            "active": (
                any(run.get("status") == "running" for run in public_runs)
                or operation["active"]
            ),
            "latest": public_runs[0] if public_runs else None,
            "runs": public_runs,
            "recovery_actions": recovery_actions(public_runs),
            "operation": operation,
            "changes": changes,
        })

    async def change_history(request: Request) -> Response:
        """List bounded, recoverable document changes for this vault."""
        try:
            changes = await _run_blocking(
                ChangeJournal(fixed_vault_root).list_change_sets,
            )
        except OSError as exc:
            logger.warning("api_change_history_unavailable", error=str(exc))
            return JSONResponse({
                "error": "change history is unavailable",
            }, status_code=503)
        return JSONResponse({
            "changes": changes,
            "retention_limit": 20,
        })

    async def restore_change_set(request: Request) -> Response:
        """Restore copied pre-change bytes after checking for later edits."""
        if not write_enabled:
            return JSONResponse({
                "error": "write operations are disabled; restart serve with an API token",
            }, status_code=503)
        body, error_response = await _read_write_body(request)
        if error_response is not None:
            return error_response
        if set(body) != {"confirm"} or body.get("confirm") != "restore-change-set":
            return JSONResponse({
                "error": "confirm must be 'restore-change-set'",
            }, status_code=400)
        runs = await _run_blocking(read_index_progress, fixed_vault_root)
        if (
            scan_operation["process"] is not None
            and scan_operation["process"].returncode is None
        ) or any(run.get("status") == "running" for run in runs):
            return JSONResponse({
                "error": "stop active indexing before restoring files",
            }, status_code=409)
        change_id = str(request.path_params.get("change_id") or "")
        try:
            result = await _run_blocking(
                ChangeJournal(fixed_vault_root).restore, change_id,
            )
        except ChangeConflictError as exc:
            return JSONResponse({
                "error": "restore refused because files changed afterwards",
                "conflicts": exc.paths,
            }, status_code=409)
        except LookupError:
            return JSONResponse({"error": "change set not found"}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except OSError as exc:
            logger.error("api_change_restore_failed", error=str(exc))
            return JSONResponse({
                "error": "files could not be restored",
            }, status_code=500)
        return JSONResponse({
            "ok": True,
            "change": result,
            "requires_reindex": True,
        })

    async def ask(request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "body must be a JSON object"}, status_code=400,
            )

        question = body.get("question", "")
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

        # Resolve the same category/tag/format scope used by /api/search.
        # ``set()`` deliberately survives as an empty scope: it must produce
        # an insufficient-evidence answer, never widen back to the vault.
        try:
            filter_note_ids = await _run_blocking(
                _resolve_note_filter, session, body,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        retrieval_scope = _retrieval_scope(filter_note_ids)
        if retrieval_scope["status"] == "no_matches":
            empty_result = {
                "answer": "No indexed documents match the selected search scope.",
                "sources": [],
                "source_details": [],
                "citation_selections": [],
                "answer_status": "insufficient_evidence",
                "retrieval_scope": retrieval_scope,
                "dropped_citations": 0,
            }
            if stream:
                async def empty_event_stream():
                    event = {
                        "type": "done",
                        **empty_result,
                        "model_execution": cognition_connectivity(session.config),
                    }
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"data: {payload}\n\n".encode("utf-8")

                return StreamingResponse(
                    empty_event_stream(), media_type="text/event-stream",
                )
            return JSONResponse({
                **empty_result,
                "model_execution": cognition_connectivity(session.config),
            })

        if stream:
            async def event_stream():
                # Oracle.ask_stream is a sync generator. Advance it in a
                # bounded worker so model I/O never blocks the ASGI event loop.
                gen = session.oracle.ask_stream(
                    question, top_k=top_k, history=history,
                    filter_note_ids=filter_note_ids,
                )
                async with aclosing(_iterate_blocking(gen)) as events:
                    async for event in events:
                        if event.get("type") == "done":
                            event = dict(event)
                            event["model_execution"] = cognition_connectivity(
                                session.config,
                            )
                            event["retrieval_scope"] = retrieval_scope
                        payload = json.dumps(event, ensure_ascii=False)
                        yield f"data: {payload}\n\n".encode("utf-8")

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = await _run_blocking(
            session.oracle.ask, question, top_k=top_k, history=history,
            filter_note_ids=filter_note_ids,
        )
        return JSONResponse({
            "answer": result.get("answer", ""),
            "sources": list(result.get("sources") or []),
            "source_details": list(result.get("source_details") or []),
            "citation_selections": result.get("citation_selections"),
            **({"sufficiency": result["sufficiency"]}
               if isinstance(result.get("sufficiency"), dict) else {}),
            "answer_status": result.get("answer_status") or "uncertain",
            "model_execution": cognition_connectivity(session.config),
            "retrieval_scope": retrieval_scope,
            "dropped_citations": int(result.get("dropped_citations") or 0),
            # Present only when the model gave no answer, so a timeout is not
            # mistaken for insufficient evidence. The streamed "done" event
            # carries the same key.
            **({"generation_error": result["generation_error"]}
               if result.get("generation_error") else {}),
        })

    def _retrieval_scope(filter_note_ids) -> dict[str, int | str | None]:
        if filter_note_ids is None:
            return {"status": "all", "matched_notes": None}
        matched = len(filter_note_ids)
        return {
            "status": "filtered" if matched else "no_matches",
            "matched_notes": matched,
        }

    def _resolve_note_filter(session, body):
        """Translate the optional filter keys into a note-id set.

        Raises ValueError on a malformed value so the caller can answer 400
        rather than silently ignoring a filter the client believes is applied
        -- returning unfiltered results to a request that asked for a filter
        is the dangerous failure here, not an error response.
        """
        def _str_list(key):
            raw = body.get(key)
            if raw is None:
                return None
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                raise ValueError(f"{key} must be a string or a list of strings")
            return raw

        category = body.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError("category must be a string")
        return session.db.resolve_note_filter(
            category=category,
            tags=_str_list("tags"),
            formats=_str_list("formats"),
        )

    async def search(request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "body must be a JSON object"}, status_code=400,
            )

        query = body.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return JSONResponse({"error": "query is required"}, status_code=400)
        try:
            top_k = SecurityGuard.coerce_top_k(body.get("top_k"), default=10)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        history = body.get("history") or None

        # Optional narrowing. Absent keys mean "search everything"; a filter
        # that matches nothing returns nothing rather than falling back to the
        # whole vault, which would answer from notes the caller excluded.
        try:
            filter_note_ids = await _run_blocking(
                _resolve_note_filter, session, body,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        retrieval_scope = _retrieval_scope(filter_note_ids)
        if filter_note_ids is not None and not filter_note_ids:
            return JSONResponse({
                "hits": [],
                "retrieval_scope": retrieval_scope,
            })

        def _search_rows():
            retrieval_query = session.oracle.rewrite_retrieval_query(
                query, history=history,
            )
            query_vector = session.embedder.embed(retrieval_query)
            use_hybrid = (
                getattr(session.config.cognition, "hybrid_search", True)
                and session.db.fts_available
            )
            if use_hybrid:
                hits = session.oracle.connector.find_hybrid(
                    query_text=retrieval_query,
                    query_vector=query_vector,
                    top_k=top_k,
                    filter_note_ids=filter_note_ids,
                )
            elif query_vector:
                hits = session.oracle.connector.find_similar_notes(
                    query_vector, top_k=top_k, filter_note_ids=filter_note_ids,
                )
            else:
                hits = []

            titles = session.db.get_note_titles(
                hit.get("note_id") for hit in hits
                if hit.get("note_id") is not None
            )
            rows = []
            for hit in hits:
                note_id = hit.get("note_id")
                title = titles.get(int(note_id), "")
                text = hit.get("text") or ""
                snippet = text[:200] + ("…" if len(text) > 200 else "")
                rows.append({
                    "note_id": int(note_id),
                    "title": title,
                    "snippet": snippet,
                    "score": float(hit.get("score") or 0.0),
                })
            return rows

        rows = await _run_blocking(_search_rows)
        return JSONResponse({
            "hits": rows,
            "retrieval_scope": retrieval_scope,
        })

    async def get_note(request: Request) -> Response:
        # When a token is set, _TokenAuthMiddleware has already required
        # it for non-loopback callers before we reach this handler.
        try:
            note_id = int(request.path_params["note_id"])
        except (ValueError, KeyError):
            return JSONResponse({"error": "note_id must be an integer"}, status_code=400)

        loc = await _run_blocking(session.db.get_note_location, note_id)
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
        sections = []
        extraction = None
        try:
            def _read_note():
                note = MarkdownParser().parse_file(
                    resolved,
                    vault_root=session.vault_root,
                    config=session.config,
                )
                anchors = []
                cursor = 0
                for section in note.sections:
                    start = note.content.find(section.text, cursor)
                    if start < 0:
                        continue
                    end = start + len(section.text)
                    anchors.append({
                        "order": int(section.order),
                        "page": section.page,
                        "heading": section.heading,
                        "start": start,
                        "end": end,
                    })
                    cursor = end
                return note.content, anchors, extraction_diagnostics(note)

            body_text, sections, extraction = await _run_blocking(_read_note)
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("api_note_read_failed", path=path, error=str(exc))
            extraction = {
                "status": "failed",
                "issues": [{"code": "document_read_failed", "page": None}],
            }
        timeline = await _run_blocking(session.db.get_note_timeline, note_id)
        if not isinstance(timeline, dict):
            timeline = {
                key: {"value": None, "source": None}
                for key in ("published", "modified", "indexed", "verified")
            }
        return JSONResponse({
            "note_id": note_id,
            "title": title,
            "path": path,
            "body": body_text,
            "sections": sections,
            "extraction": extraction,
            "timeline": timeline,
        })

    async def categories(request: Request) -> Response:
        rows = await _run_blocking(session.db.get_category_frequency)
        return JSONResponse({
            "categories": [
                {"category": cat, "count": int(n)} for cat, n in rows
            ],
        })

    async def filter_options(request: Request) -> Response:
        """Return the indexed facets the web UI can use for narrowing."""
        category_rows, tag_rows = await asyncio.gather(
            _run_blocking(session.db.get_category_frequency),
            _run_blocking(session.db.get_tag_frequency, limit=100),
        )
        formats = sorted({
            str(value).lower().lstrip(".")
            for value in session.config.vault.formats
            if str(value).strip()
        })
        return JSONResponse({
            "categories": [
                {"category": name, "count": int(count)}
                for name, count in category_rows
            ],
            "tags": [
                {"tag": name, "count": int(count)}
                for name, count in tag_rows
            ],
            "formats": formats,
        })

    async def review_inbox(request: Request) -> Response:
        """Aggregate existing maintenance signals without mutating them."""
        try:
            payload = await _run_blocking(build_review_inbox, session)
        except Exception as exc:
            logger.error("api_review_inbox_failed", error=str(exc))
            return JSONResponse({
                "error": "review inbox could not be generated",
            }, status_code=500)
        return JSONResponse(payload)

    async def preview_topic_synthesis(request: Request) -> Response:
        """Build a reviewable topic draft without writing it to the vault."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "body must be a JSON object"}, status_code=400,
            )

        allowed = {"kind", "value", "passages_per_note"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return JSONResponse({
                "error": f"unsupported synthesis fields: {', '.join(unknown)}",
            }, status_code=400)
        kind = body.get("kind")
        if kind not in {"tag", "category"}:
            return JSONResponse({
                "error": "kind must be tag or category",
            }, status_code=400)
        value = body.get("value")
        if not isinstance(value, str):
            return JSONResponse({"error": "value must be a string"}, status_code=400)
        value = value.strip()
        if (
            not value
            or len(value) > 200
            or any(ord(char) < 32 for char in value)
        ):
            return JSONResponse({"error": "value is invalid"}, status_code=400)
        passages_per_note = body.get("passages_per_note", 3)
        if (
            isinstance(passages_per_note, bool)
            or not isinstance(passages_per_note, int)
            or not 1 <= passages_per_note <= 10
        ):
            return JSONResponse({
                "error": "passages_per_note must be an integer from 1 to 10",
            }, status_code=400)

        options = {
            kind: value,
            "passages_per_note": passages_per_note,
            "dry_run": True,
        }
        try:
            report = await _run_blocking(
                Synthesizer(session).distill, **options,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("api_synthesis_preview_failed", error=str(exc))
            return JSONResponse({
                "error": "synthesis preview could not be generated",
            }, status_code=500)
        return JSONResponse(report.as_dict())

    async def decide_review_candidate(request: Request) -> Response:
        """Persist one bounded human decision without changing source files."""
        if not write_enabled:
            return JSONResponse({
                "error": "write operations are disabled; restart serve with an API token",
            }, status_code=503)
        candidate_id = str(request.path_params.get("candidate_id") or "")
        if (
            not candidate_id
            or len(candidate_id) > 200
            or any(ord(char) < 32 for char in candidate_id)
        ):
            return JSONResponse({"error": "candidate_id is invalid"}, status_code=400)

        body, error_response = await _read_write_body(request)
        if error_response is not None:
            return error_response
        allowed = {"decision", "postpone_days"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return JSONResponse({
                "error": f"unsupported review fields: {', '.join(unknown)}",
            }, status_code=400)
        decision = body.get("decision")
        if decision not in {"confirm", "dismiss", "postpone"}:
            return JSONResponse({
                "error": "decision must be confirm, dismiss, or postpone",
            }, status_code=400)
        postpone_days = body.get("postpone_days")
        if decision == "postpone":
            if (
                isinstance(postpone_days, bool)
                or not isinstance(postpone_days, int)
                or not 1 <= postpone_days <= 3650
            ):
                return JSONResponse({
                    "error": "postpone_days must be an integer from 1 to 3650",
                }, status_code=400)
        elif postpone_days is not None:
            return JSONResponse({
                "error": "postpone_days is only valid when postponing",
            }, status_code=400)

        try:
            payload = await _run_blocking(
                build_review_inbox, session, include_inactive=True,
            )
        except Exception as exc:
            logger.error("api_review_decision_lookup_failed", error=str(exc))
            return JSONResponse({
                "error": "review candidate could not be validated",
            }, status_code=500)
        candidate = next(
            (item for item in payload["items"] if item["id"] == candidate_id),
            None,
        )
        if candidate is None:
            return JSONResponse({"error": "review candidate not found"}, status_code=404)

        now = datetime.now(timezone.utc)
        decided_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        postpone_until = None
        if decision == "postpone":
            postpone_until = (now + timedelta(days=postpone_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            )
        snapshot = {
            key: candidate.get(key)
            for key in (
                "id", "kind", "priority", "documents", "reason", "coverage",
                "metadata",
            )
        }
        try:
            event = await _run_blocking(
                session.db.record_review_decision,
                candidate_id=candidate_id,
                kind=candidate["kind"],
                decision=decision,
                decided_at=decided_at,
                postpone_until=postpone_until,
                snapshot=snapshot,
                source="web",
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("api_review_decision_failed", error=str(exc))
            return JSONResponse({
                "error": "review decision could not be saved",
            }, status_code=500)
        return JSONResponse({"ok": True, "event": event})

    async def update_note_classification(request: Request) -> Response:
        """Persist a human category/tag override for one indexed note."""
        if not write_enabled:
            return JSONResponse({
                "error": "write operations are disabled; restart serve with an API token",
            }, status_code=503)
        body, error_response = await _read_write_body(request)
        if error_response is not None:
            return error_response
        allowed = {"category", "tags"}
        if set(body) != allowed:
            return JSONResponse({
                "error": "classification requires exactly category and tags",
            }, status_code=400)

        category = body.get("category")
        if category is not None and not isinstance(category, str):
            return JSONResponse({"error": "category must be a string or null"}, status_code=400)
        if isinstance(category, str):
            category = category.strip().strip("/") or None
            if (
                category is not None
                and (
                    len(category) > 200
                    or any(ord(char) < 32 for char in category)
                    or any(not part.strip() for part in category.split("/"))
                )
            ):
                return JSONResponse({"error": "category is invalid"}, status_code=400)

        raw_tags = body.get("tags")
        if not isinstance(raw_tags, list) or len(raw_tags) > 50:
            return JSONResponse({"error": "tags must be a list of at most 50 strings"}, status_code=400)
        tags: list[str] = []
        seen_tags: set[str] = set()
        for raw_tag in raw_tags:
            if not isinstance(raw_tag, str):
                return JSONResponse({"error": "every tag must be a string"}, status_code=400)
            tag = raw_tag.strip()
            if not tag:
                continue
            if len(tag) > 100 or any(ord(char) < 32 for char in tag):
                return JSONResponse({"error": "tag is invalid"}, status_code=400)
            folded = tag.casefold()
            if folded not in seen_tags:
                seen_tags.add(folded)
                tags.append(tag)

        note_id = int(request.path_params["note_id"])
        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            event = await _run_blocking(
                session.db.set_classification_override,
                note_id=note_id,
                category=category,
                tags=tags,
                updated_at=updated_at,
                source="web",
            )
        except LookupError:
            return JSONResponse({"error": "note not found"}, status_code=404)
        except Exception as exc:
            logger.error("api_classification_override_failed", error=str(exc))
            return JSONResponse({
                "error": "classification override could not be saved",
            }, status_code=500)
        return JSONResponse({"ok": True, "event": event})

    async def index(request: Request) -> Response:
        # Serve the SPA shell. Template is a flat HTML file (no Jinja
        # variables) so we just stream the bytes.
        if not _INDEX_HTML.exists():
            return PlainTextResponse(
                "Grimore web UI assets missing. Reinstall the serve extra.",
                status_code=500,
            )
        try:
            body = await _run_blocking(_INDEX_HTML.read_bytes)
        except OSError:
            return PlainTextResponse("Grimore web UI asset unreadable.", status_code=500)
        return Response(body, media_type="text/html; charset=utf-8")

    async def static_asset(request: Request) -> Response:
        asset = request.path_params.get("asset", "")
        media_types = {
            "app.js": "text/javascript; charset=utf-8",
            "grimore.css": "text/css; charset=utf-8",
            "fonts/NotoSerif-Regular.ttf": "font/ttf",
            "fonts/NotoSerif-Bold.ttf": "font/ttf",
            "fonts/NotoSerif-Italic.ttf": "font/ttf",
            "fonts/NotoSansMono-Regular.ttf": "font/ttf",
        }
        media_type = media_types.get(asset)
        if media_type is None:
            return PlainTextResponse("not found", status_code=404)
        try:
            body = await _run_blocking((_STATIC_DIR / asset).read_bytes)
        except OSError:
            return PlainTextResponse("not found", status_code=404)
        return Response(body, media_type=media_type)

    routes = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/setup", setup_summary, methods=["GET"]),
        Route("/api/setup/apply", apply_setup, methods=["POST"]),
        Route("/api/indexing", indexing_status, methods=["GET"]),
        Route("/api/indexing/start", start_indexing, methods=["POST"]),
        Route("/api/changes", change_history, methods=["GET"]),
        Route(
            "/api/changes/{change_id:str}/restore",
            restore_change_set,
            methods=["POST"],
        ),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/notes/{note_id:int}", get_note, methods=["GET"]),
        Route(
            "/api/notes/{note_id:int}/classification",
            update_note_classification,
            methods=["POST"],
        ),
        Route("/api/categories", categories, methods=["GET"]),
        Route("/api/filter-options", filter_options, methods=["GET"]),
        Route(
            "/api/syntheses/preview",
            preview_topic_synthesis,
            methods=["POST"],
        ),
        Route("/api/review-inbox", review_inbox, methods=["GET"]),
        Route(
            "/api/review-inbox/{candidate_id:str}/decision",
            decide_review_candidate,
            methods=["POST"],
        ),
        Route("/", index, methods=["GET"]),
    ]
    if _STATIC_DIR.exists():
        routes.append(Route("/static/{asset:path}", static_asset, methods=["GET"]))
    return routes


def build_app(
    session: Session,
    *,
    api_token: Optional[str] = None,
    cors_origin: Optional[str] = None,
    strict_token: bool = False,
    config_path: str | Path = "grimore.toml",
    scan_launcher=None,
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
        middleware.append(Middleware(
            _WriteTokenAuthMiddleware,
            api_token=api_token,
            exempt_loopback=False,
        ))

    return Starlette(
        debug=False,
        routes=_build_routes(
            session,
            write_enabled=bool(api_token),
            config_path=config_path,
            scan_launcher=scan_launcher,
        ),
        middleware=middleware,
    )
