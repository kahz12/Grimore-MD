"""Tiny synchronous facade over HTTPX's direct ASGI transport.

Starlette's TestClient is coupled to AnyIO's blocking portal and can lag a
new Python release. These tests only need request/response semantics, so a
fresh event loop per call is smaller, deterministic, and exercises the ASGI
application directly without opening sockets.
"""
from __future__ import annotations

import asyncio

import httpx


class ASGITestClient:
    __test__ = False

    def __init__(
        self,
        app,
        *,
        client: tuple[str, int] = ("127.0.0.1", 50000),
    ) -> None:
        self.app = app
        self.peer = client

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app, client=self.peer)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_request())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def options(self, url: str, **kwargs) -> httpx.Response:
        return self.request("OPTIONS", url, **kwargs)
