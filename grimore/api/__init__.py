"""HTTP API package.

Built on top of Starlette rather than FastAPI: Starlette is the routing
primitive FastAPI itself uses, and skipping the pydantic-core layer
keeps the ``serve`` extra installable on Termux/ARM where pydantic-core
has no prebuilt wheel. Users who want pydantic validation can still
install FastAPI on top — it shares the same ASGI app shape.

Public entry point: :func:`grimore.api.app.build_app`.
"""
