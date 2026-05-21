"""
Extension-keyed registry of document adapters.

Adapters call :func:`register` (typically as a class decorator) on import
and the dispatcher in :mod:`grimore.ingest.parser` looks them up via
:func:`for_path`. Keeping the registry in its own module avoids an import
cycle between the adapter package and the parser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TypeVar

from grimore.ingest.adapters.base import DocumentExtractor

_T = TypeVar("_T", bound=DocumentExtractor)

# Lowercase extensions (no leading dot) → adapter instance.
_REGISTRY: dict[str, DocumentExtractor] = {}


def register(extractor: _T) -> _T:
    """Register an adapter instance against every extension it declares.

    Idempotent: re-registering the same adapter (common in tests that
    import the module twice) is a no-op. A different adapter trying to
    claim an already-claimed extension raises — the explicit failure beats
    silent shadowing.
    """
    for ext in extractor.extensions:
        key = ext.lower().lstrip(".")
        existing = _REGISTRY.get(key)
        if existing is None:
            _REGISTRY[key] = extractor
        elif existing is not extractor and type(existing) is not type(extractor):
            raise ValueError(
                f"extension {key!r} already registered to {type(existing).__name__}; "
                f"refusing to overwrite with {type(extractor).__name__}"
            )
    return extractor


def for_path(path: Path | str) -> Optional[DocumentExtractor]:
    """Return the adapter for ``path`` based on its extension, or None."""
    key = Path(path).suffix.lower().lstrip(".")
    return _REGISTRY.get(key)


def supported_extensions() -> frozenset[str]:
    """Snapshot of every extension a currently-loaded adapter handles."""
    return frozenset(_REGISTRY)


def _reset_for_tests() -> None:
    """Clear the registry. Test-only — never call from production code."""
    _REGISTRY.clear()
