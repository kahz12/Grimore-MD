"""
Document adapters for Grimore's multi-format ingest pipeline.

Each adapter knows how to turn one family of file types (Markdown, PDF,
ePub, DOCX, …) into an :class:`ExtractedDocument` — the format-neutral
struct the rest of the pipeline consumes. Adapters are registered against
their file extensions in :mod:`grimore.ingest.adapters.registry` and
looked up by :func:`for_path`.

Importing this package auto-registers every built-in adapter. Phase 1
ships only the Markdown adapter; later phases add PDF, ePub, DOCX, etc.
"""
from grimore.ingest.adapters.base import (
    AdapterOptions,
    DocumentExtractor,
    ExtractedDocument,
    ExtractedSection,
)
from grimore.ingest.adapters.registry import (
    for_path,
    register,
    supported_extensions,
)

# Side-effect imports: each adapter module self-registers on import.
from grimore.ingest.adapters import markdown as _markdown  # noqa: F401
from grimore.ingest.adapters import txt as _txt              # noqa: F401
from grimore.ingest.adapters import html as _html            # noqa: F401
from grimore.ingest.adapters import docx as _docx            # noqa: F401
from grimore.ingest.adapters import pdf as _pdf              # noqa: F401
from grimore.ingest.adapters import epub as _epub            # noqa: F401

__all__ = [
    "AdapterOptions",
    "DocumentExtractor",
    "ExtractedDocument",
    "ExtractedSection",
    "for_path",
    "register",
    "supported_extensions",
]
