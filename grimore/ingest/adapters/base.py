"""
Format-neutral interface for document adapters.

Every adapter (Markdown, PDF, ePub, DOCX, …) returns an
:class:`ExtractedDocument` so the cognition layer — tagger, embedder,
oracle, mirror, chronicler, synthesizer — can stay completely ignorant
of the original file format. See ``docs/MULTIFORMAT_BLUEPRINT.md`` §3-§4
for the architectural picture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol, Union, runtime_checkable


@dataclass(frozen=True)
class ExtractedSection:
    """One natural unit of a document.

    For paginated formats (PDF) ``page`` carries the 1-based page number;
    for structured formats (DOCX, HTML, ePub) ``heading`` carries the
    nearest enclosing heading. ``order`` is the stable index used to seed
    chunk numbering when the adapter has already done structural work.
    """
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None
    order: int = 0


@dataclass(frozen=True)
class ExtractedDocument:
    """The format-neutral payload an adapter hands to the rest of Grimore.

    ``content_hash`` hashes the normalised body text (same algorithm the
    Markdown pipeline has always used) so re-tagging fires only when the
    *meaning* changes, not when bytes change. ``file_hash`` is the cheap
    "did anything on disk change at all" key used by the two-tier
    fast-skip (blueprint §6.4).
    """
    source_path: Path
    format: str
    title: str
    text: str
    content_hash: str
    file_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[ExtractedSection] = field(default_factory=list)
    size_bytes: int = 0


@dataclass(frozen=True)
class AdapterOptions:
    """Per-call adapter knobs.

    Kept tiny on purpose: anything format-specific (PDF engine choice,
    OCR toggle, max-bytes cap) is read by the adapter from the global
    config — these options only carry information the *caller* of the
    parse provides on a per-file basis.
    """
    vault_root: Optional[Path] = None


@runtime_checkable
class DocumentExtractor(Protocol):
    """Protocol every adapter implements.

    Class attributes declare the adapter's contract; the dispatcher uses
    them to decide whether a file can be ingested and whether its source
    can be written back to (Markdown only — every other format gets a
    sidecar).
    """
    # Lowercase extensions, no leading dot. Registered on import.
    extensions: ClassVar[tuple[str, ...]]
    # True when the source bytes are binary (PDF, EPUB, DOCX, ODT). The
    # output layer uses this to decide whether to write tags inline or to
    # a sidecar.
    binary: ClassVar[bool]
    # True when Grimore is allowed to rewrite frontmatter into the source
    # file. Only Markdown sets this True.
    mutable_frontmatter: ClassVar[bool]

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        """Return the format-neutral payload for ``path``.

        Adapters MUST raise :class:`ValueError` (not crash) when the file
        is unreadable, oversize, encrypted, or otherwise unprocessable —
        the scan/daemon loops catch ValueError and surface a structured
        log line rather than tearing down the whole batch.
        """
        ...
