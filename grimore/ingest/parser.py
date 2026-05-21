"""
Document parsing for Project Grimore.

Originally Markdown-only. As of v2.1 this module is a thin dispatcher in
front of the per-format adapters in :mod:`grimore.ingest.adapters`: it
picks the right :class:`DocumentExtractor` for the file's extension,
calls it, and wraps the result in the historical :class:`ParsedNote`
struct so every downstream consumer (cli.scan, daemon.process_file,
shell mentions) keeps working unchanged.

The :class:`MarkdownParser` name is preserved for the same reason — the
rest of the codebase still imports it. Internally it has nothing
Markdown-specific in it any more; it's the dispatcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from grimore.ingest.adapters import (
    AdapterOptions,
    ExtractedDocument,
    ExtractedSection,
    for_path,
)
from grimore.ingest.adapters.markdown import MarkdownAdapter, _MAX_MD_BYTES
from grimore.utils.config import is_ignored_path
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

# Historical public constant. Kept as a re-export of the Markdown adapter's
# limit so v2.0 imports (`from grimore.ingest.parser import MAX_NOTE_BYTES`)
# keep resolving. Per-format caps for non-MD documents are config-driven in
# later phases.
MAX_NOTE_BYTES = _MAX_MD_BYTES


def iter_vault_documents(
    vault_path: Path,
    formats: Iterable[str],
    ignored_dirs: list[str],
    sidecar_dir: Optional[str] = None,
) -> list[Path]:
    """Sorted list of every document the dispatcher can pick up.

    Iterates one ``rglob`` per extension because ``Path.glob`` doesn't
    understand brace alternation (``*.{md,pdf}``). Skips ignored dirs and,
    when ``sidecar_dir`` is set, skips any path under it so we never
    re-ingest the .md files Grimore itself generated.

    The result is sorted so progress UI feels stable across reruns.
    """
    vault_root = vault_path.resolve()
    sidecar_root: Optional[Path] = None
    if sidecar_dir:
        try:
            sidecar_root = (vault_root / sidecar_dir).resolve()
        except OSError:
            sidecar_root = None

    seen: set[Path] = set()
    for ext in formats:
        key = ext.lower().lstrip(".")
        if not key:
            continue
        for candidate in vault_path.rglob(f"*.{key}"):
            if is_ignored_path(candidate, ignored_dirs):
                continue
            try:
                resolved = SecurityGuard.resolve_within_vault(candidate, vault_root)
            except ValueError:
                logger.warning("path_escape_skipped", path=str(candidate))
                continue
            if sidecar_root is not None:
                try:
                    resolved.relative_to(sidecar_root)
                except ValueError:
                    pass
                else:
                    # Inside the sidecar tree — never re-ingest our own output.
                    continue
            seen.add(candidate)
    return sorted(seen)


@dataclass
class ParsedNote:
    """A fully parsed document with its metadata and idempotency keys.

    Pre-v2.1 fields (``path``, ``title``, ``metadata``, ``content``,
    ``content_hash``) are kept positional so existing tests that build
    ``ParsedNote(...)`` literals continue to work. The multi-format
    additions (``format``, ``file_hash``, ``sections``, ``size_bytes``)
    are keyword-only with sensible defaults.
    """
    path: Path
    title: str
    metadata: dict[str, Any]
    content: str
    content_hash: str
    format: str = "md"
    file_hash: str = ""
    sections: list[ExtractedSection] = field(default_factory=list)
    size_bytes: int = 0

    @classmethod
    def from_extracted(cls, doc: ExtractedDocument) -> "ParsedNote":
        """Adapt the format-neutral payload into the historical struct."""
        return cls(
            path=doc.source_path,
            title=doc.title,
            metadata=dict(doc.metadata),
            content=doc.text,
            content_hash=doc.content_hash,
            format=doc.format,
            file_hash=doc.file_hash,
            sections=list(doc.sections),
            size_bytes=doc.size_bytes,
        )


class MarkdownParser:
    """Dispatcher over the per-format adapter registry.

    Name and public signature are preserved from the v2.0 Markdown-only
    implementation so every existing call site (``MarkdownParser().parse_file(...)``)
    keeps working. When no adapter is registered for an extension the
    parser falls back to :class:`MarkdownAdapter` — that matches the
    pre-v2.1 behaviour where everything was assumed to be Markdown.

    Vault-scope contract
    --------------------
    Callers MUST ensure ``file_path`` resolves inside the vault before
    calling :py:meth:`parse_file`. The two in-tree consumers (``cli.scan``
    and ``daemon.process_file``) do this via ``SecurityGuard.resolve_within_vault``.
    For defence-in-depth, callers can also pass ``vault_root`` to
    :py:meth:`parse_file` and the parser will re-validate internally —
    use this whenever the parser is invoked from a new code path so a
    forgotten outer check cannot turn into a path-traversal bug.
    """

    # Kept as a class attribute so callers / tests that monkeypatch the
    # fallback (rare, but used by the new adapter tests) can do so cleanly.
    _fallback_adapter = MarkdownAdapter()

    def parse_file(
        self,
        file_path: Path,
        *,
        vault_root: Optional[Union[str, Path]] = None,
    ) -> ParsedNote:
        """Parse ``file_path`` into a :class:`ParsedNote`.

        Looks up the adapter for the file's extension; falls back to the
        Markdown adapter when none is registered. Raises ``ValueError``
        on oversize / unreadable / out-of-vault files (the adapter does
        the actual checks).
        """
        path = Path(file_path)

        if vault_root is not None:
            # Defence-in-depth even if the caller already filtered.
            SecurityGuard.resolve_within_vault(path, vault_root)

        adapter = for_path(path) or self._fallback_adapter
        options = AdapterOptions(
            vault_root=Path(vault_root) if vault_root is not None else None,
        )
        doc = adapter.extract(path, options=options)
        return ParsedNote.from_extracted(doc)
