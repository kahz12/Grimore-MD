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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from grimore.ingest.adapters import (
    AdapterOptions,
    ExtractedDocument,
    ExtractionIssue,
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
# as adapters are registered.
MAX_NOTE_BYTES = _MAX_MD_BYTES


def _sniff_enabled(config) -> bool:
    """Whether the parser should consult the magic-byte sniffer for files
    whose extension misses the registry. The flag lives under
    ``[ingest].sniff_magic`` and is off by default — sniffing requires
    the optional ``sniff`` extra and we don't want to silently widen
    ingestion behaviour for users who haven't opted in.
    """
    if config is None:
        return False
    ingest = getattr(config, "ingest", None)
    return bool(getattr(ingest, "sniff_magic", False))


def iter_vault_documents(
    vault_path: Path,
    formats: Iterable[str],
    ignored_dirs: list[str],
    sidecar_dir: Optional[str] = None,
    *,
    sniff_magic: bool = False,
) -> list[Path]:
    """Sorted list of every document the dispatcher can pick up.

    Walks the vault once and compares lower-cased suffixes against a set.
    This keeps extension matching case-insensitive on Linux and avoids one
    full directory traversal per configured format. Skips ignored dirs and,
    when ``sidecar_dir`` is set, skips any path under it so we never re-ingest
    the .md files Grimore itself generated.

    When ``sniff_magic`` is True, files with unknown suffixes are content-
    sniffed during that same pass (e.g. a PDF saved without an extension).

    The result is sorted so progress UI feels stable across reruns.
    """
    vault_root = vault_path.resolve()
    sidecar_root: Optional[Path] = None
    if sidecar_dir:
        try:
            sidecar_root = (vault_root / sidecar_dir).resolve()
        except OSError:
            sidecar_root = None

    known_exts = {ext.lower().lstrip(".") for ext in formats if ext}
    seen: set[Path] = set()
    sniff_extension = None
    if sniff_magic:
        from grimore.ingest.sniffer import sniff_extension as _sniff_extension
        sniff_extension = _sniff_extension

    for candidate in vault_path.rglob("*"):
        if not candidate.is_file():
            continue
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
                continue

        ext_key = candidate.suffix.lower().lstrip(".")
        if ext_key in known_exts:
            seen.add(candidate)
            continue
        if sniff_extension is not None:
            sniffed = sniff_extension(candidate)
            if sniffed and sniffed in known_exts:
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
    issues: list[ExtractionIssue] = field(default_factory=list)

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
            issues=list(doc.issues),
            size_bytes=doc.size_bytes,
        )


def extraction_diagnostics(note: ParsedNote) -> dict | None:
    """Public, bounded account of extraction coverage for supported formats."""
    issues = list(getattr(note, "issues", []) or [])
    if getattr(note, "format", None) != "pdf" and not issues:
        return None
    return {
        "status": (
            "partial" if any(not issue.recovered for issue in issues)
            else "recovered" if issues else "complete"
        ),
        "issues": [issue.as_dict() for issue in issues],
    }


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
        config=None,
    ) -> ParsedNote:
        """Parse ``file_path`` into a :class:`ParsedNote`.

        Looks up the adapter for the file's extension; falls back to the
        Markdown adapter when none is registered. Raises ``ValueError``
        on oversize / unreadable / out-of-vault files (the adapter does
        the actual checks).

        ``config`` is optional; when present, ingest-section overrides
        (e.g. ``pdf_engine``, ``ocr``) flow through ``AdapterOptions``
        so adapters that care can react without consulting global state.
        """
        path = Path(file_path)

        if vault_root is not None:
            # Defence-in-depth even if the caller already filtered.
            SecurityGuard.resolve_within_vault(path, vault_root)

        adapter = for_path(path)
        sniffed_ext: Optional[str] = None
        if adapter is None and _sniff_enabled(config):
            from grimore.ingest.sniffer import adapter_for_sniffed
            sniffed_ext, adapter = adapter_for_sniffed(path)
            if adapter is not None:
                logger.info(
                    "sniffer_matched",
                    path=str(path),
                    extension=sniffed_ext,
                )
        if adapter is None:
            adapter = self._fallback_adapter
        resolved_root = Path(vault_root) if vault_root is not None else None
        if config is not None:
            options = AdapterOptions.from_config(config, vault_root=resolved_root)
        else:
            options = AdapterOptions(vault_root=resolved_root)
        doc = adapter.extract(path, options=options)
        if sniffed_ext is not None and doc.format != sniffed_ext:
            # The adapter trusts its own ``format`` (often derived from
            # the extension on disk). For sniffed files we surface the
            # true content type so downstream chunking + DB rows match.
            doc = replace(doc, format=sniffed_ext)
        return ParsedNote.from_extracted(doc)


def read_document_text(
    file_path: Union[str, Path],
    *,
    vault_root: Optional[Union[str, Path]] = None,
    config=None,
) -> str:
    """Return adapter-extracted text for any supported vault document.

    Downstream features must use this instead of ``Path.read_text`` so PDF,
    DOCX, EPUB, ODT and other binary formats share the exact ingest semantics.
    """
    return MarkdownParser().parse_file(
        Path(file_path), vault_root=vault_root, config=config,
    ).content
