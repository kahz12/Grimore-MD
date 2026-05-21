"""
Magic-byte content sniffing for misnamed or extension-less files.

Off by default — gated by ``config.ingest.sniff_magic``. When the user
opts in, the parser dispatcher calls :func:`sniff_extension` for files
whose extension is not in ``config.vault.formats`` (or absent entirely),
and routes the file to whichever adapter matches the detected content
type. Sniffing NEVER widens the format set beyond what the registry can
already handle — a sniffed mime that maps to no registered adapter is
treated as "skip".

The implementation depends on ``python-magic`` (which wraps libmagic),
declared as the ``sniff`` optional extra. Without it, every call here
returns ``None`` so the pipeline degrades silently and the rest of the
codebase keeps working unchanged. Preflight surfaces the missing extra
when the flag is on.

Why not write our own header sniffer:

* libmagic has decades of curated signatures; we'd reinvent it badly.
* The dependency is tiny and the binding is pure-Python on the Python
  side. The libmagic shared library is preinstalled on every Linux
  distro Grimore supports, on macOS via Homebrew, on Termux via
  ``pkg install file``, and on Windows via ``python-magic-bin``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import zipfile

from grimore.ingest.adapters import for_path
from grimore.utils.logger import get_logger

logger = get_logger(__name__)


# Mime → canonical extension. Only types whose adapter we ship are listed
# — anything else stays a "skip" and the caller falls through to the
# normal extension-based dispatch (which will then no-op for unsupported
# files). Adding a new adapter automatically widens what the sniffer can
# claim because the mapping ends with a registry lookup.
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "application/epub+zip": "epub",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/msword": "doc",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/plain": "txt",
}


_SENTINEL = object()


def _load_magic():
    """Return a ``magic.Magic(mime=True)`` instance or ``None`` if the
    ``python-magic`` extra is not installed.

    Cached on the function object so repeated sniffs in a single scan
    don't reload the shared library. Any import or initialisation error
    is swallowed and logged once — sniffing is opt-in *and* best-effort;
    a broken libmagic must never block ingestion of correctly-named
    files.
    """
    cached = getattr(_load_magic, "_cached", _SENTINEL)
    if cached is not _SENTINEL:
        return cached
    try:
        import magic  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.debug("sniffer_import_failed", error=str(exc))
        _load_magic._cached = None  # type: ignore[attr-defined]
        return None
    try:
        inst = magic.Magic(mime=True)
    except Exception as exc:  # pragma: no cover - libmagic init quirks
        logger.warning("sniffer_init_failed", error=str(exc))
        _load_magic._cached = None  # type: ignore[attr-defined]
        return None
    _load_magic._cached = inst  # type: ignore[attr-defined]
    return inst


def _disambiguate_zip(path: Path) -> Optional[str]:
    """When libmagic returns plain ``application/zip``, peek inside the
    archive to tell DOCX / ODT / EPUB apart. Returns the canonical
    extension or ``None`` if no marker matches.

    OOXML, ODT and EPUB are all ZIP containers; libmagic does not always
    descend into them. We look for their well-known sentinel entries:

    * ``[Content_Types].xml`` + ``word/`` → docx
    * ``mimetype`` whose body starts with the OPF mime → odt / epub
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return "docx"
            if "mimetype" in names:
                try:
                    mime = zf.read("mimetype", pwd=None).decode("ascii", "replace").strip()
                except Exception:
                    mime = ""
                if mime == "application/vnd.oasis.opendocument.text":
                    return "odt"
                if mime == "application/epub+zip":
                    return "epub"
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def sniff_extension(path: Path | str) -> Optional[str]:
    """Return the canonical extension for ``path`` based on its content,
    or ``None`` when the file is unreadable, the sniff library is
    missing, or the detected mime maps to no adapter we ship.

    Callers should treat ``None`` as "skip" — never as "fall back to
    Markdown", since a misnamed binary read as Markdown would corrupt
    the index with raw bytes.
    """
    p = Path(path)
    if not p.is_file():
        return None
    magic = _load_magic()
    if magic is None:
        return None
    try:
        mime = magic.from_file(str(p)) or ""
    except Exception as exc:
        logger.debug("sniffer_read_failed", path=str(p), error=str(exc))
        return None
    mime = mime.split(";", 1)[0].strip().lower()
    if mime == "application/zip":
        return _disambiguate_zip(p)
    ext = _MIME_TO_EXT.get(mime)
    if ext is None:
        return None
    return ext


def adapter_for_sniffed(path: Path | str):
    """High-level helper: sniff + look up the registered adapter.

    Returns ``(extension, adapter)`` on success, ``(None, None)`` when
    the file is unsupported by any registered adapter. Keeps the parser
    dispatcher branchless: one call, two checks against ``None``.
    """
    ext = sniff_extension(path)
    if ext is None:
        return None, None
    # Build a fake path with the canonical suffix so ``for_path`` keys
    # off it. We don't move or rename the actual file.
    probe = Path(path).with_suffix(f".{ext}")
    adapter = for_path(probe)
    if adapter is None:
        return None, None
    return ext, adapter


def sniff_available() -> bool:
    """True when the ``sniff`` optional extra (``python-magic``) is
    importable and libmagic initialises. Used by preflight to surface
    an actionable hint when the user has enabled ``sniff_magic`` without
    installing the extra.
    """
    return _load_magic() is not None
