"""
Hardened XML parsing for the zip-container adapters (docx, odt, epub).

Closes audit finding **M1** (XML entity-expansion DoS / "billion laughs").
The stdlib ``xml.etree.ElementTree`` expands *internal* entities, so a
tiny crafted XML member inside a ``.docx`` / ``.odt`` / ``.epub`` can blow
up to gigabytes of in-memory strings at parse time and OOM-kill the
process. The per-format byte caps are no defense — the bomb is a few
hundred bytes on disk; the blowup happens during parsing.

Defense, in order:

1. **defusedxml** (a pure-Python, Termux-safe wheel) is the authoritative
   parser when installed. With ``forbid_dtd=True`` it refuses any DTD,
   which kills both the internal-entity bomb and external-entity XXE,
   while still distinguishing a genuine declaration from the literal
   bytes appearing inside text/CDATA (no false positives).
2. If defusedxml is somehow absent (a broken install), we fall back to the
   stdlib parser but **pre-reject** any ``<!DOCTYPE`` / ``<!ENTITY``
   marker in the raw bytes *before* the parser can expand it. This keeps
   the adapters safe even without the dependency.

Malformed (but non-malicious) XML still surfaces as
``xml.etree.ElementTree.ParseError`` — the exact type the adapters already
catch — so existing error handling is unchanged.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import IO, Union

from grimore.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import defusedxml.ElementTree as _DET
    from defusedxml.common import DefusedXmlException

    _HAVE_DEFUSED = True
except ImportError:  # pragma: no cover - defusedxml is a declared dependency
    _DET = None
    DefusedXmlException = ()  # type: ignore[assignment]
    _HAVE_DEFUSED = False

# Fallback-path marker scan. A DOCTYPE/ENTITY declaration is the only way
# to introduce the entity-expansion bomb; in well-formed XML these bytes
# never appear in element text (they'd be escaped), so scanning the raw
# bytes is a safe pre-reject when defusedxml isn't available to parse
# things properly. Case-insensitive to reject aggressively.
_DTD_MARKER_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

# Ceiling on the *decompressed* size of a single archive member. The
# adapters cap the archive on disk, but that caps the compressed bytes:
# deflate reaches ~1000:1, so a few-MB member inside a size-legal
# .docx/.odt/.epub could still inflate to gigabytes in memory before any
# parser sees it (zip-bomb DoS). 100 MB comfortably clears any real
# document.xml / content.xml / chapter while bounding the blowup.
MAX_MEMBER_BYTES = 100_000_000


class UnsafeXmlError(ValueError):
    """Raised on a construct we refuse to process: a DTD/entity
    declaration, or a member that inflates past :data:`MAX_MEMBER_BYTES`.

    Subclasses ``ValueError`` so it flows through the adapters' existing
    "raise ValueError on unprocessable input → scan loop logs a skip"
    contract without any special handling.
    """


def read_bounded(
    source: IO[bytes],
    *,
    max_bytes: int | None = None,
    what: str = "member",
) -> bytes:
    """Read a (possibly compressed) stream, refusing to inflate past the cap.

    The check runs on the bytes as they decompress — never on the zip
    header's declared ``file_size``, which an attacker controls. Raises
    :class:`UnsafeXmlError` when the cap is exceeded.
    """
    if max_bytes is None:
        max_bytes = MAX_MEMBER_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = source.read(1 << 20)
        if not chunk:
            break
        if isinstance(chunk, str):  # tolerate a text-mode file object
            chunk = chunk.encode("utf-8", errors="replace")
        total += len(chunk)
        if total > max_bytes:
            logger.warning(
                "zip_member_too_large", what=what, max=max_bytes,
            )
            raise UnsafeXmlError(
                f"{what}: decompressed content exceeds {max_bytes} bytes "
                "(zip-bomb protection)"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def safe_parse_xml(
    source: Union[IO[bytes], bytes, bytearray],
    *,
    what: str = "xml",
) -> ET.Element:
    """Parse XML from an untrusted document and return its root element.

    ``source`` may be a binary file object (e.g. ``ZipFile.open(...)``) or
    raw bytes. Refuses entity/DTD constructs (the billion-laughs vector),
    external-entity references, and file objects that decompress past
    :data:`MAX_MEMBER_BYTES` (zip bombs); raises :class:`UnsafeXmlError`
    (a ``ValueError``) on a rejected construct and
    ``xml.etree.ElementTree.ParseError`` on merely malformed XML.

    ``what`` is a short label (usually the archive member name) used only
    for logging.
    """
    if isinstance(source, (bytes, bytearray)):
        # Already materialised in memory — nothing left to bound.
        data: bytes = bytes(source)
    else:
        data = read_bounded(source, what=what)

    if _HAVE_DEFUSED:
        try:
            # forbid_dtd kills the bomb at its root; forbid_entities /
            # forbid_external default True, blocking XXE for good measure.
            return _DET.fromstring(data, forbid_dtd=True)
        except DefusedXmlException as e:
            logger.warning(
                "xml_unsafe_construct_rejected", what=what, reason=type(e).__name__,
            )
            raise UnsafeXmlError(
                f"{what}: refused unsafe XML ({type(e).__name__}) — "
                "entity-expansion / XXE protection"
            ) from e

    # Degraded path: defusedxml missing. Pre-reject any DTD/entity marker
    # in the raw bytes before the stdlib parser can expand it.
    if _DTD_MARKER_RE.search(data):
        logger.warning("xml_unsafe_construct_rejected", what=what, reason="dtd_marker")
        raise UnsafeXmlError(
            f"{what}: refused XML containing a DOCTYPE/ENTITY declaration "
            "(entity-expansion DoS protection)"
        )
    return ET.fromstring(data)
