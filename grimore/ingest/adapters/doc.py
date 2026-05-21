"""
Legacy .doc adapter — shells out to ``antiword`` when available.

Microsoft Word binary documents have no maintained pure-Python parser.
Rather than drag in unmaintained or partially-working dependencies, this
adapter is a thin wrapper around the venerable ``antiword`` CLI. It's
strictly opt-in by virtue of needing a binary on PATH: when antiword is
absent we raise a structured ``ValueError`` that the scan loop catches
and converts into a friendly skip log.

Why antiword (and not textract / catdoc / wvText):

* Antiword is in every major distro repo and Termux's package set.
* It's a single binary, dependency-free at runtime.
* It produces UTF-8 plain text on stdout — exactly what we need.

The adapter never offers heading-based sections (antiword's output is
flat text); the entire body lands in one anchor-free section, matching
the RTF / TXT shape.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar, Union

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_DOC_BYTES = 25_000_000

# Hard upper bound on how long antiword may run on one file. .doc parsing
# is normally sub-second; anything beyond this is almost certainly a wedged
# binary and we'd rather abort the file than stall the daemon.
_ANTIWORD_TIMEOUT_S = 30


def antiword_available() -> bool:
    """Whether the antiword binary is currently resolvable on PATH.

    Used by both the adapter's runtime check and the preflight probe.
    """
    return shutil.which("antiword") is not None


class DocAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("doc",)
    binary: ClassVar[bool] = True
    mutable_frontmatter: ClassVar[bool] = False

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        file_path = Path(path)
        if options.vault_root is not None:
            SecurityGuard.resolve_within_vault(file_path, options.vault_root)

        antiword = shutil.which("antiword")
        if antiword is None:
            # Surface a clear, actionable error. The caller (scan / daemon)
            # already logs ValueError as a skipped file with the message
            # attached, so the user sees what to do.
            raise ValueError(
                f"cannot read {file_path}: legacy .doc support requires the "
                "`antiword` binary on PATH. Install it (Linux: "
                "`apt install antiword`; Termux: `pkg install antiword`) "
                "or remove 'doc' from `[vault].formats`."
            )

        try:
            stat = file_path.stat()
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        size = stat.st_size
        if size > _MAX_DOC_BYTES:
            logger.warning(
                "doc_too_large", path=str(file_path), size=size, max=_MAX_DOC_BYTES,
            )
            raise ValueError(
                f"doc file exceeds {_MAX_DOC_BYTES} bytes: {file_path} ({size} bytes)"
            )

        try:
            # `-w 0` disables line wrapping; `-m UTF-8.txt` forces UTF-8
            # output. We pass the path as the final positional argument so
            # antiword can stat it directly (no stdin pipe).
            result = subprocess.run(
                [antiword, "-w", "0", "-m", "UTF-8.txt", str(file_path)],
                capture_output=True,
                check=False,
                timeout=_ANTIWORD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise ValueError(
                f"antiword timed out after {_ANTIWORD_TIMEOUT_S}s on {file_path}"
            ) from e
        except OSError as e:
            # PATH was right at shutil.which() time but exec failed (rare —
            # e.g. binary became unreadable between probe and exec).
            raise ValueError(f"failed to invoke antiword: {e}") from e

        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"antiword failed on {file_path} (exit {result.returncode}): "
                f"{stderr or 'no stderr'}"
            )

        text = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        title = file_path.stem
        return ExtractedDocument(
            source_path=file_path,
            format="doc",
            title=title,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata={},
            sections=[],
            size_bytes=size,
        )


register(DocAdapter())
