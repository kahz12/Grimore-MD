"""
Link Injector.

Injects semantic connections (wikilinks) into the right Markdown target:

* For native Markdown notes the section is appended (or replaced in place)
  on the source file itself, exactly as in v2.0.
* For non-Markdown documents the section lands in the sidecar ``.md``
  that :class:`FrontmatterWriter` has already materialised under the
  vault's sidecar tree. The original binary is never touched.

When a non-MD note has no sidecar (e.g. ``vault.write_sidecars = false``)
the injection is a logged no-op rather than an error — DB-only mode is
a legitimate user choice and the connector should not fail loudly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

from grimore.utils.atomic import atomic_write
from grimore.utils.logger import get_logger
from grimore.utils.paths import sidecar_path_for

logger = get_logger(__name__)

# Characters that would let a malicious title escape a wikilink or markdown list item.
_WIKILINK_FORBIDDEN = re.compile(r"[\[\]\|\r\n]+")

_SECTION_HEADER = "## 🔗 Suggested Connections"


def _safe_wikilink_title(raw: str) -> str:
    """Sanitizes a title to ensure it's a valid and safe Markdown wikilink."""
    title = _WIKILINK_FORBIDDEN.sub(" ", str(raw))
    return title.strip()[:200] or "Untitled"


def _safe_reason(raw: str) -> str:
    """Sanitizes the reason text to prevent breaking Markdown formatting."""
    text = re.sub(r"[\r\n]+", " ", str(raw))
    return text.strip()[:300]


class LinkInjector:
    """Handles the modification of Markdown files to include semantic links."""

    def inject_links(
        self,
        file_path: Path,
        connections: list[dict],
        dry_run: bool = True,
    ):
        """Append/replace ``## 🔗 Suggested Connections`` on ``file_path``.

        Always treats ``file_path`` as a Markdown target. Callers that
        need format-aware routing should call :py:meth:`inject_for`
        instead.
        """
        if not connections:
            return

        new_section = self._render_section(connections)

        if dry_run:
            logger.info(
                "dry_run_link_injection",
                path=str(file_path),
                connections=[c["title"] for c in connections],
            )
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = re.compile(
                rf"\n\n{re.escape(_SECTION_HEADER)}.*?(?=\n\n|\Z)", re.DOTALL,
            )
            if pattern.search(content):
                new_content = pattern.sub(new_section, content)
            else:
                new_content = content.rstrip() + new_section

            atomic_write(
                file_path,
                lambda fh: fh.write(new_content.encode("utf-8")),
                mode="wb",
            )
            logger.info("links_injected", path=str(file_path))
        except Exception as e:
            logger.error(
                "link_injection_failed", path=str(file_path), error=str(e),
            )

    def inject_for(
        self,
        *,
        source_path: Path,
        format: str,
        connections: list[dict],
        sidecar_path: Optional[Path] = None,
        vault_root: Optional[Union[str, Path]] = None,
        sidecar_dir: str = ".grimore/sidecars",
        dry_run: bool = True,
    ) -> Optional[Path]:
        """Format-aware injection. Returns the path written, or None.

        * ``format == "md"`` → writes to ``source_path``.
        * Otherwise → writes to ``sidecar_path`` if given; else computes
          the canonical sidecar under ``vault_root / sidecar_dir``. If
          that file doesn't yet exist (the FrontmatterWriter hasn't been
          run, or write_sidecars is off), the injection is skipped with
          a log line rather than an error.
        """
        if not connections:
            return None

        if format == "md":
            self.inject_links(source_path, connections, dry_run=dry_run)
            return None if dry_run else source_path

        target = sidecar_path
        if target is None:
            if vault_root is None:
                logger.warning(
                    "link_inject_skipped",
                    source=str(source_path),
                    reason="no_target_and_no_vault_root",
                )
                return None
            target = sidecar_path_for(source_path, vault_root, sidecar_dir)

        if not target.exists():
            logger.info(
                "link_inject_skipped",
                source=str(source_path),
                sidecar=str(target),
                reason="sidecar_absent",
            )
            return None

        self.inject_links(target, connections, dry_run=dry_run)
        return None if dry_run else target

    @staticmethod
    def _render_section(connections: list[dict]) -> str:
        """Return the full Markdown block (leading blank lines included)."""
        links_content = "\n".join([
            f"- [[{_safe_wikilink_title(c['title'])}]] — "
            f"{_safe_reason(c.get('reason', 'Semantic connection found.'))}"
            for c in connections
        ])
        return f"\n\n{_SECTION_HEADER}\n{links_content}\n"
