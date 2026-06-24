"""
Frontmatter Writer.

Writes Grimore's cognitive metadata (tags, summary, category, …) back
where the user can see it. For Markdown documents the metadata goes
inline as YAML frontmatter on the source file. For every other format
the metadata lives in a *sidecar* ``.md`` mirrored under the vault's
``sidecar_dir`` (default ``.grimore/sidecars/``); the original binary
is never touched.

See ``docs/MULTIFORMAT_BLUEPRINT.md`` §7 for the contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import frontmatter

from grimore.ingest.parser import ParsedNote
from grimore.utils.atomic import atomic_write
from grimore.utils.logger import get_logger
from grimore.utils.paths import sidecar_path_for

logger = get_logger(__name__)

# Frontmatter flag that marks a file as a Grimore-managed sidecar.
# Distinct from ``grimore_generated`` (Synthesizer output) so the two
# kinds of generated notes can be filtered independently downstream.
SIDECAR_FLAG_KEY = "grimore_sidecar"


class FrontmatterWriter:
    """Routes metadata writes to the right place per document format.

    Public API:

    * :py:meth:`write_metadata` — preferred, takes a ``ParsedNote`` and
      handles inline vs sidecar routing.
    * :py:meth:`write_inline` — original Path-based behaviour, retained
      so callers that only have a file path (e.g. legacy test code) keep
      working.
    """

    def write_metadata(
        self,
        note: ParsedNote,
        updates: dict[str, Any],
        *,
        vault_root: Optional[Union[str, Path]] = None,
        sidecar_dir: str = ".grimore/sidecars",
        write_sidecars: bool = True,
        dry_run: bool = True,
    ) -> Optional[Path]:
        """Write ``updates`` to the right target for ``note``.

        Returns the path actually written (source for MD, sidecar for
        non-MD), or ``None`` when no write happened (``dry_run``, or
        ``write_sidecars=False`` for a non-MD note).
        """
        if note.format == "md":
            self.write_inline(note.path, updates, dry_run=dry_run)
            return None if dry_run else note.path

        if not write_sidecars:
            logger.info(
                "sidecar_write_skipped",
                path=str(note.path),
                reason="write_sidecars=false",
            )
            return None

        if vault_root is None:
            # Caller is misconfigured: routing to a sidecar needs to know
            # the vault root so the mirror path is reproducible. Falling
            # back to the source's parent would scatter sidecars all over
            # the vault; refuse the write loudly instead.
            raise ValueError(
                "vault_root is required for sidecar writes; "
                f"refusing to materialise a sidecar for {note.path}"
            )

        sidecar = sidecar_path_for(note.path, vault_root, sidecar_dir)
        self._write_sidecar(note, sidecar, updates, dry_run=dry_run)
        return None if dry_run else sidecar

    # ── Inline (Markdown) ─────────────────────────────────────────────────

    def write_inline(
        self,
        file_path: Path,
        metadata_updates: dict[str, Any],
        *,
        dry_run: bool = True,
    ) -> None:
        """Original Markdown round-trip: merge YAML frontmatter in place."""
        if dry_run:
            logger.info(
                "dry_run_metadata_update",
                path=str(file_path),
                updates=metadata_updates,
            )
            return

        try:
            post = frontmatter.load(file_path)
            post.metadata.update(metadata_updates)
            # Encode explicitly rather than relying on frontmatter.dump()'s
            # handling of a binary handle — that differs across versions
            # (1.1.0 encodes; 1.3.0 writes str, which fails on a "wb" file).
            atomic_write(
                file_path,
                lambda fh: fh.write(frontmatter.dumps(post).encode("utf-8")),
                mode="wb",
            )
            logger.info("metadata_updated", path=str(file_path))
        except Exception as e:
            logger.error(
                "metadata_update_failed", path=str(file_path), error=str(e),
            )

    # ── Sidecar (non-MD) ──────────────────────────────────────────────────

    def _write_sidecar(
        self,
        note: ParsedNote,
        sidecar_path: Path,
        metadata_updates: dict[str, Any],
        *,
        dry_run: bool,
    ) -> None:
        """Materialise (or refresh) the sidecar ``.md`` for a non-MD note.

        The sidecar is itself a valid Markdown note: Obsidian users can
        open it, link to it, see suggested-connections sections injected
        by :class:`LinkInjector`, etc. The ``grimore_sidecar: true`` flag
        in frontmatter excludes it from Synthesizer / re-ingest paths.
        """
        if dry_run:
            logger.info(
                "dry_run_sidecar_update",
                source=str(note.path),
                sidecar=str(sidecar_path),
                updates=metadata_updates,
            )
            return

        # Build the merged frontmatter. The sidecar's own "source"
        # pointer, format, and content/file hashes are managed by us;
        # caller-supplied updates take precedence on everything else so
        # tags/summary/category can be refreshed without losing them.
        existing: dict[str, Any] = {}
        existing_body = ""
        if sidecar_path.exists():
            try:
                post = frontmatter.load(sidecar_path)
                existing = dict(post.metadata)
                existing_body = post.content
            except Exception as e:
                # A corrupt sidecar shouldn't poison the scan — overwrite
                # rather than crash, and log loudly so the user notices.
                logger.warning(
                    "sidecar_corrupt_overwriting",
                    sidecar=str(sidecar_path), error=str(e),
                )

        merged: dict[str, Any] = {**existing, **metadata_updates}
        # Provenance keys are owned by Grimore — caller updates can never
        # override them (otherwise a stale value would mask the truth).
        merged["source"] = str(note.path)
        merged["format"] = note.format
        merged["content_hash"] = note.content_hash
        if note.file_hash:
            merged["file_hash"] = note.file_hash
        merged[SIDECAR_FLAG_KEY] = True
        if "title" not in merged or not merged["title"]:
            merged["title"] = note.title

        # Preserve any human-edited body below the auto-generated header
        # so a user can scribble notes on a PDF and not lose them on the
        # next scan. We rewrite only the header line + any prior
        # ## Suggested Connections section is left for LinkInjector.
        body = existing_body.strip() or self._default_sidecar_body(note)

        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(body, **merged)
        try:
            atomic_write(
                sidecar_path,
                lambda fh: fh.write(frontmatter.dumps(post).encode("utf-8")),
                mode="wb",
            )
            logger.info(
                "sidecar_written", source=str(note.path), sidecar=str(sidecar_path),
            )
        except Exception as e:
            logger.error(
                "sidecar_write_failed",
                source=str(note.path), sidecar=str(sidecar_path), error=str(e),
            )

    @staticmethod
    def _default_sidecar_body(note: ParsedNote) -> str:
        """A short, human-readable header pointing back to the original."""
        return (
            f"# {note.title}\n\n"
            f"> Auto-generated by Grimore for `{note.path.name}` "
            f"({note.format}). Edit the source document, then re-run "
            f"`grimore scan`."
        )
