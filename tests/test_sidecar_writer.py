"""
Sidecar / inline routing tests for FrontmatterWriter and LinkInjector.

Pins down: Markdown notes keep their v2.0 inline behaviour; every other
format routes through a mirrored sidecar tree at
``<vault>/<sidecar_dir>/<rel>.md``; ``write_sidecars=false`` is a clean
no-op; dry-run never touches disk; reruns merge cleanly without losing
user edits to sidecar bodies.
"""
from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from grimore.ingest.parser import ParsedNote
from grimore.output.frontmatter_writer import FrontmatterWriter, SIDECAR_FLAG_KEY
from grimore.output.link_injector import LinkInjector


def _note(path: Path, *, fmt: str, title: str = "T",
          content: str = "body", content_hash: str = "ch",
          file_hash: str = "fh") -> ParsedNote:
    return ParsedNote(
        path=path,
        title=title,
        metadata={},
        content=content,
        content_hash=content_hash,
        format=fmt,
        file_hash=file_hash,
    )


# ── FrontmatterWriter ────────────────────────────────────────────────────


class TestFrontmatterWriterInline:
    def test_md_routes_to_source_file(self, tmp_path):
        src = tmp_path / "note.md"
        src.write_text("---\ntitle: T\n---\nbody\n")
        note = _note(src, fmt="md")
        target = FrontmatterWriter().write_metadata(
            note, {"tags": ["a", "b"]},
            vault_root=tmp_path, dry_run=False,
        )
        assert target == src
        # Frontmatter merged in place; body preserved.
        post = frontmatter.load(src)
        assert post.metadata["tags"] == ["a", "b"]
        assert "body" in post.content
        # No sidecar created for MD.
        sidecar_root = tmp_path / ".grimore" / "sidecars"
        assert not sidecar_root.exists()

    def test_md_dry_run_returns_none(self, tmp_path):
        src = tmp_path / "note.md"
        src.write_text("---\n---\nbody\n")
        target = FrontmatterWriter().write_metadata(
            _note(src, fmt="md"), {"tags": ["x"]},
            vault_root=tmp_path, dry_run=True,
        )
        assert target is None
        # Source untouched.
        post = frontmatter.load(src)
        assert "tags" not in post.metadata


class TestFrontmatterWriterSidecar:
    def test_non_md_creates_sidecar_in_mirrored_tree(self, tmp_path):
        src = tmp_path / "Books" / "spec.pdf"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"%PDF-1.4 fake")
        note = _note(src, fmt="pdf", title="Spec", content_hash="ch1", file_hash="fh1")

        target = FrontmatterWriter().write_metadata(
            note, {"tags": ["specs"]},
            vault_root=tmp_path, dry_run=False,
        )
        expected = tmp_path / ".grimore" / "sidecars" / "Books" / "spec.pdf.md"
        assert target == expected
        assert expected.exists()

        post = frontmatter.load(expected)
        assert post.metadata["tags"] == ["specs"]
        assert post.metadata["source"] == str(src)
        assert post.metadata["format"] == "pdf"
        assert post.metadata["content_hash"] == "ch1"
        assert post.metadata["file_hash"] == "fh1"
        assert post.metadata[SIDECAR_FLAG_KEY] is True

    def test_non_md_dry_run_creates_nothing(self, tmp_path):
        src = tmp_path / "a.pdf"
        src.write_bytes(b"%PDF")
        target = FrontmatterWriter().write_metadata(
            _note(src, fmt="pdf"), {"tags": ["t"]},
            vault_root=tmp_path, dry_run=True,
        )
        assert target is None
        assert not (tmp_path / ".grimore" / "sidecars").exists()

    def test_write_sidecars_false_returns_none(self, tmp_path):
        src = tmp_path / "a.pdf"
        src.write_bytes(b"%PDF")
        target = FrontmatterWriter().write_metadata(
            _note(src, fmt="pdf"), {"tags": ["t"]},
            vault_root=tmp_path, write_sidecars=False, dry_run=False,
        )
        assert target is None
        assert not (tmp_path / ".grimore" / "sidecars").exists()

    def test_rerun_merges_without_losing_user_body(self, tmp_path):
        src = tmp_path / "doc.docx"
        src.write_bytes(b"PK\x03\x04 fake")
        note = _note(src, fmt="docx", title="Doc")
        sidecar = FrontmatterWriter().write_metadata(
            note, {"tags": ["v1"]}, vault_root=tmp_path, dry_run=False,
        )

        # Simulate a user editing the sidecar body to add notes.
        post = frontmatter.load(sidecar)
        post.content = post.content + "\n\nUser-added notes go here.\n"
        sidecar.write_text(frontmatter.dumps(post))

        # Second scan with updated tags.
        FrontmatterWriter().write_metadata(
            note, {"tags": ["v2"], "summary": "fresh"},
            vault_root=tmp_path, dry_run=False,
        )
        post2 = frontmatter.load(sidecar)
        assert post2.metadata["tags"] == ["v2"]
        assert post2.metadata["summary"] == "fresh"
        # User text survives the merge.
        assert "User-added notes" in post2.content

    def test_missing_vault_root_for_non_md_raises(self, tmp_path):
        src = tmp_path / "a.pdf"
        src.write_bytes(b"%PDF")
        with pytest.raises(ValueError, match="vault_root"):
            FrontmatterWriter().write_metadata(
                _note(src, fmt="pdf"), {"tags": ["t"]},
                vault_root=None, dry_run=False,
            )


# ── LinkInjector ─────────────────────────────────────────────────────────


class TestLinkInjectorSidecarMode:
    def test_md_writes_into_source(self, tmp_path):
        src = tmp_path / "n.md"
        src.write_text("body\n")
        injector = LinkInjector()
        target = injector.inject_for(
            source_path=src, format="md",
            connections=[{"title": "Other", "reason": "rel"}],
            vault_root=tmp_path, dry_run=False,
        )
        assert target == src
        text = src.read_text()
        assert "Suggested Connections" in text
        assert "[[Other]]" in text

    def test_non_md_writes_into_existing_sidecar(self, tmp_path):
        # Materialise sidecar first via the writer.
        src = tmp_path / "Books" / "x.pdf"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"%PDF")
        sidecar = FrontmatterWriter().write_metadata(
            _note(src, fmt="pdf"), {"tags": ["t"]},
            vault_root=tmp_path, dry_run=False,
        )
        assert sidecar is not None and sidecar.exists()

        target = LinkInjector().inject_for(
            source_path=src, format="pdf",
            connections=[{"title": "Sibling"}],
            sidecar_path=sidecar, dry_run=False,
        )
        assert target == sidecar
        body = sidecar.read_text()
        assert "Suggested Connections" in body
        assert "[[Sibling]]" in body
        # Original binary must NOT have been touched.
        assert src.read_bytes() == b"%PDF"

    def test_non_md_skips_when_sidecar_absent(self, tmp_path):
        src = tmp_path / "x.pdf"
        src.write_bytes(b"%PDF")
        target = LinkInjector().inject_for(
            source_path=src, format="pdf",
            connections=[{"title": "Y"}],
            vault_root=tmp_path, dry_run=False,
        )
        assert target is None  # logged as link_inject_skipped
        assert not (tmp_path / ".grimore").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "n.md"
        src.write_text("body\n")
        before = src.read_text()
        LinkInjector().inject_for(
            source_path=src, format="md",
            connections=[{"title": "Other"}],
            vault_root=tmp_path, dry_run=True,
        )
        assert src.read_text() == before
