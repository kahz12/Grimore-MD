"""
Tests for ``iter_vault_documents``.

Pins down: (a) multi-extension enumeration is deterministic and respects
``ignored_dirs``; (b) the sidecar tree is never re-enumerated, so the
files Grimore writes for its own purposes don't loop back into ingest.
"""
from __future__ import annotations

from pathlib import Path

from grimore.ingest.parser import iter_vault_documents


class TestIterVaultDocuments:
    def _populate(self, vault: Path) -> None:
        vault.mkdir(parents=True, exist_ok=True)
        (vault / "a.md").write_text("a\n")
        (vault / "b.md").write_text("b\n")
        (vault / "ignored").mkdir()
        (vault / "ignored" / "skip.md").write_text("skip\n")
        (vault / "Templates").mkdir()
        (vault / "Templates" / "tpl.md").write_text("tpl\n")
        (vault / "sub").mkdir()
        (vault / "sub" / "deep.md").write_text("deep\n")
        # An unsupported extension — must not be picked up when "md" is
        # the only enabled format.
        (vault / "sub" / "image.png").write_bytes(b"\x89PNG")

    def test_only_enabled_extensions_are_returned(self, tmp_path):
        vault = tmp_path / "vault"
        self._populate(vault)
        out = iter_vault_documents(
            vault, formats=["md"], ignored_dirs=["Templates", "ignored"],
        )
        names = {p.name for p in out}
        assert names == {"a.md", "b.md", "deep.md"}

    def test_ignored_dirs_filtered_as_path_segments(self, tmp_path):
        vault = tmp_path / "vault"
        self._populate(vault)
        out = iter_vault_documents(
            vault, formats=["md"], ignored_dirs=["ignored"],
        )
        # Templates wasn't passed as ignored, so its child should appear.
        assert any(p.name == "tpl.md" for p in out)
        # ignored/ child must be gone.
        assert not any("ignored" in p.parts for p in out)

    def test_multiple_extensions_dedupe_and_sort(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "z.md").write_text("z\n")
        (vault / "a.txt").write_text("a\n")
        (vault / "m.md").write_text("m\n")
        out = iter_vault_documents(
            vault, formats=["md", "txt"], ignored_dirs=[],
        )
        assert [p.name for p in out] == sorted(p.name for p in out)
        names = {p.name for p in out}
        assert names == {"z.md", "a.txt", "m.md"}

    def test_sidecar_tree_is_skipped(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "real.md").write_text("real\n")
        sidecar_root = vault / ".grimore" / "sidecars"
        sidecar_root.mkdir(parents=True)
        (sidecar_root / "book.pdf.md").write_text("autogen\n")

        out = iter_vault_documents(
            vault,
            formats=["md"],
            ignored_dirs=[],
            sidecar_dir=".grimore/sidecars",
        )
        names = {p.name for p in out}
        assert "real.md" in names
        assert "book.pdf.md" not in names

    def test_normalises_extensions_case_and_dot(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "n.md").write_text("n\n")
        # Caller passes ".MD" with a leading dot and uppercase — both
        # should be tolerated.
        out = iter_vault_documents(
            vault, formats=[".MD"], ignored_dirs=[],
        )
        assert {p.name for p in out} == {"n.md"}
