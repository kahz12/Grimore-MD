import pytest

from grimore.ingest.parser import MAX_NOTE_BYTES, MarkdownParser


class TestParseFile:
    def setup_method(self):
        self.parser = MarkdownParser()

    def test_title_from_frontmatter(self, tmp_path):
        f = tmp_path / "a.md"
        f.write_text('---\ntitle: "My Title"\n---\n\nbody text\n')
        note = self.parser.parse_file(f)
        assert note.title == "My Title"

    def test_title_from_h1_when_no_frontmatter_title(self, tmp_path):
        f = tmp_path / "b.md"
        f.write_text("# Heading One\n\nbody\n")
        note = self.parser.parse_file(f)
        assert note.title == "Heading One"

    def test_title_falls_back_to_filename_stem(self, tmp_path):
        f = tmp_path / "my-note.md"
        f.write_text("no heading here\n")
        note = self.parser.parse_file(f)
        assert note.title == "my-note"

    def test_content_excludes_frontmatter(self, tmp_path):
        f = tmp_path / "c.md"
        f.write_text("---\ntitle: X\n---\n\nactual body\n")
        note = self.parser.parse_file(f)
        assert "title: X" not in note.content
        assert "actual body" in note.content

    def test_metadata_parsed(self, tmp_path):
        f = tmp_path / "d.md"
        f.write_text("---\ntags: [a, b]\nprivacy: local\n---\nbody\n")
        note = self.parser.parse_file(f)
        assert note.metadata["privacy"] == "local"
        assert note.metadata["tags"] == ["a", "b"]

    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "e.md"
        f.write_text("same content\n")
        h1 = self.parser.parse_file(f).content_hash
        h2 = self.parser.parse_file(f).content_hash
        assert h1 == h2

    def test_hash_changes_with_content(self, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("one\n")
        h1 = self.parser.parse_file(f).content_hash
        f.write_text("two\n")
        h2 = self.parser.parse_file(f).content_hash
        assert h1 != h2

    def test_rejects_oversized_file(self, tmp_path):
        f = tmp_path / "huge.md"
        f.write_bytes(b"x" * (MAX_NOTE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            self.parser.parse_file(f)

    def test_accepts_file_at_size_limit(self, tmp_path):
        f = tmp_path / "edge.md"
        f.write_bytes(b"x" * MAX_NOTE_BYTES)
        # Should not raise
        note = self.parser.parse_file(f)
        assert note.title == "edge"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="cannot stat"):
            self.parser.parse_file(tmp_path / "nope.md")

    def test_vault_root_optional_validates_when_set(self, tmp_path):
        # B-02: parser re-validates vault scope as defence-in-depth.
        vault = tmp_path / "vault"
        outside = tmp_path / "outside"
        vault.mkdir()
        outside.mkdir()
        rogue = outside / "rogue.md"
        rogue.write_text("# rogue\n")
        with pytest.raises(ValueError, match="escapes vault"):
            self.parser.parse_file(rogue, vault_root=vault)

    def test_vault_root_accepts_in_scope_file(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        ok = vault / "ok.md"
        ok.write_text("# ok\nbody\n")
        note = self.parser.parse_file(ok, vault_root=vault)
        assert note.title == "ok"
