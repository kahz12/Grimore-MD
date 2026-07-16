"""
Adapter registry + dispatcher tests.

These tests pin down the registry contract itself: lookup by extension,
fallback semantics in the parser, and that the Markdown adapter's
output round-trips into ParsedNote without losing any of the
multi-format fields.
"""
from __future__ import annotations

import pytest

from grimore.ingest.adapters import for_path, supported_extensions
from grimore.ingest.adapters.base import AdapterOptions, ExtractedDocument
from grimore.ingest.adapters.markdown import MarkdownAdapter
from grimore.ingest.adapters.registry import register
from grimore.ingest.parser import MarkdownParser, ParsedNote


class TestRegistry:
    def test_markdown_adapter_registered_on_import(self):
        assert "md" in supported_extensions()
        assert isinstance(for_path("notes/whatever.md"), MarkdownAdapter)

    def test_lookup_is_case_insensitive_and_dot_tolerant(self):
        # Path("FOO.MD").suffix == ".MD" — registry normalises.
        assert for_path("FOO.MD") is for_path("foo.md")

    def test_unknown_extension_returns_none(self):
        assert for_path("strange.xyz") is None

    def test_register_is_idempotent_for_same_adapter(self):
        adapter = MarkdownAdapter()
        # Re-registering the same class for the same extension is a no-op
        # rather than an error — protects against double-import in tests.
        register(adapter)
        register(adapter)
        assert isinstance(for_path("z.md"), MarkdownAdapter)

    def test_register_rejects_conflicting_adapter_for_same_extension(self):
        class _RogueMarkdown:
            extensions = ("md",)
            binary = False
            mutable_frontmatter = True

            def extract(self, path, *, options):  # pragma: no cover - never called
                raise NotImplementedError

        with pytest.raises(ValueError, match="already registered"):
            register(_RogueMarkdown())


class TestMarkdownAdapterRoundTrip:
    def test_extract_produces_filled_extracted_document(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("---\ntitle: 'X'\n---\n\nbody one\n")
        doc = MarkdownAdapter().extract(f, options=AdapterOptions(vault_root=tmp_path))
        assert isinstance(doc, ExtractedDocument)
        assert doc.format == "md"
        assert doc.title == "X"
        assert "body one" in doc.text
        assert doc.content_hash and len(doc.content_hash) == 64
        assert doc.file_hash and len(doc.file_hash) == 64
        assert doc.content_hash != doc.file_hash  # one hashes text, the other bytes
        assert doc.size_bytes > 0
        assert doc.sections == []  # Markdown adapter doesn't segment yet

    def test_parser_dispatches_through_registry(self, tmp_path):
        f = tmp_path / "n.md"
        f.write_text("# H\nbody\n")
        note = MarkdownParser().parse_file(f, vault_root=tmp_path)
        assert isinstance(note, ParsedNote)
        assert note.format == "md"
        assert note.file_hash and len(note.file_hash) == 64
        assert note.title == "H"

    def test_parser_falls_back_to_markdown_for_unknown_extension(self, tmp_path):
        # For an extension with no registered adapter, the dispatcher
        # must not crash; it falls through to the Markdown adapter,
        # which then refuses the file because frontmatter.load() can't
        # parse arbitrary bytes.
        # Either UTF-8 garbage or a ValueError is acceptable; we just
        # require it doesn't import-crash or AttributeError.
        f = tmp_path / "thing.unknownext"
        f.write_text("plain text\n")
        # MarkdownAdapter will read it as a markdown body — no error path
        # to exercise. Just assert it parses as MD format-tagged content.
        note = MarkdownParser().parse_file(f, vault_root=tmp_path)
        assert note.format == "md"  # fallback adapter's declared format
        assert "plain text" in note.content
