"""Markdown notes are chunked per heading and re-chunked once after an upgrade."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from grimore.cognition.chunker import chunk_sections
from grimore.cognition.embedder import Embedder
from grimore.daemon import GrimoreDaemon
from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.markdown import MarkdownAdapter, markdown_sections
from grimore.ingest.extraction_profile import MARKDOWN_INDEX_PROFILE, markdown_layout_stale
from grimore.ingest.parser import MarkdownParser
from grimore.memory.db import Database
from grimore.utils.config import Config
from grimore.utils.security import SecurityGuard

NOTE = """Loose text before any heading.

# Kotlin Notes

Kotlin runs on the JVM.

## Null Safety ##

Types are non-null by default.

```kotlin
# not a heading
val name: String? = null
```

### Elvis Operator

Use ?: for a default.

## Coroutines

Lightweight threads.
"""


def test_sections_follow_headings_outside_code_fences():
    sections = markdown_sections(NOTE, "Kotlin Notes")
    assert [(s.heading, s.breadcrumb) for s in sections] == [
        (None, ("Kotlin Notes",)),
        (None, ("Kotlin Notes",)),
        ("Null Safety", ("Kotlin Notes", "Null Safety")),
        ("Elvis Operator", ("Kotlin Notes", "Null Safety", "Elvis Operator")),
        ("Coroutines", ("Kotlin Notes", "Coroutines")),
    ]
    assert "# not a heading" in sections[2].text
    assert all(not s.text.startswith("#") for s in sections)
    assert [s.order for s in sections] == list(range(5))


def test_h1_that_is_not_the_title_is_a_heading():
    sections = markdown_sections("# Chapter One\n\nText.\n", "Book")
    assert [(s.heading, s.breadcrumb) for s in sections] == [
        ("Chapter One", ("Book", "Chapter One")),
    ]


def test_adapter_titles_and_sections_the_note(tmp_path):
    note = tmp_path / "kotlin.md"
    note.write_text(NOTE, encoding="utf-8")
    doc = MarkdownAdapter().extract(note, options=AdapterOptions(vault_root=tmp_path))
    assert doc.title == "Kotlin Notes"
    assert doc.sections[2].heading == "Null Safety"


def test_chunks_carry_the_breadcrumb_within_the_budget():
    sections = markdown_sections("# T\n\n## Long\n\n" + "word " * 400, "T")
    chunks = chunk_sections(sections, max_chars=300)
    assert len(chunks) > 1
    assert all(c.text.startswith("T > Long\n\nword") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.heading for c in chunks} == {"Long"}


def test_only_markdown_with_another_profile_is_stale():
    assert markdown_layout_stale(("h", 1, 1, "md", None))
    assert markdown_layout_stale(("h", 1, 1, "md", "md-old"))
    assert not markdown_layout_stale(("h", 1, 1, "md", MARKDOWN_INDEX_PROFILE))
    assert not markdown_layout_stale(("h", 1, 1, "txt", None))
    assert not markdown_layout_stale(None)


@pytest.mark.parametrize("surface", ["scan", "daemon"])
def test_old_layout_is_rechunked_without_retagging(tmp_path, monkeypatch, surface):
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "kotlin.md"
    source.write_text(NOTE, encoding="utf-8")
    config = Config()
    config.vault.path = str(vault)
    config.memory.db_path = str(tmp_path / "index.db")
    config.output.dry_run = False
    config.output.auto_commit = False

    parser = MagicMock(wraps=MarkdownParser())
    tagger = MagicMock()
    tagger.tag_note.return_value = {"tags": [], "summary": "", "category": ""}
    embedder = MagicMock()
    embedder.model = "stub"
    embedder.serialize_vector.side_effect = Embedder.serialize_vector
    embedder.embed_batch.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    writer = MagicMock()
    writer.write_metadata.return_value = source
    writer.preview_metadata.return_value = None

    with Database(config.memory.db_path) as db:
        if surface == "scan":
            from grimore.cli import app

            monkeypatch.setattr("grimore.cli.load_config", lambda: config)
            monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a: None)
            monkeypatch.setattr("grimore.cli.Database", lambda *a: db)
            monkeypatch.setattr("grimore.cli.MarkdownParser", lambda: parser)
            monkeypatch.setattr("grimore.cli.LLMRouter", lambda *a: MagicMock())
            monkeypatch.setattr("grimore.cli.Embedder", lambda *a, **kw: embedder)
            monkeypatch.setattr("grimore.cli.Tagger", lambda *a: tagger)
            monkeypatch.setattr("grimore.cli.FrontmatterWriter", lambda: writer)

            def process(*, dry_run=False):
                result = CliRunner().invoke(
                    app, ["scan", "--dry-run" if dry_run else "--no-dry-run"],
                )
                assert result.exit_code == 0, result.output
        else:
            daemon = GrimoreDaemon.__new__(GrimoreDaemon)
            daemon.config, daemon.db, daemon.vault_root = config, db, vault
            daemon.parser, daemon.security = parser, SecurityGuard(str(vault))
            daemon.event_log, daemon.notifier = MagicMock(), MagicMock()
            daemon.writer, daemon.tagger, daemon.embedder = writer, tagger, embedder
            daemon._counter_lock = threading.Lock()
            daemon.processed_count = 0

            def process(*, dry_run=False):
                config.output.dry_run = dry_run
                daemon.process_file(source)

        def headings():
            note_id = db.get_note_id_by_path(str(source))
            return [db.get_chunk_records(note_id)[i][2] for i in sorted(db.get_chunk_records(note_id))]

        process()
        indexed = headings()
        assert indexed == [None, None, "Null Safety", "Elvis Operator", "Coroutines"]
        assert db.get_ingest_state(str(source))[4] == MARKDOWN_INDEX_PROFILE
        assert tagger.tag_note.call_count == 1

        # A note indexed before section chunking: bare body chunks, no profile.
        with db._get_connection() as conn:
            conn.execute("UPDATE notes SET extraction_profile = NULL")
            conn.execute("UPDATE embeddings SET heading = NULL, chunk_hash = 'old-layout'")
        process(dry_run=True)
        assert headings() == [None] * len(indexed)
        assert db.get_ingest_state(str(source))[4] is None

        embedded = embedder.embed_batch.call_count
        process()
        assert headings() == indexed
        assert db.get_ingest_state(str(source))[4] == MARKDOWN_INDEX_PROFILE
        assert embedder.embed_batch.call_count == embedded + 1
        assert tagger.tag_note.call_count == 1
        writer.write_metadata.assert_called_once()

        parsed = parser.parse_file.call_count
        process()
        assert parser.parse_file.call_count == parsed
