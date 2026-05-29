"""Tests for the knowledge-graph export.

Exercises the three writers (JSON, DOT, Obsidian Canvas) against a
hand-built 5-note vault. The semantic-suggested pass is exercised
with a tiny mocked Connector so the test stays Ollama-independent.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grimore.cognition.graph import (
    Graph,
    GraphEdge,
    GraphNode,
    SUPPORTED_FORMATS,
    build_graph,
    write_canvas,
    write_dot,
    write_graph,
    write_json,
)
from grimore.memory.db import Database


# ── fixture: a tiny vault with 5 cross-linked notes ──────────────────────


@pytest.fixture
def fixture_vault(tmp_path):
    """Build a vault + DB pair:

    * gothic.md      — wikilinks to [[buttress]] and [[romanesque]]
    * buttress.md    — wikilinks to [[gothic]]
    * romanesque.md  — no outbound links
    * jung.md        — disconnected (different topic)
    * orphan.md      — no inbound, no outbound; bare title
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    files = {
        "gothic.md":     "See [[buttress]] and [[romanesque]] for context.",
        "buttress.md":   "Flying buttress, see [[gothic]].",
        "romanesque.md": "Predecessor style. No outbound links here.",
        "jung.md":       "Carl Jung wrote on individuation.",
        "orphan.md":     "Alone in the vault.",
    }
    for name, body in files.items():
        (vault / name).write_text(body, encoding="utf-8")

    db = Database(str(tmp_path / "graph.db"))
    ids: dict[str, int] = {}
    for name in files:
        note_id = db.upsert_note(
            path=name,
            title=name.removesuffix(".md"),
            content_hash=f"hash-{name}",
        )
        ids[name] = note_id

    db.set_note_category(ids["gothic.md"],     "architecture/medieval")
    db.set_note_category(ids["buttress.md"],   "architecture/medieval")
    db.set_note_category(ids["romanesque.md"], "architecture/medieval")
    db.set_note_category(ids["jung.md"],       "psychology")
    # orphan stays uncategorised

    db.upsert_tags(ids["gothic.md"],     ["history", "architecture"])
    db.upsert_tags(ids["jung.md"],       ["psychology"])

    return SimpleNamespace(vault=vault, db=db, ids=ids)


def _make_session(fixture, *, connector=None):
    """A minimal Session-shape Object for graph.build_graph.

    ``build_graph`` only touches ``session.db``, ``session.vault_root``,
    and ``session.oracle.connector`` — passing a SimpleNamespace lets
    the tests stay clear of the warm Session's heavy LLM wiring.
    """
    oracle = SimpleNamespace(connector=connector or _NullConnector())
    return SimpleNamespace(
        db=fixture.db,
        vault_root=fixture.vault,
        oracle=oracle,
    )


class _NullConnector:
    """Mocked Connector that never proposes a neighbour.

    Lets the wikilink and contradiction passes be tested in isolation.
    """
    def find_similar_notes(self, *_a, **_kw):
        return []


# ── wikilink + node building ─────────────────────────────────────────────


def test_load_nodes_includes_every_note_with_tags_and_category(fixture_vault):
    session = _make_session(fixture_vault)
    g = build_graph(session, include_suggested=False)
    titles = {n.title for n in g.nodes}
    assert titles == {"gothic", "buttress", "romanesque", "jung", "orphan"}

    by_title = {n.title: n for n in g.nodes}
    assert by_title["gothic"].category == "architecture/medieval"
    assert by_title["orphan"].category is None
    assert sorted(by_title["gothic"].tags) == ["architecture", "history"]
    assert by_title["orphan"].tags == []


def test_wikilink_edges_resolve_and_dedupe(fixture_vault):
    session = _make_session(fixture_vault)
    g = build_graph(session, include_suggested=False)
    wiki = [e for e in g.edges if e.kind == "wikilink"]
    gothic_id = fixture_vault.ids["gothic.md"]
    buttress_id = fixture_vault.ids["buttress.md"]
    romanesque_id = fixture_vault.ids["romanesque.md"]

    pairs = {(e.src, e.dst) for e in wiki}
    assert (gothic_id, buttress_id) in pairs
    assert (gothic_id, romanesque_id) in pairs
    assert (buttress_id, gothic_id) in pairs
    # Bidirectional reference is two distinct directed edges, not collapsed.
    assert len([e for e in wiki if e.src == gothic_id]) == 2


def test_wikilink_unresolved_target_is_dropped(tmp_path):
    """`[[Bogus]]` referencing a non-existent note must not create an edge."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "alone.md").write_text("Refers to [[NotARealNote]].", encoding="utf-8")
    db = Database(str(tmp_path / "g.db"))
    db.upsert_note(path="alone.md", title="alone", content_hash="h")
    session = SimpleNamespace(
        db=db, vault_root=vault,
        oracle=SimpleNamespace(connector=_NullConnector()),
    )
    g = build_graph(session, include_suggested=False)
    assert [e for e in g.edges if e.kind == "wikilink"] == []


def test_wikilink_self_link_is_dropped(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "self.md").write_text("I link to [[self]].", encoding="utf-8")
    db = Database(str(tmp_path / "g.db"))
    db.upsert_note(path="self.md", title="self", content_hash="h")
    session = SimpleNamespace(
        db=db, vault_root=vault,
        oracle=SimpleNamespace(connector=_NullConnector()),
    )
    g = build_graph(session, include_suggested=False)
    assert all(e.src != e.dst for e in g.edges)


def test_wikilink_anchor_lands_in_label(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("See [[b#Heading]].", encoding="utf-8")
    (vault / "b.md").write_text("body", encoding="utf-8")
    db = Database(str(tmp_path / "g.db"))
    db.upsert_note(path="a.md", title="a", content_hash="h")
    db.upsert_note(path="b.md", title="b", content_hash="h")
    session = SimpleNamespace(
        db=db, vault_root=vault,
        oracle=SimpleNamespace(connector=_NullConnector()),
    )
    g = build_graph(session, include_suggested=False)
    wiki = [e for e in g.edges if e.kind == "wikilink"]
    assert len(wiki) == 1
    assert wiki[0].label == "Heading"


# ── suggested edges (mocked) ─────────────────────────────────────────────


class _FixedConnector:
    """Per-source-id canned hits. Used to exercise the suggested pass
    without a real embedding stack."""
    def __init__(self, edges_by_src, embeddings_per_note):
        self._by_src = edges_by_src
        self._calls = []
        self._embeddings_per_note = embeddings_per_note

    def find_similar_notes(self, query_vector, top_k, *, exclude_note_id,
                           dedupe_by_note):
        # The fixture's _populate_embeddings hands us a note id via the
        # exclude_note_id; we use it as the lookup key.
        self._calls.append(exclude_note_id)
        return list(self._by_src.get(exclude_note_id, []))


def _populate_embeddings(db, note_ids):
    """Insert a single dummy embedding row per note so the suggested
    pass actually visits each note. The vector content is unused —
    the mocked Connector ignores it."""
    from grimore.cognition.embedder import Embedder
    vec = Embedder.serialize_vector([1.0, 0.0, 0.0])
    with db._get_connection() as conn:
        for nid in note_ids:
            conn.execute(
                "INSERT INTO embeddings (note_id, chunk_index, text_content, vector) "
                "VALUES (?, 0, ?, ?)",
                (nid, "dummy", vec),
            )
        conn.commit()


def test_suggested_edges_respect_threshold_and_top(fixture_vault):
    ids = fixture_vault.ids
    _populate_embeddings(fixture_vault.db, ids.values())
    # gothic suggests jung (0.85), buttress (0.65); jung suggests orphan (0.5)
    edges = {
        ids["gothic.md"]: [
            {"note_id": ids["jung.md"],     "score": 0.85},
            {"note_id": ids["buttress.md"], "score": 0.65},
        ],
        ids["jung.md"]: [
            {"note_id": ids["orphan.md"], "score": 0.50},
        ],
    }
    connector = _FixedConnector(edges, embeddings_per_note=1)
    session = _make_session(fixture_vault, connector=connector)
    g = build_graph(
        session,
        include_suggested=True, suggested_top=3, suggested_threshold=0.7,
    )
    suggested = [e for e in g.edges if e.kind == "suggested"]
    assert len(suggested) == 1
    e = suggested[0]
    assert e.src == ids["gothic.md"]
    assert e.dst == ids["jung.md"]
    assert e.weight == pytest.approx(0.85)


def test_suggested_skipped_when_no_embeddings(fixture_vault):
    """Notes with zero embedding rows must not even be queried."""
    connector = _FixedConnector({}, embeddings_per_note=0)
    session = _make_session(fixture_vault, connector=connector)
    g = build_graph(session, include_suggested=True, suggested_top=3,
                    suggested_threshold=0.0)
    assert [e for e in g.edges if e.kind == "suggested"] == []
    assert connector._calls == []  # never reached the connector


def test_include_suggested_false_skips_pass(fixture_vault):
    _populate_embeddings(fixture_vault.db, fixture_vault.ids.values())
    connector = _FixedConnector(
        {fixture_vault.ids["gothic.md"]: [{"note_id": fixture_vault.ids["jung.md"],
                                           "score": 0.99}]},
        embeddings_per_note=1,
    )
    session = _make_session(fixture_vault, connector=connector)
    g = build_graph(session, include_suggested=False)
    assert [e for e in g.edges if e.kind == "suggested"] == []
    assert connector._calls == []


# ── contradiction edges ──────────────────────────────────────────────────


def test_contradiction_edges_lift_to_note_level(fixture_vault):
    """An open contradiction between claims in two notes becomes one edge."""
    ids = fixture_vault.ids
    with fixture_vault.db._get_connection() as conn:
        conn.execute(
            "INSERT INTO claims (note_path, claim_text, extracted_at) "
            "VALUES (?, ?, ?)",
            ("gothic.md", "Pointed arches transfer thrust outward.", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO claims (note_path, claim_text, extracted_at) "
            "VALUES (?, ?, ?)",
            ("romanesque.md", "Round arches kept thrust vertical.", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO contradictions "
            "(claim_a_id, claim_b_id, severity, explanation, status, detected_at) "
            "VALUES (1, 2, 'medium', 'argued in scholarship', 'open', '2025-01-02')"
        )
        # And one DISMISSED contradiction — must not show up.
        conn.execute(
            "INSERT INTO claims (note_path, claim_text, extracted_at) "
            "VALUES (?, ?, ?)",
            ("jung.md", "Individuation is a process.", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO claims (note_path, claim_text, extracted_at) "
            "VALUES (?, ?, ?)",
            ("buttress.md", "Individuation is unrelated.", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO contradictions "
            "(claim_a_id, claim_b_id, severity, explanation, status, detected_at) "
            "VALUES (3, 4, 'low', 'false alarm', 'dismissed', '2025-01-02')"
        )
        conn.commit()

    session = _make_session(fixture_vault)
    g = build_graph(session, include_suggested=False)
    contradicts = [e for e in g.edges if e.kind == "contradicts"]
    assert len(contradicts) == 1
    e = contradicts[0]
    pair = (e.src, e.dst)
    assert pair == (
        min(ids["gothic.md"], ids["romanesque.md"]),
        max(ids["gothic.md"], ids["romanesque.md"]),
    )
    assert e.label == "medium"


# ── writers ──────────────────────────────────────────────────────────────


def _sample_graph():
    nodes = [
        GraphNode(id=1, title="Gothic", path="gothic.md",
                  category="architecture", tags=["history"]),
        GraphNode(id=2, title="Buttress", path="buttress.md",
                  category="architecture", tags=[]),
        GraphNode(id=3, title="Jung", path="jung.md",
                  category="psychology", tags=["depth-psych"]),
    ]
    edges = [
        GraphEdge(src=1, dst=2, kind="wikilink", weight=1.0),
        GraphEdge(src=1, dst=3, kind="suggested", weight=0.81),
        GraphEdge(src=2, dst=3, kind="contradicts", weight=1.0, label="high"),
    ]
    return Graph(nodes=nodes, edges=edges)


def test_write_json_has_stable_schema_and_round_trips(tmp_path):
    g = _sample_graph()
    out = tmp_path / "g.json"
    write_json(g, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert {n["title"] for n in payload["nodes"]} == {"Gothic", "Buttress", "Jung"}
    edges = payload["edges"]
    assert {e["kind"] for e in edges} == {"wikilink", "suggested", "contradicts"}
    suggested = next(e for e in edges if e["kind"] == "suggested")
    assert suggested["weight"] == pytest.approx(0.81)


def test_write_dot_produces_valid_shape(tmp_path):
    g = _sample_graph()
    out = tmp_path / "g.dot"
    write_dot(g, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("digraph grimore {")
    assert text.rstrip().endswith("}")
    # Every node and edge appears in the output.
    for n in g.nodes:
        assert f"n{n.id}" in text
    assert "n1 -> n2" in text
    assert "n1 -> n3" in text
    assert "n2 -> n3" in text
    # Edge styling differs by kind — quick sanity checks.
    assert "style=dashed" in text   # suggested
    assert "dir=both" in text       # contradicts


def test_write_canvas_is_valid_json_and_groups_by_category(tmp_path):
    g = _sample_graph()
    out = tmp_path / "g.canvas"
    write_canvas(g, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {n["id"] for n in payload["nodes"]} == {"n1", "n2", "n3"}
    # Same-category notes (Gothic + Buttress) share x; Jung is in its own column.
    x_by_id = {n["id"]: n["x"] for n in payload["nodes"]}
    assert x_by_id["n1"] == x_by_id["n2"]
    assert x_by_id["n3"] != x_by_id["n1"]
    # Suggested edges carry an Obsidian colour code; contradicts use red ("1").
    by_id = {e["id"]: e for e in payload["edges"]}
    assert any(e.get("color") == "1" for e in by_id.values())  # contradicts
    assert any(e.get("color") == "4" for e in by_id.values())  # suggested


def test_write_graph_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        write_graph(_sample_graph(), Path("/tmp/x"), "yaml")


def test_supported_formats_listed():
    # Cheap sentinel test — the constant is the contract the CLI checks.
    assert "json" in SUPPORTED_FORMATS
    assert "dot" in SUPPORTED_FORMATS
    assert "obsidian-canvas" in SUPPORTED_FORMATS


def test_graph_stats_counts_by_kind():
    g = _sample_graph()
    s = g.stats()
    assert s["nodes"] == 3
    assert s["edges"] == 3
    assert s["by_kind"] == {"wikilink": 1, "suggested": 1, "contradicts": 1}
