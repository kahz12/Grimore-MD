"""
Knowledge-graph export.

Crawls the vault's relationships into a portable graph and writes it
out in one of three formats:

* **JSON** — ``{nodes, edges}`` with a stable schema; the canonical
  interchange shape.
* **DOT** — Graphviz source; pipe to ``dot -Tsvg`` for a static render.
* **Obsidian Canvas** — drops the graph into a ``.canvas`` file that
  opens directly in Obsidian, with nodes grouped by category on a
  rough grid.

Three edge kinds are produced:

* ``wikilink`` — explicit ``[[Title]]`` references found in the note
  body (Markdown notes only — non-text formats don't have a body to
  scan and their sidecar markdown is generated rather than authored).
* ``suggested`` — top-N most semantically similar notes per source
  (``Connector.find_similar_notes`` on the mean of the note's chunk
  vectors); weight is the cosine similarity.
* ``contradicts`` — non-dismissed claim-pair edges from Black Mirror's
  ``contradictions`` table, lifted up to the note level.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from grimore.cognition.embedder import Embedder

# Matches `[[Title]]` and `[[Title#anchor]]` references in note bodies.
# The same shape the Oracle uses for citations — keeps wikilinks and
# citations on a single grammar.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]*)?\]\]")


@dataclass(frozen=True)
class GraphNode:
    """One vault note as a graph vertex."""
    id: int
    title: str
    path: str
    category: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdge:
    """A relationship between two notes.

    ``kind`` is one of ``wikilink`` / ``suggested`` / ``contradicts``.
    ``weight`` is in ``[0, 1]`` for ``suggested`` (cosine), ``1.0`` for
    the unweighted kinds — keeps every edge comparable in the JSON
    output. ``label`` carries the contradiction severity or the
    wikilink anchor when present.
    """
    src: int
    dst: int
    kind: str
    weight: float = 1.0
    label: Optional[str] = None


@dataclass
class Graph:
    """The fully-resolved graph; written by the format-specific helpers."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def stats(self) -> dict:
        """Compact ``{nodes, edges, by_kind}`` summary for CLI output."""
        by_kind: dict[str, int] = {}
        for e in self.edges:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "by_kind": by_kind,
        }


# ── building ─────────────────────────────────────────────────────────────


def _load_nodes(db) -> tuple[list[GraphNode], dict[int, GraphNode], dict[str, int]]:
    """Pull every note from ``notes`` and join its tag list.

    Returns ``(nodes_list, by_id, by_title_lower)``. The two indexes
    cache lookups the edge-building passes do thousands of times for
    larger vaults — a vault with N notes asks each pass for O(N)
    title→id resolutions.
    """
    nodes: list[GraphNode] = []
    by_id: dict[int, GraphNode] = {}
    by_title: dict[str, int] = {}

    with db._get_connection() as conn:
        note_rows = conn.execute(
            "SELECT id, path, title, category FROM notes ORDER BY id"
        ).fetchall()
        # One JOIN gets every (note_id, tag_name) pair in a single
        # round-trip; faster than N separate get_tags_for_note calls.
        tag_rows = conn.execute(
            "SELECT nt.note_id, t.name "
            "FROM note_tags nt JOIN tags t ON t.id = nt.tag_id "
            "ORDER BY nt.note_id, t.name"
        ).fetchall()

    tags_by_note: dict[int, list[str]] = {}
    for note_id, tag in tag_rows:
        tags_by_note.setdefault(int(note_id), []).append(tag)

    for note_id, path, title, category in note_rows:
        node = GraphNode(
            id=int(note_id),
            title=title or Path(path).stem,
            path=path,
            category=category or None,
            tags=list(tags_by_note.get(int(note_id), [])),
        )
        nodes.append(node)
        by_id[node.id] = node
        # Title collisions: first-seen wins. Two notes with the same
        # stem is a known vault problem; logging once on collision
        # rather than every wikilink resolve keeps the output quiet.
        key = node.title.lower()
        if key not in by_title:
            by_title[key] = node.id

    return nodes, by_id, by_title


def _wikilink_edges(
    nodes: list[GraphNode],
    by_title: dict[str, int],
    vault_root: Path,
) -> list[GraphEdge]:
    """Scan Markdown bodies for ``[[Title]]`` references.

    Non-Markdown sources are skipped — their bodies live in binary
    formats whose text was lifted by the ingest adapter into sidecars
    that *we* wrote, so they don't contain user-authored wikilinks.
    Self-links and unresolved titles are dropped without warning;
    duplicate (src, dst) pairs collapse to one edge.
    """
    edges: list[GraphEdge] = []
    seen: set[tuple[int, int]] = set()

    for node in nodes:
        if not node.path.lower().endswith(".md"):
            continue
        src_path = vault_root / node.path
        if not src_path.is_file():
            continue
        try:
            body = src_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _WIKILINK_RE.finditer(body):
            raw = match.group(1).strip()
            # Anchor (`Title#p.4` / `Title#Heading`) — store as label
            # but resolve the target by title alone.
            if "#" in raw:
                title, anchor = raw.split("#", 1)
                title = title.strip()
                anchor = anchor.strip() or None
            else:
                title, anchor = raw, None
            dst = by_title.get(title.lower())
            if dst is None or dst == node.id:
                continue
            key = (node.id, dst)
            if key in seen:
                continue
            seen.add(key)
            edges.append(GraphEdge(
                src=node.id, dst=dst, kind="wikilink",
                weight=1.0, label=anchor,
            ))
    return edges


def _suggested_edges(
    nodes: list[GraphNode],
    connector,
    db,
    *,
    top: int,
    threshold: float,
) -> list[GraphEdge]:
    """Per-note top-``top`` semantic neighbours above ``threshold``.

    For each note we mean-pool the chunk vectors (matches the same
    trick the MCP ``grimore_connect`` tool uses) and route through
    ``Connector.find_similar_notes`` so the matmul + dedupe paths
    we already trust are reused. Notes with zero chunks (failed
    ingest, image-only PDF without OCR) are silently skipped.
    """
    if top <= 0:
        return []

    edges: list[GraphEdge] = []
    seen: set[tuple[int, int]] = set()

    for node in nodes:
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT vector FROM embeddings WHERE note_id = ? AND vector IS NOT NULL",
                (node.id,),
            ).fetchall()
        if not rows:
            continue
        try:
            vectors = [Embedder.deserialize_vector(r[0]) for r in rows]
        except Exception:
            # A ragged-dim row would blow up the deserializer; skip
            # the note rather than abort the whole export.
            continue
        if not vectors:
            continue
        dim = len(vectors[0])
        avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        normed = Embedder.normalize(avg)

        hits = connector.find_similar_notes(
            normed, top_k=top, exclude_note_id=node.id, dedupe_by_note=True,
        )
        for hit in hits:
            score = float(hit.get("score") or 0.0)
            if score < threshold:
                continue
            dst = int(hit["note_id"])
            if dst == node.id:
                continue
            key = (node.id, dst)
            if key in seen:
                continue
            seen.add(key)
            edges.append(GraphEdge(
                src=node.id, dst=dst, kind="suggested",
                weight=score, label=None,
            ))
    return edges


def _contradiction_edges(
    db,
    by_path: dict[str, int],
) -> list[GraphEdge]:
    """Lift open / resolved contradictions up to the note level.

    The claims table keys on ``note_path``, not ``note_id``, so we
    re-resolve through ``by_path``. Dismissed contradictions are
    excluded — the user already decided they're not interesting.
    """
    edges: list[GraphEdge] = []
    seen: set[tuple[int, int]] = set()
    try:
        with db._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ca.note_path, cb.note_path, c.severity
                FROM contradictions c
                JOIN claims ca ON ca.id = c.claim_a_id
                JOIN claims cb ON cb.id = c.claim_b_id
                WHERE c.status <> 'dismissed'
                """
            ).fetchall()
    except Exception:
        # Mirror tables only exist after the first scan; on a fresh
        # vault the table is missing and that's fine.
        return edges

    for path_a, path_b, severity in rows:
        src = by_path.get(path_a)
        dst = by_path.get(path_b)
        if src is None or dst is None or src == dst:
            continue
        # Contradictions are symmetric — collapse (a,b) and (b,a) into
        # one canonical edge so the output stays deduped.
        key = (min(src, dst), max(src, dst))
        if key in seen:
            continue
        seen.add(key)
        edges.append(GraphEdge(
            src=key[0], dst=key[1], kind="contradicts",
            weight=1.0, label=severity or None,
        ))
    return edges


def build_graph(
    session,
    *,
    include_suggested: bool = True,
    suggested_top: int = 3,
    suggested_threshold: float = 0.7,
) -> Graph:
    """Crawl ``session``'s DB + vault and return the fully-resolved Graph.

    ``include_suggested`` flips the semantic-neighbour pass on or off;
    the wikilink and contradiction passes are always cheap enough to
    run unconditionally. ``suggested_top`` caps per-source fan-out;
    ``suggested_threshold`` filters out weak matches that would just
    clutter the rendered graph.
    """
    db = session.db
    nodes, by_id, by_title = _load_nodes(db)
    by_path = {n.path: n.id for n in nodes}

    edges: list[GraphEdge] = []
    edges.extend(_wikilink_edges(nodes, by_title, session.vault_root))

    if include_suggested and nodes:
        edges.extend(_suggested_edges(
            nodes,
            session.oracle.connector,
            db,
            top=suggested_top,
            threshold=suggested_threshold,
        ))

    edges.extend(_contradiction_edges(db, by_path))
    return Graph(nodes=nodes, edges=edges)


# ── writers ──────────────────────────────────────────────────────────────


def write_json(graph: Graph, path: Path) -> Path:
    """Dump the graph as a pretty-printed JSON document.

    Schema (stable): ``{"version": 1, "nodes": [...], "edges": [...]}``.
    Field names match the dataclass attributes so downstream consumers
    can mirror them without translation.
    """
    payload = {
        "version": 1,
        "nodes": [
            {
                "id": n.id,
                "title": n.title,
                "path": n.path,
                "category": n.category,
                "tags": list(n.tags),
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "src": e.src,
                "dst": e.dst,
                "kind": e.kind,
                "weight": round(e.weight, 6),
                "label": e.label,
            }
            for e in graph.edges
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _dot_escape(text: str) -> str:
    """Escape a string for safe embedding inside a DOT label literal."""
    return (text or "").replace("\\", "\\\\").replace("\"", "\\\"")


def write_dot(graph: Graph, path: Path) -> Path:
    """Emit Graphviz DOT source for ``dot -Tsvg`` rendering.

    Edge styling distinguishes the three kinds at a glance:

    * ``wikilink``    — solid black
    * ``suggested``   — dashed gray, with the cosine score as the label
    * ``contradicts`` — solid red, bidirectional arrowhead
    """
    lines: list[str] = ["digraph grimore {"]
    lines.append('  graph [overlap=false, splines=true, fontname="Helvetica"];')
    lines.append('  node  [shape=ellipse, fontname="Helvetica"];')
    lines.append('  edge  [fontname="Helvetica", fontsize=10];')
    for n in graph.nodes:
        cat = f" — {_dot_escape(n.category)}" if n.category else ""
        lines.append(f'  n{n.id} [label="{_dot_escape(n.title)}{cat}"];')
    for e in graph.edges:
        if e.kind == "wikilink":
            attrs = 'color="#333", penwidth=1.4'
        elif e.kind == "suggested":
            attrs = f'color="#888", style=dashed, label="{e.weight:.2f}"'
        else:  # contradicts
            attrs = 'color="#c0392b", penwidth=1.6, dir=both'
        lines.append(f"  n{e.src} -> n{e.dst} [{attrs}];")
    lines.append("}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_canvas(graph: Graph, path: Path) -> Path:
    """Write an Obsidian Canvas (``.canvas``) layout of the graph.

    Nodes are grouped by category on a rough grid: each category gets
    its own column, with notes stacked vertically. Uncategorised notes
    land in a trailing column. Suggested edges use a dashed style and
    contradictions are coloured red — same visual language as DOT.
    """
    column_w, column_gap = 260, 60
    row_h, row_gap = 80, 30

    by_category: dict[str, list[GraphNode]] = {}
    for n in graph.nodes:
        by_category.setdefault(n.category or "", []).append(n)

    # Stable column order: named categories alphabetically, uncategorised last.
    cat_keys = sorted(c for c in by_category if c) + ([""] if "" in by_category else [])

    positions: dict[int, tuple[int, int]] = {}
    canvas_nodes: list[dict] = []
    for col_idx, cat in enumerate(cat_keys):
        x = col_idx * (column_w + column_gap)
        for row_idx, node in enumerate(by_category[cat]):
            y = row_idx * (row_h + row_gap)
            positions[node.id] = (x, y)
            canvas_nodes.append({
                "id": f"n{node.id}",
                "type": "text",
                "x": x,
                "y": y,
                "width": column_w,
                "height": row_h,
                "text": f"# {node.title}\n{cat or 'uncategorised'}",
            })

    canvas_edges: list[dict] = []
    for i, e in enumerate(graph.edges):
        if e.src not in positions or e.dst not in positions:
            continue
        edge = {
            "id": f"e{i}",
            "fromNode": f"n{e.src}",
            "fromSide": "right",
            "toNode": f"n{e.dst}",
            "toSide": "left",
        }
        if e.kind == "suggested":
            edge["color"] = "4"   # Obsidian colour-4 ≈ blue
            edge["label"] = f"{e.weight:.2f}"
        elif e.kind == "contradicts":
            edge["color"] = "1"   # Obsidian colour-1 ≈ red
            edge["fromEnd"] = "arrow"
            edge["toEnd"] = "arrow"
        # wikilink → default styling.
        canvas_edges.append(edge)

    payload = {"nodes": canvas_nodes, "edges": canvas_edges}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ── format dispatch ──────────────────────────────────────────────────────


SUPPORTED_FORMATS = ("json", "dot", "obsidian-canvas")


def write_graph(graph: Graph, path: Path, fmt: str) -> Path:
    """Format-router used by ``_do_graph_export``. Raises ``ValueError``
    on an unknown format so the CLI surfaces a clean error instead of
    silently writing nothing."""
    fmt = fmt.lower()
    if fmt == "json":
        return write_json(graph, path)
    if fmt == "dot":
        return write_dot(graph, path)
    if fmt in ("obsidian-canvas", "canvas"):
        return write_canvas(graph, path)
    raise ValueError(
        f"Unknown format {fmt!r}. Supported: {', '.join(SUPPORTED_FORMATS)}."
    )
