"""Deterministic synthetic vault generator for the benchmark harness.

The whole point of this file is reproducibility: a baseline is only useful if
the "before" and "after" runs ingest byte-identical input. Everything here is
derived from a single integer seed, so `make_vault.py --notes N --seed S`
always produces the same tree on any machine and any Python build.

Why generated rather than a checked-in fixture vault: the note count has to
scale (a 2000-note vault is what makes `connect`'s O(notes x N) behaviour
visible, but a 200-note one is what keeps iteration fast), and committing
thousands of prose files would bloat the repo for no gain.

Sizing note: notes target roughly 6 KB of body text so that the default
markdown chunker (chunk_max_chars = 1500) yields several chunks each. Chunk
COUNT is what drives the embedding and persistence load the harness measures,
so a vault of tiny notes would understate exactly the cost we care about.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# Fixed vocabulary. Drawn from infrastructure/knowledge-management prose so the
# generated text is plausible for an embedding model, but the specific words do
# not matter -- only that the pool is a constant, so a given seed reproduces a
# given vault forever.
_WORDS = [
    "retention", "policy", "cluster", "latency", "throughput", "index",
    "vector", "embedding", "snapshot", "rollback", "quorum", "replica",
    "partition", "consistency", "durability", "checkpoint", "journal",
    "compaction", "ingestion", "pipeline", "adapter", "schema", "migration",
    "threshold", "cardinality", "namespace", "scheduler", "backpressure",
    "idempotent", "deterministic", "observability", "telemetry", "tracing",
    "quantization", "recall", "precision", "corpus", "chunk", "anchor",
    "citation", "provenance", "taxonomy", "ontology", "heuristic", "gradient",
    "checksum", "manifest", "artifact", "lease", "tombstone", "watermark",
]

# Directory layout. Categories exist so retrieval filters and the category
# tree have something non-trivial to resolve against.
_CATEGORIES = ["infra", "theory", "ops", "research"]

_TOPICS = [
    "Retention windows", "Replica placement", "Vector recall", "Index rebuilds",
    "Backpressure limits", "Snapshot cadence", "Chunk anchoring", "Query fusion",
    "Schema migration", "Failure domains", "Cache invalidation", "Batch sizing",
]


def _sentence(rng: random.Random) -> str:
    """One sentence of 8-18 words from the fixed pool."""
    n = rng.randint(8, 18)
    words = [rng.choice(_WORDS) for _ in range(n)]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _paragraph(rng: random.Random) -> str:
    return " ".join(_sentence(rng) for _ in range(rng.randint(4, 8)))


def _note_body(rng: random.Random, index: int, total: int) -> str:
    """Build one note: title, several ## sections, and a few wikilinks.

    Headings matter beyond cosmetics: the chunker propagates them as anchors,
    so a vault of heading-less notes would leave the anchor path in
    `chunk_sections` untested by the benchmark.
    """
    topic = _TOPICS[index % len(_TOPICS)]
    lines = [f"# {topic} {index:05d}", ""]

    for section in range(rng.randint(3, 5)):
        lines.append(f"## {rng.choice(_TOPICS)} {section}")
        lines.append("")
        for _ in range(rng.randint(2, 4)):
            lines.append(_paragraph(rng))
            lines.append("")

    # Cross-links to other notes in the vault. Deterministic targets, and
    # always to a note that exists, so `connect` and the graph walker have a
    # real link structure rather than dangling references.
    lines.append("## Related")
    lines.append("")
    for _ in range(rng.randint(2, 4)):
        target = rng.randrange(total)
        lines.append(f"- [[{_TOPICS[target % len(_TOPICS)]} {target:05d}]]")
    lines.append("")

    return "\n".join(lines)


def build(out: Path, notes: int, seed: int) -> dict:
    """Regenerate the vault from scratch. Returns a small stats dict.

    The tree is wiped first on purpose. `scan` writes frontmatter back into
    .md files in place, so a vault reused across runs would arrive at the
    second run already tagged and fast-skip -- measuring nothing. Cold-scan
    numbers are only comparable against a freshly generated tree.
    """
    if out.exists():
        shutil.rmtree(out)

    total_bytes = 0
    for i in range(notes):
        # Per-note RNG seeded from (seed, i) so a note's content depends only
        # on its index -- generating 2000 notes and then 200 gives the same
        # first 200 files.
        rng = random.Random(f"{seed}:{i}")
        category = _CATEGORIES[i % len(_CATEGORIES)]
        directory = out / category
        directory.mkdir(parents=True, exist_ok=True)
        body = _note_body(rng, i, notes)
        path = directory / f"note-{i:05d}.md"
        path.write_text(body, encoding="utf-8")
        total_bytes += len(body.encode("utf-8"))

    return {
        "notes": notes,
        "seed": seed,
        "total_bytes": total_bytes,
        "avg_bytes": total_bytes // max(notes, 1),
        "path": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "vault")
    parser.add_argument("--notes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = build(args.out, args.notes, args.seed)
    print(
        f"vault: {stats['path']}  notes={stats['notes']}  "
        f"seed={stats['seed']}  avg={stats['avg_bytes']}B  "
        f"total={stats['total_bytes'] / 1e6:.1f}MB"
    )


if __name__ == "__main__":
    main()
