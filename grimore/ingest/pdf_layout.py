"""Conservative geometric reading order for the optional pdfplumber engine.

Ruled tables are atomic blocks; newlines inside a cell become spaces while
column positions (including empty cells) are retained with tabs. Recursive
whitespace cuts separate prose columns, headers and footers without using
document-specific coordinates or vocabulary. Ambiguous/unsupported geometry
falls back to the engine's original extraction, never to guessed cell text.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import median

MAX_WORDS = 20_000
MAX_EDGES = 2_000
MAX_TABLES = 50
MAX_CELLS = 10_000
MAX_DEPTH = 16
LINE_TOLERANCE = 3.0


class LayoutUnavailable(ValueError):
    """Stable diagnostic, without exposing library exception messages."""


@dataclass(frozen=True)
class Block:
    x0: float
    top: float
    x1: float
    bottom: float
    text: str
    table: bool = False


def _inside(block: Block, bbox: tuple) -> bool:
    x, y = (block.x0 + block.x1) / 2, (block.top + block.bottom) / 2
    return bbox[0] <= x < bbox[2] and bbox[1] <= y < bbox[3]


def _tokens(text: str) -> Counter:
    return Counter(text.split())


def _lines(blocks: list[Block]) -> list[list[Block]]:
    lines: list[list[Block]] = []
    for block in sorted(blocks, key=lambda b: (b.top, b.x0)):
        if (lines and not block.table and not lines[-1][0].table
                and abs(block.top - lines[-1][0].top) <= LINE_TOLERANCE):
            lines[-1].append(block)
        else:
            lines.append([block])
    return [sorted(line, key=lambda b: b.x0) for line in lines]


def _linear(blocks: list[Block]) -> str:
    return "\n".join(" ".join(b.text for b in line) for line in _lines(blocks))


def _gap(blocks: list[Block], *, vertical: bool) -> tuple[float, float]:
    intervals = sorted((b.x0, b.x1) if vertical else (b.top, b.bottom) for b in blocks)
    end = intervals[0][1]
    best = (0.0, 0.0)
    for start, stop in intervals[1:]:
        if start > end and start - end > best[0]:
            best = (start - end, (end + start) / 2)
        end = max(end, stop)
    return best


def _prose_column(blocks: list[Block]) -> bool:
    if any(b.table for b in blocks):
        return True
    # Short aligned keys/values or a single spaced-out heading aren't prose
    # columns. Unruled complex tables remain inherently ambiguous.
    lines = _lines(blocks)
    return len(lines) >= 3 and sum(len(line) >= 4 for line in lines) >= 2


def reading_order(blocks: list[Block], *, depth: int = 0) -> str:
    if not blocks:
        return ""
    if depth >= MAX_DEPTH:
        raise LayoutUnavailable("layout_limit")
    height = median(b.bottom - b.top for b in blocks if not b.table) if any(
        not b.table for b in blocks
    ) else 10.0
    x_gap, x_cut = _gap(blocks, vertical=True)
    y_gap, y_cut = _gap(blocks, vertical=False)
    left = [b for b in blocks if b.x1 < x_cut]
    right = [b for b in blocks if b.x0 > x_cut]
    split_columns = (x_gap >= max(18.0, height * 2) and left and right
                     and _prose_column(left) and _prose_column(right))
    if split_columns and x_gap > y_gap:
        parts = (left, right)
    elif y_gap >= max(8.0, height):
        parts = ([b for b in blocks if b.bottom < y_cut],
                 [b for b in blocks if b.top > y_cut])
    else:
        return _linear(blocks)
    return "\n\n".join(reading_order(part, depth=depth + 1) for part in parts if part)


def extract_page(page, *, baseline_text: str | None = None) -> str:
    """Return geometry-aware text, or raise so the caller can retain baseline.

    Word-token conservation is checked with multiplicity, not a set: repeated
    values must neither disappear nor be duplicated during table replacement.
    """
    words = page.extract_words()
    if len(words) > MAX_WORDS or len(page.edges) > MAX_EDGES:
        raise LayoutUnavailable("layout_limit")
    blocks = []
    for word in words:
        coords = [word[key] for key in ("x0", "top", "x1", "bottom")]
        if (not word.get("upright", True) or word.get("direction", "ltr") != "ltr"
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords)
                or coords[2] <= coords[0] or coords[3] <= coords[1]):
            raise LayoutUnavailable("layout_unsupported")
        blocks.append(Block(*coords, word["text"]))
    if not blocks:
        return ""
    original = _tokens(" ".join(b.text for b in blocks))
    if baseline_text is not None and original != _tokens(baseline_text):
        raise LayoutUnavailable("layout_text_mismatch")
    tables = page.find_tables({"vertical_strategy": "lines_strict",
                               "horizontal_strategy": "lines_strict"})
    if len(tables) > MAX_TABLES or sum(len(t.cells) for t in tables) > MAX_CELLS:
        raise LayoutUnavailable("layout_limit")
    for table in tables:
        rows = table.extract()
        # A frame or decorative line isn't enough to claim a table.
        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            continue
        text = "\n".join("\t".join(" ".join((cell or "").split()) for cell in row)
                         for row in rows)
        covered = [b for b in blocks if _inside(b, table.bbox)]
        if any(b.table for b in covered) or _tokens(text) != _tokens(" ".join(b.text for b in covered)):
            raise LayoutUnavailable("layout_table_ambiguous")
        if not text.strip():
            continue
        blocks = [b for b in blocks if not _inside(b, table.bbox)]
        blocks.append(Block(*table.bbox, text, table=True))
    # A trailing tab still identifies an empty last column of a final row.
    text = reading_order(blocks).strip("\n")
    if _tokens(text) != original:
        raise LayoutUnavailable("layout_text_mismatch")
    return text
