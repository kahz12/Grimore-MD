"""Geometry variations, table conservation and safe layout fallback."""
from collections import Counter
from types import SimpleNamespace

import pytest

from grimore.ingest import pdf_layout as layout
from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.pdf import PdfAdapter
from grimore.ingest.extraction_profile import pdf_extraction_profile
from grimore.ingest.parser import extraction_diagnostics
from grimore.utils.config import Config


def word(text, x, y, width=35, height=10):
    return {"text": text, "x0": x, "x1": x + width, "top": y, "bottom": y + height,
            "upright": True, "direction": "ltr"}


def page(words, tables=(), edges=()):
    return SimpleNamespace(extract_words=lambda: words, find_tables=lambda _settings: tables,
                           edges=edges)


@pytest.mark.parametrize("columns", [2, 3])
@pytest.mark.parametrize("scale,offset", [(1, 0), (.5, 25), (1.5, -40)])
def test_whitespace_cuts_order_prose_columns_without_fixed_coordinates(columns, scale, offset):
    words = []
    for col in reversed(range(columns)):
        for row in range(4):
            for index in range(4):
                words.append(word(f"C{col}R{row}W{index}", 40 + col * 230 + index * 40,
                                  100 + row * 15))
    words += [word("Header", 40, 20, width=230 * columns - 50, height=16),
              word("Footer", 40, 700, width=180), word("Folio", 230 * columns - 50, 700)]
    for item in words:
        for key in ("x0", "x1", "top", "bottom"):
            item[key] = item[key] * scale + offset
    text = layout.extract_page(page(words))
    ordered = ["Header", *[f"C{col}R{row}W{index}" for col in range(columns)
                           for row in range(4) for index in range(4)], "Footer", "Folio"]
    assert text.split() == ordered
    assert Counter(text.split()) == Counter(item["text"] for item in words)


def test_short_unruled_keys_and_values_are_not_interpreted_as_prose_columns():
    words = [word(text, x, 50 + row * 15) for row in range(4)
             for text, x in [(f"Key{row}", 40), (f"Value{row}", 300)]]
    assert layout.extract_page(page(words)).splitlines() == [
        f"Key{row} Value{row}" for row in range(4)
    ]


def test_table_cell_wrapping_merged_header_empty_cells_and_duplicates_are_preserved():
    rows = [["Header", None], ["A", "two\nlines"], ["B", ""], ["A", "two\nlines"]]
    words = [word("Header", 10, 60), word("A", 10, 80), word("two", 250, 80),
             word("lines", 250, 90), word("B", 10, 110), word("A", 10, 130),
             word("two", 250, 130), word("lines", 250, 140)]
    table = SimpleNamespace(bbox=(0, 50, 500, 160), cells=[None] * 8, extract=lambda: rows)
    assert layout.extract_page(page(words, [table])).splitlines() == [
        "Header\t", "A\ttwo lines", "B\t", "A\ttwo lines",
    ]


def test_final_empty_columns_keep_their_positions_instead_of_being_trimmed():
    words = [word("Header", 10, 60), word("A", 10, 90)]
    table = SimpleNamespace(bbox=(0, 50, 500, 120), cells=[None] * 6,
                            extract=lambda: [["Header", None, None], ["A", "", None]])
    assert layout.extract_page(page(words, [table])) == "Header\t\t\nA\t\t"


@pytest.mark.parametrize("rows", [
    [["A", "wrong"], ["B", "two lines"]],
    [["A", "two lines"], ["B", "two lines two lines"]],
])
def test_table_replacement_must_conserve_every_word_including_multiplicity(rows):
    words = [word("A", 10, 60), word("two", 250, 60), word("lines", 250, 70),
             word("B", 10, 90), word("two", 250, 90), word("lines", 250, 100)]
    table = SimpleNamespace(bbox=(0, 50, 500, 120), cells=[None] * 4, extract=lambda: rows)
    with pytest.raises(layout.LayoutUnavailable, match="layout_table_ambiguous"):
        layout.extract_page(page(words, [table]))


def test_decorative_frame_is_not_a_table_and_empty_scan_has_no_layout_text():
    words = [word("caption", 10, 60)]
    frame = SimpleNamespace(bbox=(0, 50, 500, 120), cells=[None], extract=lambda: [["caption"]])
    assert layout.extract_page(page(words, [frame])) == "caption"
    assert layout.extract_page(page([])) == ""


def test_comparison_to_baseline_prevents_dropping_a_word_before_layout_detection():
    with pytest.raises(layout.LayoutUnavailable, match="layout_text_mismatch"):
        layout.extract_page(page([word("visible", 10, 60)]), baseline_text="visible lost")


@pytest.mark.parametrize("change", [
    {"upright": False}, {"direction": "rtl"}, {"x1": float("nan")}, {"bottom": 0},
])
def test_unsupported_geometry_does_not_get_reconstructed(change):
    item = word("word", 10, 60)
    item.update(change)
    with pytest.raises(layout.LayoutUnavailable, match="layout_unsupported"):
        layout.extract_page(page([item]))


@pytest.mark.parametrize("limit", ["MAX_WORDS", "MAX_EDGES", "MAX_TABLES", "MAX_CELLS", "MAX_DEPTH"])
def test_layout_limits_are_explicit_and_bounded(monkeypatch, limit):
    words = [word("A", 10, 60), word("B", 250, 60)]
    table = SimpleNamespace(bbox=(0, 50, 500, 120), cells=[None] * 4,
                            extract=lambda: [["A", "B"], ["", ""]])
    monkeypatch.setattr(layout, limit, 0)
    with pytest.raises(layout.LayoutUnavailable, match="layout_limit"):
        layout.extract_page(page(words, [table], [object()]))


@pytest.mark.parametrize("failure", ["unexpected", "limit", "empty"])
def test_adapter_retains_baseline_text_and_reports_fallback(tmp_path, monkeypatch, failure):
    pdfplumber = pytest.importorskip("pdfplumber")
    fake_page = page([word("Baseline", 10, 60), word("text", 60, 60)])
    fake_page.extract_text = lambda: "Baseline text"

    class Document:
        metadata = {"Title": "Fallback"}
        pages = [fake_page]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda _path: Document())

    def fail(*_args, **_kwargs):
        if failure == "empty":
            return ""
        if failure == "limit":
            raise layout.LayoutUnavailable("layout_limit")
        raise RuntimeError("private internal details")

    monkeypatch.setattr(layout, "extract_page", fail)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"placeholder")
    result = PdfAdapter(engine="pdfplumber", ocr=True).extract(
        source, options=AdapterOptions(vault_root=tmp_path),
    )
    assert result.text == "Baseline text"
    assert result.sections[0].page == 1
    diagnostics = extraction_diagnostics(result)
    assert diagnostics["status"] == "recovered"
    assert diagnostics["issues"][0] == {
        "code": "page_layout_fallback", "page": 1, "engine": "pdfplumber",
        "recovered": True, "ocr_requested": True,
        "reason": {"unexpected": "layout_failed", "limit": "layout_limit",
                   "empty": "layout_text_mismatch"}[failure],
    }
    assert "private internal details" not in str(diagnostics)


def test_layout_revision_invalidates_only_pdfplumber_profiles():
    config = Config()
    assert pdf_extraction_profile(config) == "pdf-layout-v1:engine=pypdf:ocr=0"
    config.ingest.pdf_engine = "pdfplumber"
    assert pdf_extraction_profile(config) == "pdf-layout-v2:engine=pdfplumber:ocr=0"
    config.ingest.pdf_engine = "pymupdf"
    assert pdf_extraction_profile(config) == "pdf-layout-v1:engine=pymupdf:ocr=0"
