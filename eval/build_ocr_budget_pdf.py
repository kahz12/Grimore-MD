"""Repeat a controlled scanned page to measure per-document OCR budgets.

Every page deliberately has the same image and printed folio. This is a
synthetic work/coverage control, not a diverse corpus or a retrieval test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ROOT / "test_vault" / "test_vault" / "informe_humedal_escaneado.pdf"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "phase4_ocr_budget_humedal.pdf"


def build(output: Path) -> Path:
    """Copy the unchanged image into three physical pages; refuse overwrite."""
    reader, writer = PdfReader(SCANNED), PdfWriter()
    for _ in range(3):
        writer.add_page(reader.pages[0])
    writer.add_metadata({
        "/Title": "Control Grimore: presupuesto de OCR",
        "/Subject": "Synthetic three-page OCR budget control; repeated scan, not varied content",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        writer.write(stream)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
