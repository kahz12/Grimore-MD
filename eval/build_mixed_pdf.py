"""Build a mixed text/scan PDF without touching the reference corpus in test_vault/.

Physical pages are: original digital page 2, scanned page 1, digital page 3.
Printed folios deliberately differ from PDF page ordinals on the first two
pages, so the fixture checks physical page anchors rather than footer text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "phase4_mixed_humedal.pdf"
DIGITAL = ROOT / "test_vault" / "test_vault" / "informe_humedal_la_esperanza.pdf"
SCANNED = ROOT / "test_vault" / "test_vault" / "informe_humedal_escaneado.pdf"


def build(output: Path) -> Path:
    """Create a deterministic fixture; never overwrite an existing file."""
    digital, scanned = PdfReader(DIGITAL), PdfReader(SCANNED)
    writer = PdfWriter()
    writer.add_page(digital.pages[1])
    writer.add_page(scanned.pages[0])
    writer.add_page(digital.pages[2])
    writer.add_metadata({
        "/Title": "Control mixto Grimore: texto y escaneo",
        "/Subject": "Synthetic control; PDF page ordinals differ from printed folios",
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
