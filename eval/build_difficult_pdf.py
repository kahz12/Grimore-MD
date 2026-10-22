"""Build layout/scan stress controls derived only from test_vault content.

Optional authoring dependencies: reportlab, pypdf and Pillow. Production
extraction and the published fixture do not require reportlab.
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "phase4_difficult_humedal.pdf"
SCANNED = ROOT / "test_vault" / "test_vault" / "informe_humedal_escaneado.pdf"


def build(output: Path, font_dir: Path = Path("/usr/share/fonts/truetype/liberation")) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    from PIL import ImageEnhance, ImageFilter
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle

    # Embed fonts so visual inspection does not depend on viewer substitution.
    pdfmetrics.registerFont(TTFont("ControlSans", str(font_dir / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("ControlSans-Bold", str(font_dir / "LiberationSans-Bold.ttf")))
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=A4, invariant=1)
    canvas.setTitle("Control de extracción compleja del Humedal")
    canvas.setSubject("Synthetic variants of test_vault: columns, wrapped table, degraded scan")
    width, height = A4
    normal = ParagraphStyle("Body", fontName="ControlSans", fontSize=10, leading=15)
    heading = ParagraphStyle("Heading", fontName="ControlSans-Bold", fontSize=13, leading=18)

    def header(title):
        canvas.setFillColor(colors.HexColor("#155B39"))
        canvas.setFont("ControlSans-Bold", 18)
        canvas.drawString(42, height - 58, title)
        canvas.setFillColor(colors.black)

    def footer(page):
        canvas.setFont("ControlSans", 8)
        canvas.drawString(42, 30, "Control sintético derivado de test_vault - no es muestra del piloto")
        canvas.drawRightString(width - 42, 30, f"Página física {page}")

    def column(x, entries):
        top = height - 115
        for text, style in entries:
            paragraph = Paragraph(text, style)
            _, size = paragraph.wrap(235, 500)
            paragraph.drawOn(canvas, x, top - size)
            top -= size + 14

    header("Diseño de seguimiento - dos columnas")
    # Draw the right column first to make content-stream order different from
    # the intended left-column-then-right-column visual reading order.
    column(318, [
        ("Restauración vegetal", heading),
        ("La revegetación usa tres especies nativas: junco de estera, botoncillo y eneas locales.", normal),
        ("La densidad inicial es de cuatro plantas por metro cuadrado en las franjas ribereñas.", normal),
        ("No se plantará carrizo gigante porque está clasificado como invasor en el plan de manejo.", normal),
        ("Control del jacinto de agua", heading),
        ("No se aplican herbicidas dentro del espejo de agua.", normal),
    ])
    column(42, [
        ("Estaciones de muestreo", heading),
        ("Norte-01 funciona como punto de referencia; Sur-02 permite observar cambios asociados a la intervención.", normal),
        ("La calidad del agua se mide el primer martes de cada mes.", normal),
        ("Si una tormenta supera 30 mm de lluvia en 24 horas, se añade una muestra extraordinaria entre 24 y 48 horas después del evento.", normal),
    ])
    footer(1)
    canvas.showPage()

    header("Responsabilidades - tabla multilínea")
    cells = [
        ["Responsabilidades", ""],
        ["Responsable", "Función"],
        ["Equipo Agua Clara", Paragraph("Muestreo de agua y<br/>custodia de resultados", normal)],
        ["Cooperativa Verde Vivo", Paragraph("Retiro de jacinto<br/>y revegetación", normal)],
        ["Comité Comunitario", Paragraph("Conteos participativos<br/>de aves y anfibios", normal)],
    ]
    table = Table(cells, colWidths=[175, 335], rowHeights=[28, 28, 54, 54, 54])
    table.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#DFEEE4")),
        ("FONTNAME", (0, 0), (-1, -1), "ControlSans"),
        ("FONTNAME", (0, 0), (-1, 1), "ControlSans-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#80968A")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    _, table_height = table.wrap(510, 500)
    table.drawOn(canvas, 42, height - 115 - table_height)
    footer(2)
    canvas.showPage()

    # Reuse the existing raster image, not a new invented document. These
    # deterministic degradations are part of the fixture, not OCR cleanup.
    image = PdfReader(SCANNED).pages[0].images[0].image.convert("L")
    image = image.resize((1000, round(image.height * 1000 / image.width)))
    image = ImageEnhance.Contrast(image).enhance(0.75)
    image = image.filter(ImageFilter.GaussianBlur(0.35))
    jpeg = BytesIO()
    image.save(jpeg, format="JPEG", quality=55, optimize=False)
    jpeg.seek(0)
    canvas.drawImage(ImageReader(jpeg), 0, 0, width=width, height=height)
    canvas.showPage()
    canvas.save()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(buffer.getvalue())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/liberation"))
    args = parser.parse_args()
    print(build(args.output, args.font_dir))


if __name__ == "__main__":
    main()
