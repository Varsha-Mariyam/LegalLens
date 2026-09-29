"""Render the synthetic sample .txt documents into PDF, DOCX and a scanned image (to test all extractors)."""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import docx
import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from backend.config import settings

SAMPLES = settings.RAW_DIR / "samples"


def to_pdf(txt: Path) -> Path:
    out = txt.with_suffix(".pdf")
    styles = getSampleStyleSheet()
    story = []
    for block in txt.read_text(encoding="utf-8").split("\n\n"):
        for line in block.split("\n"):
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(out), pagesize=A4, title=txt.stem).build(story)
    return out


def to_docx(txt: Path) -> Path:
    out = txt.with_suffix(".docx")
    d = docx.Document()
    for block in txt.read_text(encoding="utf-8").split("\n\n"):
        for line in block.split("\n"):
            d.add_paragraph(line)
    d.save(str(out))
    return out


def to_scanned(pdf: Path) -> tuple[Path, Path]:
    """Rasterise page 1 (no text layer) to PNG and to an image-only PDF."""
    png = pdf.with_name(pdf.stem + "_scanned.png")
    img_pdf = pdf.with_name(pdf.stem + "_scanned.pdf")
    with pymupdf.open(pdf) as src:
        pix = src[0].get_pixmap(dpi=200)
        pix.save(str(png))
        out = pymupdf.open()
        page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        page.insert_image(page.rect, filename=str(png))
        out.save(str(img_pdf))
        out.close()
    return png, img_pdf


if __name__ == "__main__":
    for txt in sorted(SAMPLES.glob("*_sample.txt")):
        pdf = to_pdf(txt)
        to_docx(txt)
        print(f"rendered {txt.name} -> .pdf/.docx")
    png, img_pdf = to_scanned(SAMPLES / "rental_agreement_sample.pdf")
    print(f"scanned versions: {png.name}, {img_pdf.name}")
