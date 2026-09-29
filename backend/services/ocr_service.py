"""OCR for scanned PDFs and images using Tesseract (guide section 6)."""
from __future__ import annotations

import io
import shutil

from backend.config import settings


class OCRUnavailable(RuntimeError):
    pass


def _pytesseract():
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise OCRUnavailable("pytesseract is not installed (pip install pytesseract)") from exc
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    elif not shutil.which("tesseract"):
        raise OCRUnavailable("Tesseract OCR engine not found. Install it (see README) or set TESSERACT_CMD.")
    return pytesseract


def status() -> dict:
    try:
        pt = _pytesseract()
        return {"available": True, "version": str(pt.get_tesseract_version())}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def ocr_image(image) -> str:
    """OCR a PIL image."""
    pt = _pytesseract()
    gray = image.convert("L")
    return pt.image_to_string(gray, lang=settings.OCR_LANG, config="--psm 6")


def ocr_image_file(path: str) -> list[dict]:
    from PIL import Image, ImageSequence
    pages = []
    with Image.open(path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            pages.append({"page": i + 1, "text": ocr_image(frame.copy()), "method": "ocr"})
    return pages


def ocr_pdf_page(page, dpi: int = 300) -> str:
    """Render a PyMuPDF page to an image and OCR it."""
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return ocr_image(img)
