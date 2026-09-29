"""PDF text extraction with PyMuPDF; pages without a text layer fall back to OCR."""
from __future__ import annotations

import pymupdf as fitz  # PyMuPDF

from backend.services import ocr_service

MIN_CHARS_FOR_TEXT_LAYER = 40


def extract_pdf(path: str) -> list[dict]:
    pages = []
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("The PDF is password protected. Remove the password and upload again.")
        for i, page in enumerate(doc):
            text = page.get_text("text")
            method = "text-layer"
            if len(text.strip()) < MIN_CHARS_FOR_TEXT_LAYER:
                try:
                    text = ocr_service.ocr_pdf_page(page)
                    method = "ocr"
                except ocr_service.OCRUnavailable as exc:
                    if not text.strip():
                        raise ValueError(f"Page {i + 1} is scanned but OCR is unavailable: {exc}") from exc
            pages.append({"page": i + 1, "text": text, "method": method})
    return pages
