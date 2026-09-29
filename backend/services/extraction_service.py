"""Upload -> identify file type -> extract text -> clean text (guide section 6)."""
from __future__ import annotations

from pathlib import Path

from backend.services import docx_service, ocr_service, pdf_service
from backend.services.text_cleaning import clean_text

SUPPORTED = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class ExtractionError(ValueError):
    pass


def extract(path: str) -> dict:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED:
        raise ExtractionError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}")
    try:
        if ext == ".pdf":
            pages = pdf_service.extract_pdf(path)
        elif ext == ".docx":
            pages = docx_service.extract_docx(path)
        elif ext == ".txt":
            raw = Path(path).read_bytes()
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    txt = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            pages = [{"page": 1, "text": txt, "method": "plain-text"}]
        else:
            pages = ocr_service.ocr_image_file(path)
    except ocr_service.OCRUnavailable as exc:
        raise ExtractionError(str(exc)) from exc
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not read the file: {exc}") from exc

    raw_text = "\n\n".join(p["text"] for p in pages)
    cleaned = clean_text(raw_text)
    if len(cleaned) < 20:
        raise ExtractionError("No readable text was found in the document.")
    methods = sorted({p["method"] for p in pages})
    return {"pages": len(pages), "methods": methods, "raw_text": raw_text, "text": cleaned}
