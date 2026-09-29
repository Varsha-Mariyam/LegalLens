"""Phase 3 — document processing: PDF (PyMuPDF), DOCX (python-docx), TXT, and OCR for scanned PDF/images."""
import pytest

from backend.services import extraction_service, ocr_service
from backend.services.text_cleaning import clean_text

needs_ocr = pytest.mark.skipif(not ocr_service.status()["available"], reason="Tesseract not installed")


@pytest.mark.parametrize("ext,method", [("pdf", "text-layer"), ("docx", "python-docx"), ("txt", "plain-text")])
def test_extract_digital_formats(samples, ext, method):
    out = extraction_service.extract(str(samples / f"employment_agreement_sample.{ext}"))
    assert method in out["methods"]
    assert "EMPLOYMENT AGREEMENT" in out["text"].upper()
    assert "seven (7) days" in out["text"]


@needs_ocr
@pytest.mark.parametrize("name", ["rental_agreement_sample_scanned.png", "rental_agreement_sample_scanned.pdf"])
def test_ocr_scanned(samples, name):
    out = extraction_service.extract(str(samples / name))
    assert "ocr" in out["methods"]
    low = out["text"].lower()
    assert "security deposit" in low and "tenant" in low


def test_all_formats_give_same_text_content(samples):
    texts = [extraction_service.extract(str(samples / f"nda_sample.{e}"))["text"] for e in ("pdf", "docx", "txt")]
    words = [set(t.lower().split()) for t in texts]
    assert len(words[0] & words[1] & words[2]) / len(words[2]) > 0.9


def test_unsupported_type(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("hello")
    with pytest.raises(extraction_service.ExtractionError):
        extraction_service.extract(str(f))


def test_clean_text_normalises_whitespace_and_hyphenation():
    raw = "The employ-\nee shall   receive\u00a0salary.\n\n\n\nPage 1 of 3\n"
    cleaned = clean_text(raw)
    assert "employee" in cleaned
    assert "   " not in cleaned and "\u00a0" not in cleaned
