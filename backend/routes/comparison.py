"""/api/documents/{id}/comparison — clause-by-clause comparison with the project reference template."""
from fastapi import APIRouter, Depends

from backend.database.models import Document
from backend.routes.security import get_owned_document, require_completed
from backend.routes.serializers import comparison_out
from backend.services import comparison_service

router = APIRouter(prefix="/api", tags=["comparison"])

REFERENCE_LABEL = ("Project reference template created for LegalLens. It is NOT an official legal standard for any "
                   "jurisdiction; it is used only as a comparison baseline.")


@router.get("/documents/{document_id}/comparison")
def get_comparison(doc: Document = Depends(get_owned_document)):
    require_completed(doc)
    rows = [comparison_out(r) for r in doc.comparisons]
    summary = {s: sum(1 for r in rows if r["status"] == s) for s in ("equivalent", "differs", "missing", "additional")}
    return {"document_type": doc.detected_type, "reference": f"standard_{doc.detected_type}.txt",
            "reference_label": REFERENCE_LABEL, "summary": summary, "rows": rows}


@router.get("/standards/{doc_type}")
def get_standard(doc_type: str):
    return {"document_type": doc_type, "reference_label": REFERENCE_LABEL,
            "clauses": comparison_service.load_standard(doc_type)}
