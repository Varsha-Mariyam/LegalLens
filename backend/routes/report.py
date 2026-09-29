"""/api/documents/{id}/report — generates and downloads the PDF report (ReportLab)."""
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from backend.database.models import Document
from backend.routes.security import get_owned_document, require_completed
from backend.services import report_service

router = APIRouter(prefix="/api/documents", tags=["report"])


@router.get("/{document_id}/report")
def download_report(doc: Document = Depends(get_owned_document)):
    require_completed(doc)
    path = report_service.build_report(doc, list(doc.clauses), list(doc.comparisons), list(doc.persona_summaries),
                                       list(doc.conversations))
    stem = doc.filename.rsplit(".", 1)[0]
    return FileResponse(path, media_type="application/pdf", filename=f"LegalLens_Report_{stem}.pdf")
