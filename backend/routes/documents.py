"""/api/documents — upload, list, status, text, clauses, reprocess, delete."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db
from backend.database.models import Clause, Document, User
from backend.routes.security import get_current_user, get_owned_document
from backend.routes.serializers import clause_out, document_out
from backend.services import cache_service, pipeline
from backend.services.extraction_service import SUPPORTED

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _safe_name(name: str) -> str:
    base = Path(name or "document").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:120] or "document"


@router.post("", status_code=202)
async def upload(background: BackgroundTasks, file: UploadFile = File(...), document_type: str = Form("auto"),
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Supported: {', '.join(sorted(SUPPORTED))}")
    if document_type not in ["auto", *settings.DOCUMENT_TYPES]:
        raise HTTPException(422, f"document_type must be one of auto, {', '.join(settings.DOCUMENT_TYPES)}")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {settings.MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(422, "The uploaded file is empty")
    stored = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}_{_safe_name(file.filename)}"
    stored.write_bytes(data)
    doc = Document(user_id=user.user_id, filename=_safe_name(file.filename), stored_path=str(stored),
                   document_type=document_type, status="uploaded", current_step="Queued", progress=0)
    db.add(doc)
    db.commit()
    background.add_task(pipeline.process_document, doc.document_id)
    return document_out(doc, with_counts=False)


@router.get("")
def list_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.user_id == user.user_id).order_by(Document.upload_date.desc()).all()
    return [document_out(d) for d in docs]


@router.get("/{document_id}")
def get_document(doc: Document = Depends(get_owned_document)):
    return document_out(doc)


@router.get("/{document_id}/status")
def get_status(doc: Document = Depends(get_owned_document)):
    return {"document_id": doc.document_id, "status": doc.status, "current_step": doc.current_step,
            "progress": doc.progress, "error": doc.error, "steps": pipeline.STEPS}


@router.get("/{document_id}/text")
def get_text(doc: Document = Depends(get_owned_document)):
    return {"document_id": doc.document_id, "text": doc.full_text or "", "extraction_method": doc.extraction_method}


@router.get("/{document_id}/clauses")
def get_clauses(doc: Document = Depends(get_owned_document)):
    return [clause_out(c, full=False) for c in doc.clauses]


@router.get("/{document_id}/clauses/{clause_id}")
def get_clause(clause_id: int, doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    c = db.get(Clause, clause_id)
    if c is None or c.document_id != doc.document_id:
        raise HTTPException(404, "Clause not found")
    return clause_out(c)


@router.post("/{document_id}/reprocess", status_code=202)
def reprocess(background: BackgroundTasks, doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    if doc.status == "processing":
        raise HTTPException(409, "Document is already being processed")
    doc.status, doc.current_step, doc.progress, doc.error = "uploaded", "Queued", 0, None
    for p in list(doc.persona_summaries):
        db.delete(p)
    db.commit()
    cache_service.delete_document_context(doc.document_id)
    background.add_task(pipeline.process_document, doc.document_id)
    return document_out(doc)


@router.delete("/{document_id}", status_code=204)
def delete_document(doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    paths = [Path(doc.stored_path), settings.PROCESSED_DIR / f"upload_doc{doc.document_id}.txt",
             settings.CLAUSES_DIR / f"upload_doc{doc.document_id}.json",
             settings.REPORTS_DIR / f"LegalLens_Report_doc{doc.document_id}.pdf"]
    cache_service.delete_document_context(doc.document_id)
    db.delete(doc)
    db.commit()
    for p in paths:
        p.unlink(missing_ok=True)
