"""/api/documents/{id}/qa — document Q&A with stored conversation history (RAG + CAG)."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Conversation, Document
from backend.routes.security import get_owned_document, require_completed
from backend.routes.serializers import conversation_out
from backend.services import pipeline, qa_service

router = APIRouter(prefix="/api/documents", tags=["qa"])


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@router.get("/{document_id}/qa")
def history(doc: Document = Depends(get_owned_document)):
    return [conversation_out(c) for c in doc.conversations]


@router.post("/{document_id}/qa")
def ask(body: QuestionIn, doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    require_completed(doc)
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question is empty")
    ctx = pipeline.get_context(doc)
    hist = [{"question": c.question, "answer": c.answer, "sources": c.sources or []} for c in doc.conversations]
    result = qa_service.answer(question, ctx, hist)
    row = Conversation(document_id=doc.document_id, question=question, answer=result["answer"],
                       sources=result["sources"], method=result["method"])
    db.add(row)
    db.commit()
    return conversation_out(row)


@router.delete("/{document_id}/qa", status_code=204)
def clear(doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    for c in list(doc.conversations):
        db.delete(c)
    db.commit()
