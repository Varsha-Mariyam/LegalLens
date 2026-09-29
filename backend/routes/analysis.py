"""/api/documents/{id}/analysis and persona summaries."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db
from backend.database.models import Document, PersonaSummary
from backend.routes.security import get_owned_document, require_completed
from backend.routes.serializers import RISK_ORDER, clause_out, comparison_out, document_out, persona_out
from backend.services import persona_service, pipeline, rag_service

router = APIRouter(prefix="/api/documents", tags=["analysis"])


@router.get("/{document_id}/analysis")
def get_analysis(doc: Document = Depends(get_owned_document)):
    require_completed(doc)
    clauses = [clause_out(c) for c in doc.clauses]
    cats = {}
    for c in clauses:
        cats[c["clause_type"]] = cats.get(c["clause_type"], 0) + 1
    return {"document": document_out(doc), "overall_summary": doc.overall_summary,
            "summary_method": doc.summary_method, "recommendations": doc.recommendations or [],
            "analysis_meta": doc.analysis_meta or {}, "category_counts": cats, "clauses": clauses,
            "comparisons": [comparison_out(r) for r in doc.comparisons],
            "highest_risk": sorted(clauses, key=lambda c: (RISK_ORDER.get(c["risk_level"], 3), c["clause_number"]))[:5]}


class PersonaIn(BaseModel):
    persona: str


@router.get("/{document_id}/persona")
def list_personas(doc: Document = Depends(get_owned_document)):
    return {"available": [{"key": k, "label": v["label"]} for k, v in persona_service.PERSONAS.items()],
            "summaries": [persona_out(p) for p in doc.persona_summaries]}


@router.post("/{document_id}/persona")
def generate_persona(body: PersonaIn, doc: Document = Depends(get_owned_document), db: Session = Depends(get_db)):
    require_completed(doc)
    persona = body.persona.strip().lower()
    if persona not in settings.PERSONAS:
        raise HTTPException(422, f"persona must be one of {', '.join(settings.PERSONAS)}")
    ctx = pipeline.get_context(doc)  # CAG: cached document context
    focus = " ".join(persona_service.PERSONAS[persona]["focus"])
    refs = rag_service.retrieve_knowledge(f"{doc.detected_type} agreement {focus} rights obligations", doc.detected_type, k=3)
    text, method = persona_service.generate(persona, doc.detected_type, ctx["clauses"], refs)
    existing = next((p for p in doc.persona_summaries if p.persona == persona), None)
    if existing:
        existing.summary, existing.method = text, method
        row = existing
    else:
        row = PersonaSummary(document_id=doc.document_id, persona=persona, summary=text, method=method)
        db.add(row)
    db.commit()
    return persona_out(row)
