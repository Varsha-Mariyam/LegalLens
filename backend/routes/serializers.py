"""Converts ORM rows to JSON-friendly dicts for the API."""
from backend.database.models import Clause, ComparisonRow, Conversation, Document, PersonaSummary

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _iso(dt):
    return dt.isoformat() if dt else None


def document_out(d: Document, with_counts: bool = True) -> dict:
    out = {"document_id": d.document_id, "filename": d.filename, "upload_date": _iso(d.upload_date),
           "document_type": d.document_type, "detected_type": d.detected_type, "status": d.status,
           "current_step": d.current_step, "progress": d.progress, "error": d.error,
           "extraction_method": d.extraction_method, "page_count": d.page_count, "char_count": d.char_count,
           "processed_at": _iso(d.processed_at)}
    if with_counts:
        risks = [c.risk_level for c in d.clauses]
        out["clause_count"] = len(risks)
        out["risk_counts"] = {lvl: risks.count(lvl) for lvl in ("HIGH", "MEDIUM", "LOW")}
    return out


def clause_out(c: Clause, full: bool = True) -> dict:
    out = {"clause_id": c.clause_id, "clause_number": c.clause_number, "label": c.label, "title": c.title,
           "clause_type": c.clause_type, "category_confidence": c.category_confidence, "risk_level": c.risk_level}
    if full:
        out.update({"original_text": c.original_text, "summary": c.summary, "summary_method": c.summary_method,
                    "risk_reason": c.risk_reason, "precaution": c.precaution, "risk_evidence": c.risk_evidence,
                    "risk_method": c.risk_method, "classifier_details": c.classifier_details,
                    "references": c.references, "standard_match": c.standard_match})
    else:
        out["preview"] = (c.original_text or "")[:220]
    return out


def comparison_out(r: ComparisonRow) -> dict:
    return {k: getattr(r, k) for k in ("id", "position", "category", "status", "standard_title", "standard_text",
                                        "user_clause_number", "user_clause_label", "user_title", "user_text", "similarity",
                                        "standard_value", "user_value", "difference", "method")}


def conversation_out(c: Conversation) -> dict:
    return {"conversation_id": c.conversation_id, "question": c.question, "answer": c.answer,
            "sources": c.sources, "method": c.method, "timestamp": _iso(c.timestamp)}


def persona_out(p: PersonaSummary) -> dict:
    return {"id": p.id, "persona": p.persona, "summary": p.summary, "method": p.method,
            "created_at": _iso(p.created_at)}
