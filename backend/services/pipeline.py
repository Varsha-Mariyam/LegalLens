"""End-to-end document analysis pipeline (guide sections 21-23)."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone

import numpy as np

from backend.config import settings
from backend.database.db import SessionLocal
from backend.database.models import Clause, ComparisonRow, Document
from backend.services import (cache_service, classifier, clause_service, comparison_service, document_type,
                              embedding_service, extraction_service, rag_service, risk_analyzer, summarizer)
from backend.services import legal_parsing as lp
from backend.services.llm_service import get_client

log = logging.getLogger("legallens.pipeline")

STEPS = ["Extracting text", "Segmenting clauses", "Classifying clauses", "Caching document context (CAG)",
         "Explaining clauses with RAG", "Comparing with reference", "Summarising", "Completed"]


def _progress(db, doc: Document, step: str, pct: int):
    doc.current_step, doc.progress = step, pct
    db.commit()


def document_facts(doc_type: str, clauses: list[dict]) -> dict:
    facts = {}
    if doc_type == "rental":
        for c in clauses:
            for s in lp.sentences(c["text"]):
                low = s.lower()
                if "rent" in low and ("month" in low) and "deposit" not in low:
                    amounts = lp.extract_amounts(s)
                    if amounts:
                        facts["monthly_rent"] = amounts[0]["value"]
                        return facts
    return facts


def build_recommendations(clauses: list[dict], comparisons: list[dict]) -> list[dict]:
    recs = []
    for level in ("HIGH", "MEDIUM"):
        for c in clauses:
            if c.get("risk_level") == level and c.get("precaution"):
                recs.append({"priority": level, "clause_number": c["number"],
                             "text": f"Clause {c.get('label') or c['number']} ({c.get('title') or c['category']}): {c['precaution']}"})
    for r in comparisons:
        if r["status"] == "missing":
            recs.append({"priority": "MEDIUM", "clause_number": None,
                         "text": f"Consider adding a {r['category'].lower()} clause similar to the reference "
                                 f"'{r['standard_title'] or r['category']}' clause, which this document does not contain."})
    recs.append({"priority": "INFO", "clause_number": None,
                 "text": "Have a qualified lawyer review the document before signing; this analysis is automated and educational."})
    return recs


def context_from_db(doc: Document) -> dict:
    """Rebuild the CAG context from the database (used on a cache miss)."""
    clauses = [{"number": c.clause_number, "label": c.label, "title": c.title, "text": c.original_text,
                "category": c.clause_type, "summary": c.summary, "risk_level": c.risk_level,
                "risk_reason": c.risk_reason, "precaution": c.precaution} for c in doc.clauses]
    emb = embedding_service.encode([f"{c['title'] or ''} {c['text']}" for c in clauses]) if clauses else np.zeros((0, 1))
    ctx = {"document_id": doc.document_id, "filename": doc.filename, "doc_type": doc.detected_type,
           "full_text": doc.full_text, "clauses": clauses, "clause_embeddings": emb.tolist(),
           "summary": doc.overall_summary, "embedder": embedding_service.get_embedder().name}
    cache_service.set_document_context(doc.document_id, ctx)
    return ctx


def get_context(doc: Document) -> dict:
    ctx = cache_service.get_document_context(doc.document_id)
    if ctx is None or ctx.get("embedder") != embedding_service.get_embedder().name:
        ctx = context_from_db(doc)
    return ctx


def process_document(document_id: int) -> None:
    db = SessionLocal()
    doc = db.get(Document, document_id)
    if doc is None:
        db.close()
        return
    try:
        doc.status, doc.error = "processing", None
        for old in list(doc.clauses) + list(doc.comparisons):
            db.delete(old)
        db.commit()

        _progress(db, doc, STEPS[0], 5)
        extracted = extraction_service.extract(doc.stored_path)
        doc.full_text, doc.page_count, doc.char_count = extracted["text"], extracted["pages"], len(extracted["text"])
        doc.extraction_method = ", ".join(extracted["methods"])
        (settings.PROCESSED_DIR / f"upload_doc{doc.document_id}.txt").write_text(extracted["text"], encoding="utf-8")
        detected, _ = document_type.detect(extracted["text"])
        doc.detected_type = doc.document_type if doc.document_type in settings.DOCUMENT_TYPES else detected

        _progress(db, doc, STEPS[1], 15)
        segs = clause_service.segment(extracted["text"])
        (settings.CLAUSES_DIR / f"upload_doc{doc.document_id}.json").write_text(
            json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")

        _progress(db, doc, STEPS[2], 25)
        cats = classifier.classify_batch(segs)
        clauses = []
        for s, c in zip(segs, cats):
            clauses.append({"number": s["id"], "label": s["label"], "title": s["title"], "text": s["text"],
                            "category": c["category"], "confidence": c["confidence"],
                            "classifier": {"method": c["method"], **c["details"]}})

        _progress(db, doc, STEPS[3], 35)
        emb = embedding_service.encode([f"{c['title'] or ''} {c['text']}" for c in clauses])
        ctx = {"document_id": doc.document_id, "filename": doc.filename, "doc_type": doc.detected_type,
               "full_text": doc.full_text, "clauses": [{k: c[k] for k in ("number", "label", "title", "text", "category")} for c in clauses],
               "clause_embeddings": emb.tolist(), "summary": None, "embedder": embedding_service.get_embedder().name}
        cache_service.set_document_context(doc.document_id, ctx)

        facts = document_facts(doc.detected_type, clauses)
        rag_service.ensure_index()
        n = max(1, len(clauses))
        for i, c in enumerate(clauses):
            _progress(db, doc, f"{STEPS[4]} ({i + 1}/{n})", 40 + int(40 * i / n))
            query = f"{c['title'] or ''} {c['text']}"
            refs = rag_service.retrieve_knowledge(query, doc.detected_type, c["category"], k=3)
            std = rag_service.retrieve_standard(query, doc.detected_type, c["category"], k=1)
            standard = std[0] if std and std[0]["score"] > 0.15 else None
            c["references"] = [{"title": r["metadata"].get("title"), "source": r["metadata"].get("source"),
                                "citation": r["metadata"].get("citation"), "score": r.get("adjusted_score", r["score"]),
                                "text": r["text"]} for r in refs]
            c["standard"] = ({"title": standard["metadata"].get("title"), "text": standard["text"], "score": standard["score"],
                              "label": "Project reference template (not an official legal standard)"} if standard else None)
            c["summary"], c["summary_method"] = summarizer.explain_clause(c, refs)
            risk = risk_analyzer.analyze(c, doc.detected_type, standard, refs, facts)
            c.update({"risk_level": risk["risk_level"], "risk_reason": risk["risk_reason"], "precaution": risk["precaution"],
                      "risk_evidence": risk["evidence"], "risk_method": risk["method"]})

        _progress(db, doc, STEPS[5], 82)
        comparisons = comparison_service.compare(doc.detected_type, clauses, emb)

        _progress(db, doc, STEPS[6], 92)
        doc.overall_summary, doc.summary_method = summarizer.overall_summary(doc.detected_type, clauses, doc.full_text)
        doc.recommendations = build_recommendations(clauses, comparisons)

        for c in clauses:
            db.add(Clause(document_id=doc.document_id, clause_number=c["number"], label=c["label"], title=c["title"],
                          clause_type=c["category"], category_confidence=c["confidence"], classifier_details=c["classifier"],
                          original_text=c["text"], summary=c["summary"], summary_method=c["summary_method"],
                          risk_level=c["risk_level"], risk_reason=c["risk_reason"], precaution=c["precaution"],
                          risk_evidence=c["risk_evidence"], risk_method=c["risk_method"], references=c["references"],
                          standard_match=c["standard"]))
        for pos, r in enumerate(comparisons):
            db.add(ComparisonRow(document_id=doc.document_id, position=pos, **r))
        emb_status = embedding_service.status()
        doc.analysis_meta = {"generation": get_client().label if get_client().available else "rule-based (no LLM configured)",
                             "embedding_backend": emb_status["backend"],
                             "vector_store": json.loads(
                                 rag_service.META_FILE.read_text()).get("vector_store"),
                             "cache_backend": cache_service.status()["backend"],
                             "classifier": classifier.status(), "document_facts": facts}
        ctx["clauses"] = [{k: c.get(k) for k in ("number", "label", "title", "text", "category", "summary", "risk_level",
                                                  "risk_reason", "precaution")} for c in clauses]
        ctx["summary"] = doc.overall_summary
        cache_service.set_document_context(doc.document_id, ctx)
        doc.status, doc.processed_at = "completed", datetime.now(timezone.utc)
        _progress(db, doc, STEPS[7], 100)
    except Exception as exc:  # noqa: BLE001
        log.error("Processing failed for document %s: %s\n%s", document_id, exc, traceback.format_exc())
        db.rollback()
        doc = db.get(Document, document_id)
        doc.status, doc.error = "failed", str(exc)
        db.commit()
    finally:
        db.close()
