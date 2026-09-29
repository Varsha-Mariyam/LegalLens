"""Run the full LegalLens pipeline on a file from the command line (no web server needed).

Usage:  python scripts/run_pipeline_cli.py data/raw_documents/samples/employment_agreement_sample.pdf [--type employment] [--persona employee] [--question "What is the notice period?"]
Creates (or reuses) a local CLI user, stores the analysis in the database, prints the results and writes the PDF report.
"""
import argparse
import shutil
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from backend.config import settings
from backend.database.db import SessionLocal, init_db
from backend.database.models import Conversation, Document, PersonaSummary, User
from backend.routes.security import hash_password
from backend.services import persona_service, pipeline, qa_service, rag_service, report_service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--type", default="auto", choices=["auto", *settings.DOCUMENT_TYPES])
    ap.add_argument("--persona", default="individual", choices=settings.PERSONAS)
    ap.add_argument("--question", action="append", default=[])
    args = ap.parse_args()
    init_db()
    db = SessionLocal()
    user = db.query(User).filter(User.email == "cli@legallens.local").first()
    if user is None:
        user = User(name="CLI user", email="cli@legallens.local", password_hash=hash_password(uuid.uuid4().hex))
        db.add(user)
        db.commit()
    src = Path(args.file)
    stored = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}_{src.name}"
    shutil.copy(src, stored)
    doc = Document(user_id=user.user_id, filename=src.name, stored_path=str(stored), document_type=args.type)
    db.add(doc)
    db.commit()
    doc_id = doc.document_id
    db.close()

    pipeline.process_document(doc_id)
    db = SessionLocal()
    doc = db.get(Document, doc_id)
    if doc.status != "completed":
        raise SystemExit(f"Processing failed: {doc.error}")
    print(f"\n=== {doc.filename} ({doc.detected_type}, {doc.extraction_method}) ===\n")
    print(doc.overall_summary, "\n")
    for c in doc.clauses:
        print(f"[{c.clause_number:>2}] {c.clause_type:<15} {c.risk_level:<6} {c.title or ''}")
    ctx = pipeline.get_context(doc)
    refs = rag_service.retrieve_knowledge(" ".join(persona_service.PERSONAS[args.persona]["focus"]), doc.detected_type, k=3)
    text, method = persona_service.generate(args.persona, doc.detected_type, ctx["clauses"], refs)
    db.add(PersonaSummary(document_id=doc_id, persona=args.persona, summary=text, method=method))
    print(f"\n--- Persona: {args.persona} ({method}) ---\n{text}")
    history = []
    for q in args.question:
        res = qa_service.answer(q, ctx, history)
        history.append({"question": q, "answer": res["answer"], "sources": res["sources"]})
        db.add(Conversation(document_id=doc_id, question=q, answer=res["answer"], sources=res["sources"], method=res["method"]))
        print(f"\nQ: {q}\nA: {res['answer']}")
    db.commit()
    doc = db.get(Document, doc_id)
    path = report_service.build_report(doc, list(doc.clauses), list(doc.comparisons), list(doc.persona_summaries),
                                       list(doc.conversations))
    print(f"\nPDF report: {path}")
    db.close()


if __name__ == "__main__":
    main()
