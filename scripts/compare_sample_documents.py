"""Risk profile of every raw document, fallback configuration vs upgraded configuration.

  fallback  = TF-IDF + Logistic Regression classifier, rule-based risk signals only (what the app does without the
              Transformer models; same rules as the original version)
  upgraded  = fine-tuned Transformer classifier, rule signals + learned risk detector (the deployed configuration)

The synthetic samples were written to be either balanced or one-sided, so this shows whether the learned detector adds
flags to balanced documents. Output: reports/evaluation/sample_documents.json (included in RESULTS.md).
Usage: python scripts/compare_sample_documents.py
"""
from __future__ import annotations

import json
from collections import Counter

import _bootstrap  # noqa: F401

from backend.config import settings
from backend.services import (classifier, clause_service, document_type, extraction_service, pipeline, rag_service,
                              risk_analyzer)
from backend.services import transformer_models as tm

OUT = settings.REPORTS_DIR / "evaluation" / "sample_documents.json"


LOADED: dict = {}


def analyse(path, use_models: bool) -> dict:
    text = extraction_service.extract(str(path))["text"]
    doc_type, _ = document_type.detect(text)
    segs = clause_service.segment(text)
    for task in tm.TASKS:
        tm.set_model(task, LOADED[task] if use_models else None)
    cats = classifier.classify_batch(segs)
    clauses = [{"number": s["id"], "label": s["label"], "title": s["title"], "text": s["text"], "category": c["category"]}
               for s, c in zip(segs, cats)]
    facts = pipeline.document_facts(doc_type, clauses)
    detections = tm.detect_risks([tm.risk_input(c["title"], c["text"]) for c in clauses]) if use_models else None
    levels, model_items = Counter(), []
    for i, c in enumerate(clauses):
        std = rag_service.retrieve_standard(f"{c['title'] or ''} {c['text']}", doc_type, c["category"], k=1)
        standard = std[0] if std and std[0]["score"] > 0.15 else None
        r = risk_analyzer.analyze(c, doc_type, standard, [], facts,
                                  model_risks=detections[i] if detections is not None else None)
        levels[r["risk_level"]] += 1
        model_items += [f"{c['title'] or c['number']}: {e['rule'][6:]} ({e['level']})" for e in r["evidence"] if e.get("kind") == "model"]
    return {"doc_type": doc_type, "clauses": len(clauses), "HIGH": levels["HIGH"], "MEDIUM": levels["MEDIUM"],
            "LOW": levels["LOW"], "categories": dict(Counter(c["category"] for c in clauses)), "detector_evidence": model_items}


def main() -> None:
    rag_service.ensure_index()
    for task in tm.TASKS:
        LOADED[task] = tm.get(task)
        if LOADED[task] is None:
            raise SystemExit(f"{task} model not available: {tm.status()[task]['error']}")
    files = sorted((settings.RAW_DIR / "samples").glob("*.txt")) + sorted((settings.RAW_DIR / "cuad").glob("*.txt"))
    rows = []
    for p in files:
        before, after = analyse(p, False), analyse(p, True)
        rows.append({"file": str(p.relative_to(settings.RAW_DIR)), "fallback": before, "upgraded": after})
        print(f"{p.name[:60]:60s} fallback H/M/L {before['HIGH']}/{before['MEDIUM']}/{before['LOW']}  "
              f"upgraded {after['HIGH']}/{after['MEDIUM']}/{after['LOW']}  detector items {len(after['detector_evidence'])}")
    tm.reload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
