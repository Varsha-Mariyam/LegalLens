"""Phase 9 — semantic comparison with the project reference template (guide example table)."""
from backend.services import classifier, clause_service, comparison_service, embedding_service, extraction_service


def _rows(path, doc_type):
    text = extraction_service.extract(str(path))["text"]
    segs = clause_service.segment(text)
    clauses = [{"number": s["id"], "label": s["label"], "title": s["title"], "text": s["text"], "category": c["category"]}
               for s, c in zip(segs, classifier.classify_batch(segs))]
    emb = embedding_service.encode([f"{c['title'] or ''} {c['text']}" for c in clauses])
    return comparison_service.compare(doc_type, clauses, emb)


def test_employment_comparison_matches_guide_table(samples):
    rows = _rows(samples / "employment_agreement_sample.txt", "employment")
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], r)
    assert by_cat["Termination"]["standard_value"] == "30 days"
    assert "7 days" in by_cat["Termination"]["user_value"]
    assert by_cat["Liability"]["standard_value"] == "Limited" and by_cat["Liability"]["user_value"] == "Unlimited"
    assert by_cat["Confidentiality"]["user_value"].startswith("Present")
    assert by_cat["Payment"]["standard_value"] == "Rs. 50,000"


def test_balanced_lease_is_closer_to_reference(samples):
    one_sided = _rows(samples / "rental_agreement_sample.txt", "rental")
    balanced = _rows(samples / "lease_agreement_balanced_sample.txt", "rental")
    assert sum(r["status"] == "equivalent" for r in balanced) > sum(r["status"] == "equivalent" for r in one_sided)


def test_statuses_valid(samples):
    for r in _rows(samples / "nda_sample.txt", "nda"):
        assert r["status"] in ("equivalent", "differs", "missing", "additional")
