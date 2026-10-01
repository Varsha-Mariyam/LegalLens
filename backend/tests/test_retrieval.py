"""Hybrid Q&A retrieval (BM25 + dense), the calibrated "not found" decision, and the extractive summariser."""
import numpy as np
import pytest

from backend.services import (classifier, clause_service, embedding_service, extraction_service, qa_service, retrieval,
                              summarizer)

CLAUSES = [
    {"title": "Salary", "text": "The Employee shall receive a monthly salary of Rs. 50,000 payable on the last day of each month.",
     "category": "Payment"},
    {"title": "Termination", "text": "Either party may terminate this Agreement by giving thirty (30) days' written notice.",
     "category": "Termination"},
    {"title": "Governing Law", "text": "This Agreement is governed by the laws of India and courts at Chennai have jurisdiction.",
     "category": "Dispute"},
]


def test_tokenize_drops_stop_words_and_keeps_numbers():
    assert retrieval.tokenize("What is the notice period of 30 days?") == ["notice", "period", "30", "day"]
    assert retrieval.tokenize("How long does confidentiality last?") == ["confidentiality", "last"]
    assert "long" in retrieval.terms("How long does it last")  # the original ranking keeps its original stop list


def test_bm25_prefers_matching_clause_and_is_non_negative():
    docs = [retrieval.tokenize(f"{c['title']} {c['text']}") for c in CLAUSES]
    s = retrieval.BM25Index(docs).scores(retrieval.tokenize("notice to terminate"))
    assert int(np.argmax(s)) == 1 and s.min() >= 0
    assert retrieval.BM25Index([[]]).scores(["x"]).tolist() == [0.0]
    assert retrieval.BM25Index([["notice"]]).scores(["notice"]).min() >= 0  # single clause: idf would be negative


def test_fusion_modes():
    dense, lex = np.array([0.9, 0.1, 0.5]), np.array([0.0, 4.0, 1.0])
    assert np.allclose(retrieval.fuse(dense, lex, "weighted", alpha=1.0), retrieval.minmax(dense))
    assert np.allclose(retrieval.fuse(dense, lex, "weighted", alpha=0.0), retrieval.minmax(lex))
    rrf = retrieval.fuse(dense, lex, "rrf")
    assert rrf.shape == (3,) and rrf[2] == pytest.approx(2 / 62)  # second in both rankings


def _ctx(clauses):
    return {"doc_type": "employment", "clauses": clauses, "embedder": embedding_service.get_embedder().name,
            "clause_embeddings": embedding_service.encode([f"{c['title']} {c['text']}" for c in clauses]).tolist()}


def test_hybrid_ranking_and_legacy_mode_agree_on_easy_questions():
    ctx = _ctx(CLAUSES)
    for mode in ("hybrid", "dense", "legacy"):
        q = embedding_service.encode(["How much is the salary?"])[0]
        res = retrieval.rank("How much is the salary?", q, np.asarray(ctx["clause_embeddings"]), CLAUSES, ctx["embedder"],
                             mode=mode, intent="Payment")
        assert res["ranked"][0][1] == 0 and res["answerable"], mode
    assert qa_service.rank_clauses("Which court has jurisdiction?", ctx)[0][1]["title"] == "Governing Law"


def test_calibrated_config_covers_both_embedding_backends():
    cfg = retrieval.load_config("sbert-all-MiniLM-L6-v2")
    if not cfg.get("answerability"):
        pytest.skip("run scripts/evaluate_models.py to calibrate retrieval")
    assert 0 <= cfg["alpha"] <= 1 and 0 < cfg["answerability"]["threshold"] < 1
    assert retrieval.load_config("lsa-tfidf-svd").get("answerability")
    low = retrieval.p_answerable([0.10, 0.0, 0.0], cfg["answerability"])
    high = retrieval.p_answerable([0.70, 3.0, 1.0], cfg["answerability"])
    assert low < cfg["answerability"]["threshold"] <= high


def test_uncalibrated_fallback_rule():
    assert retrieval.decide_answerable([0.2, 0.0, 0.5], {"answerability": None}) == (True, None)
    assert retrieval.decide_answerable([0.2, 0.0, 0.0], {"answerability": None}) == (False, None)


def test_not_found_and_notice_period_on_sample(samples):
    text = extraction_service.extract(str(samples / "employment_agreement_sample.txt"))["text"]
    segs = clause_service.segment(text)
    clauses = [{"number": s["id"], "label": s["label"], "title": s["title"], "text": s["text"], "category": c["category"]}
               for s, c in zip(segs, classifier.classify_batch(segs))]
    ctx = _ctx(clauses)
    ranked = qa_service.rank_clauses("What is the notice period?", ctx)
    # MiniLM (default): the termination clause ranks first. LSA fallback: within the top 2, which the answer quotes.
    top = 1 if ctx["embedder"].startswith("sbert-") else 2
    assert any("seven (7) days" in c["text"] for _, c in ranked[:top])
    assert qa_service.rank_clauses("Is there a clause about pet ownership?", ctx) == []
    res = qa_service.answer("What is the notice period?", ctx, [])
    assert "retrieval" in res["method"] and res["sources"][0].get("bm25") is not None


def test_extractive_summary_quotes_document_in_order():
    clauses = [{"number": i + 1, "label": str(i + 1), "category": c["category"], **c} for i, c in enumerate(CLAUSES)]
    clauses.insert(0, {"number": 0, "label": None, "title": "Preamble / Parties", "category": "Others",
                       "text": "This Agreement is made between ABC Pvt. Ltd. and Mr. X on 1 January 2026 at Chennai."})
    key = summarizer.extractive_summary(clauses, k=2)
    assert len(key) == 2
    assert all(any(k["sentence"] in c["text"] for c in clauses[1:]) for k in key)
    assert [k["clause_number"] for k in key] == sorted(k["clause_number"] for k in key)
    assert summarizer.extractive_summary([]) == []
    text, method = summarizer.overall_summary("employment", clauses, "", key)
    assert "Key sentences from the document (extractive):" in text and "extractive" in method
