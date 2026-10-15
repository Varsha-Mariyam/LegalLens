"""End-to-end API tests: auth -> upload -> processing -> analysis -> persona -> Q&A -> comparison -> PDF report."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _register(client, email):
    r = client.post("/api/auth/register", json={"name": "Tester", "email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _upload(client, headers, path, doc_type="auto"):
    with open(path, "rb") as fh:
        r = client.post("/api/documents", files={"file": (path.name, fh)}, data={"document_type": doc_type}, headers=headers)
    assert r.status_code == 202, r.text
    return r.json()["document_id"]


@pytest.fixture(scope="module")
def employment_doc(client, samples):
    h = _register(client, "owner@example.com")
    doc_id = _upload(client, h, samples / "employment_agreement_sample.pdf")
    return h, doc_id


def test_auth_flow(client):
    h = _register(client, "auth@example.com")
    assert client.get("/api/auth/me", headers=h).json()["email"] == "auth@example.com"
    assert client.post("/api/auth/register", json={"name": "x", "email": "AUTH@example.com", "password": "password123"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "auth@example.com", "password": "wrong-pass"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "auth@example.com", "password": "password123"}).status_code == 200
    assert client.post("/api/auth/register", json={"name": "x", "email": "short@example.com", "password": "short"}).status_code == 422
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_processing_completes(client, employment_doc):
    h, doc_id = employment_doc
    s = client.get(f"/api/documents/{doc_id}/status", headers=h).json()
    assert s["status"] == "completed", s
    assert s["progress"] == 100


def test_analysis_content(client, employment_doc):
    h, doc_id = employment_doc
    a = client.get(f"/api/documents/{doc_id}/analysis", headers=h).json()
    assert a["document"]["detected_type"] == "employment"
    by_title = {c["title"]: c for c in a["clauses"]}
    assert by_title["Termination"]["clause_type"] == "Termination" and by_title["Termination"]["risk_level"] == "HIGH"
    assert by_title["Confidentiality"]["clause_type"] == "Confidentiality"
    assert by_title["Liability"]["risk_level"] == "HIGH"
    for c in a["clauses"]:
        assert c["risk_level"] in ("LOW", "MEDIUM", "HIGH") and c["summary"] and c["precaution"]
    assert a["recommendations"][-1]["priority"] == "INFO"
    assert "rule-based" in a["analysis_meta"]["generation"]


def test_persona_summaries(client, employment_doc):
    h, doc_id = employment_doc
    texts = {}
    for p in ("individual", "employee", "business"):
        r = client.post(f"/api/documents/{doc_id}/persona", json={"persona": p}, headers=h)
        assert r.status_code == 200
        texts[p] = r.json()["summary"]
    assert len(set(texts.values())) == 3
    assert client.post(f"/api/documents/{doc_id}/persona", json={"persona": "judge"}, headers=h).status_code == 422
    assert len(client.get(f"/api/documents/{doc_id}/persona", headers=h).json()["summaries"]) == 3


def test_qa_with_follow_up_and_not_found(client, employment_doc):
    h, doc_id = employment_doc
    ask = lambda q: client.post(f"/api/documents/{doc_id}/qa", json={"question": q}, headers=h).json()
    a1 = ask("What is the notice period?")
    assert "seven (7) days" in a1["answer"]
    a2 = ask("Is that risky?")
    assert "Potential risk" in a2["answer"] and "HIGH" in a2["answer"]
    a3 = ask("Is there a clause about pet ownership?")
    assert "does not appear to contain" in a3["answer"]
    assert len(client.get(f"/api/documents/{doc_id}/qa", headers=h).json()) == 3


def test_comparison(client, employment_doc):
    h, doc_id = employment_doc
    c = client.get(f"/api/documents/{doc_id}/comparison", headers=h).json()
    assert "NOT an official legal standard" in c["reference_label"]
    term = next(r for r in c["rows"] if r["category"] == "Termination" and r["standard_value"] == "30 days")
    assert "7 days" in term["user_value"]


def test_pdf_report(client, employment_doc):
    h, doc_id = employment_doc
    r = client.get(f"/api/documents/{doc_id}/report", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    import fitz
    text = "".join(p.get_text() for p in fitz.open(stream=r.content, filetype="pdf"))
    for section in ("Document information", "Overall summary", "Clause classification", "Clause-wise analysis",
                    "Persona", "Comparison", "Q&A", "Recommendations", "Disclaimer"):
        assert section.lower() in text.lower(), section


def test_other_users_cannot_access(client, employment_doc):
    _, doc_id = employment_doc
    other = _register(client, "intruder@example.com")
    for path in ("", "/analysis", "/qa", "/comparison", "/report", "/text"):
        assert client.get(f"/api/documents/{doc_id}{path}", headers=other).status_code == 404
    assert client.delete(f"/api/documents/{doc_id}", headers=other).status_code == 404


def test_rejects_bad_uploads(client, tmp_path):
    h = _register(client, "uploads@example.com")
    bad = tmp_path / "virus.exe"
    bad.write_bytes(b"MZ")
    with open(bad, "rb") as fh:
        assert client.post("/api/documents", files={"file": ("virus.exe", fh)}, headers=h).status_code == 415
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with open(empty, "rb") as fh:
        assert client.post("/api/documents", files={"file": ("empty.txt", fh)}, headers=h).status_code == 422


def test_rental_docx_and_delete(client, samples):
    h = _register(client, "tenant@example.com")
    doc_id = _upload(client, h, samples / "rental_agreement_sample.docx", "rental")
    a = client.get(f"/api/documents/{doc_id}/analysis", headers=h).json()
    deposit = next(c for c in a["clauses"] if c["title"] == "Security Deposit")
    assert deposit["risk_level"] == "HIGH"
    assert client.delete(f"/api/documents/{doc_id}", headers=h).status_code == 204
    assert client.get(f"/api/documents/{doc_id}", headers=h).status_code == 404


def test_system_status(client):
    s = client.get("/api/system/status").json()
    assert {"llm", "embedding", "rag_index", "cache", "ocr", "classifier", "risk_detector", "qa_retrieval"} <= set(s)
    assert s["classifier"]["primary"] in ("ensemble", "transformer", "tfidf-logreg", "keyword/embedding baselines")
    assert client.get("/api/health").json() == {"status": "ok"}


def _status_after_upload(client, h, tmp_path, name, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    doc_id = _upload(client, h, p)
    return client.get(f"/api/documents/{doc_id}/status", headers=h).json()


def test_empty_and_unreadable_documents_fail_cleanly(client, tmp_path):
    h = _register(client, "edge@example.com")
    blank = _status_after_upload(client, h, tmp_path, "blank.txt", b"   \n\n   \t  ")
    assert blank["status"] == "failed" and "No readable text" in blank["error"]
    broken = _status_after_upload(client, h, tmp_path, "broken.pdf", b"%PDF-1.4 this is not really a pdf")
    assert broken["status"] == "failed" and broken["error"]
    with open(tmp_path / "blank.txt", "rb") as fh:
        assert client.post("/api/documents", files={"file": ("notes.odt", fh)}, headers=h).status_code == 415


def test_analysis_reports_models_and_key_sentences(client, employment_doc):
    h, doc_id = employment_doc
    a = client.get(f"/api/documents/{doc_id}/analysis", headers=h).json()
    meta = a["analysis_meta"]
    assert {"risk_detector", "qa_retrieval", "extractive_summary"} <= set(meta)
    assert 0 < len(meta["extractive_summary"]) <= 5
    texts = " ".join(c["original_text"] for c in a["clauses"])
    assert all(k["sentence"] in texts for k in meta["extractive_summary"])
    if meta["classifier"]["primary"] == "transformer":
        assert all("transformer" in c["classifier_details"] for c in a["clauses"])
    qa = client.post(f"/api/documents/{doc_id}/qa", json={"question": "How much is the salary?"}, headers=h).json()
    assert "retrieval" in qa["method"] and "45,000" in qa["answer"]


def test_pipeline_completes_without_transformer_models(client, samples):
    from backend.services import transformer_models as tm
    tm.set_model("clause", None)
    tm.set_model("risk", None)
    try:
        h = _register(client, "fallback@example.com")
        doc_id = _upload(client, h, samples / "nda_sample.txt", "nda")
        a = client.get(f"/api/documents/{doc_id}/analysis", headers=h).json()
        assert a["document"]["status"] == "completed"
        assert {c["classifier_details"]["method"] for c in a["clauses"]} <= {
            "ml:tfidf-logreg", "heading-keyword (low model confidence)", "structural (preamble/execution)"}
        assert all("transformer" not in c["classifier_details"] for c in a["clauses"])
        assert a["analysis_meta"]["classifier"]["primary"] == "tfidf-logreg"
        det = a["analysis_meta"]["risk_detector"]
        # the TF-IDF risk model (models/risk_logreg.joblib) takes over when the Transformer is missing
        assert det["used"] is False or det["mode"] == "logreg"
    finally:
        tm.reload()


def test_pipeline_rules_only_when_risk_model_disabled(client, samples, monkeypatch):
    from backend.config import settings
    from backend.services import transformer_models as tm
    monkeypatch.setattr(settings, "RISK_MODEL", "off")
    tm.reload()
    try:
        h = _register(client, "rulesonly@example.com")
        doc_id = _upload(client, h, samples / "rental_agreement_sample.txt", "rental")
        a = client.get(f"/api/documents/{doc_id}/analysis", headers=h).json()
        assert a["analysis_meta"]["risk_detector"]["used"] is False
        assert all(c["risk_method"] == "rule-based risk signals" for c in a["clauses"])
        assert next(c for c in a["clauses"] if c["title"] == "Security Deposit")["risk_level"] == "HIGH"
    finally:
        tm.reload()
