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
    assert {"llm", "embedding", "rag_index", "cache", "ocr", "classifier"} <= set(s)
    assert client.get("/api/health").json() == {"status": "ok"}
