"""Phase 7 — evidence-based risk analysis."""
from backend.services import rag_service, risk_analyzer


def _analyze(text, category, doc_type="employment", facts=None):
    std = rag_service.retrieve_standard(text, doc_type, category, k=1)
    return risk_analyzer.analyze({"text": text, "category": category, "title": None}, doc_type,
                                 std[0] if std else None, [], facts or {})


def test_short_one_sided_notice_is_high():
    r = _analyze("The Company may terminate this Agreement at any time by giving seven (7) days' notice.", "Termination")
    assert r["risk_level"] == "HIGH"
    assert any(e["finding"].startswith("Notice period") for e in r["evidence"])
    assert r["precaution"]


def test_mutual_reasonable_notice_is_low():
    r = _analyze("Either party may terminate this Agreement by giving thirty (30) days' written notice to the other.", "Termination")
    assert r["risk_level"] == "LOW"


def test_unlimited_liability_is_high():
    r = _analyze("The Employee shall be liable for any and all losses, damages and claims suffered by the Company.", "Liability")
    assert r["risk_level"] == "HIGH"


def test_excess_rental_deposit():
    r = _analyze("The Tenant shall pay a security deposit of Rs. 1,80,000 (ten months' rent).", "Payment", "rental",
                 {"monthly_rent": 18000})
    assert r["risk_level"] in ("MEDIUM", "HIGH")
    assert any("Model Tenancy Act" in (e["source"] or "") for e in r["evidence"])


def test_wording_is_potential_risk_not_verdict():
    r = _analyze("The Employee shall not join any competitor for two (2) years after leaving the Company.", "Others")
    assert r["risk_level"] == "HIGH"
    assert "illegal" not in r["risk_reason"].lower()
