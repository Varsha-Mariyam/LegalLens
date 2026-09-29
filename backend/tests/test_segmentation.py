"""Phase 4 — deterministic clause segmentation into [{id, title, text}]."""
from backend.services import clause_service, extraction_service

GUIDE_EXAMPLE = """EMPLOYMENT AGREEMENT
1. Salary
The employee shall receive a monthly salary of Rs. 50,000.
2. Confidentiality
The employee shall not disclose confidential information.
3. Termination
Either party may terminate this agreement with 30 days notice.
"""


def test_guide_example_structure():
    clauses = clause_service.segment(GUIDE_EXAMPLE)
    titled = [c for c in clauses if c["title"] in ("Salary", "Confidentiality", "Termination")]
    assert [c["title"] for c in titled] == ["Salary", "Confidentiality", "Termination"]
    assert titled[0]["text"].startswith("The employee shall receive")
    assert all({"id", "title", "text", "label"} <= set(c) for c in clauses)
    assert [c["id"] for c in clauses] == list(range(1, len(clauses) + 1))


def test_sample_documents_segment_consistently_across_formats(samples):
    counts = {e: len(clause_service.segment(extraction_service.extract(str(samples / f"employment_agreement_sample.{e}"))["text"]))
              for e in ("pdf", "docx", "txt")}
    assert len(set(counts.values())) == 1 and counts["txt"] >= 10


def test_paragraph_fallback_without_numbering():
    p1 = "The Client shall pay the Service Provider a fee of Rs. 40,000 per month within thirty days of receiving a valid invoice."
    p2 = "Either party may terminate this arrangement by giving the other party thirty days' written notice of termination."
    clauses = clause_service.segment(f"{p1}\n\n{p2}")
    assert [c["text"] for c in clauses] == [p1, p2]


def test_short_fragments_are_merged_not_split():
    clauses = clause_service.segment("Fee: Rs. 500.\n\nPaid monthly.")
    assert len(clauses) == 1
