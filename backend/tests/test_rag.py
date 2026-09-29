"""Phase 5 — RAG over the controlled knowledge base and the project reference templates."""
from backend.services import rag_service


def test_index_contents():
    meta = rag_service.build_index()
    assert meta["knowledge_chunks"] > 20 and meta["standard_clauses"] > 20


def test_guide_example_retrieval():
    q = "Either party can terminate with 7 days notice."
    refs = rag_service.retrieve_knowledge(q, "employment", "Termination", k=3)
    assert refs[0]["metadata"]["source"] == "employment/termination.txt"
    std = rag_service.retrieve_standard(q, "employment", "Termination", k=1)[0]
    assert "thirty (30) days" in std["text"]
    assert "not an official legal standard" in std["metadata"]["citation"]


def test_retrieval_is_filtered_by_document_type():
    refs = rag_service.retrieve_knowledge("security deposit refund", "rental", "Payment", k=4)
    assert all(r["metadata"]["doc_type"] in ("rental", "general") for r in refs)
    assert any(r["metadata"]["source"] == "rental/deposit.txt" for r in refs)


def test_knowledge_files_cite_sources():
    for s in rag_service.list_sources():
        assert s["title"] and s["sources"], s["file"]


def test_reference_templates_have_annotated_categories():
    rental = {c["title"]: c["category"] for c in rag_service.load_standard_template("rental")}
    assert rental["Premises and Term"] == "Others" and rental["Security Deposit"] == "Payment"
