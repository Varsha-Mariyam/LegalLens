"""Phase 6 — CAG document context cache and LLM response cache."""
from backend.services import cache_service


def test_document_context_roundtrip():
    ctx = {"document_id": 999, "full_text": "abc", "clauses": [{"number": 1, "text": "x"}], "summary": None}
    cache_service.set_document_context(999, ctx)
    assert cache_service.get_document_context(999)["clauses"][0]["text"] == "x"
    cache_service.update_document_context(999, summary="short summary")
    assert cache_service.get_document_context(999)["summary"] == "short summary"
    cache_service.delete_document_context(999)
    assert cache_service.get_document_context(999) is None


def test_context_survives_memory_reset():
    cache_service.set_document_context(1000, {"document_id": 1000, "full_text": "persisted"})
    cache_service.backend()._mem.clear()
    assert cache_service.get_document_context(1000)["full_text"] == "persisted"


def test_llm_cache():
    key = cache_service.llm_key("p", "m", "prompt")
    assert cache_service.get_llm(key) is None
    cache_service.set_llm(key, "answer")
    assert cache_service.get_llm(key) == "answer"
