"""LLM client (Anthropic / OpenAI-compatible / Ollama) tested against mocked HTTP endpoints — no network, no key."""
import json

import httpx
import pytest

from backend.config import settings
from backend.services import llm_service, qa_service, risk_analyzer, summarizer
from backend.services.llm_service import LLMClient, LLMError


def _transport(reply_json, calls, status=200):
    def handler(request: httpx.Request):
        calls.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx.Response(status, json=reply_json)
    return httpx.MockTransport(handler)


@pytest.fixture
def anthropic_key(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-key")


def test_anthropic_request_format(anthropic_key):
    calls = []
    c = LLMClient("anthropic", _transport({"content": [{"type": "text", "text": "Hello"}]}, calls))
    assert c.generate("sys", "prompt one", use_cache=False) == "Hello"
    req = calls[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "test-key" and req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["body"]["system"] == "sys" and req["body"]["messages"][0]["content"] == "prompt one"


def test_openai_compatible_request_format(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    calls = []
    c = LLMClient("openai", _transport({"choices": [{"message": {"content": "Hi"}}]}, calls))
    assert c.generate("sys", "p", use_cache=False) == "Hi"
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["headers"]["authorization"] == "Bearer sk-test"
    assert calls[0]["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_request_format(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3.1")
    calls = []
    c = LLMClient("ollama", _transport({"message": {"content": "Namaskaram"}}, calls))
    assert c.generate("s", "p", use_cache=False) == "Namaskaram"
    assert calls[0]["url"].endswith("/api/chat") and calls[0]["body"]["stream"] is False


def test_responses_are_cached(anthropic_key):
    calls = []
    c = LLMClient("anthropic", _transport({"content": [{"type": "text", "text": "cached"}]}, calls))
    c.generate("s", "unique prompt for cache test")
    c.generate("s", "unique prompt for cache test")
    assert len(calls) == 1


def test_http_error_raises_llm_error(anthropic_key):
    c = LLMClient("anthropic", _transport({"error": "overloaded"}, [], status=529))
    with pytest.raises(LLMError):
        c.generate("s", "p", use_cache=False)


def test_unconfigured_client_is_unavailable():
    c = LLMClient("none")
    assert not c.available
    with pytest.raises(LLMError):
        c.generate("s", "p")


def test_parse_json_handles_fences():
    assert llm_service.parse_json('```json\n{"risk_level": "HIGH"}\n```') == {"risk_level": "HIGH"}
    with pytest.raises(LLMError):
        llm_service.parse_json("no json here")


def test_risk_uses_llm_but_never_lowers_rule_level(anthropic_key):
    reply = {"content": [{"type": "text", "text": '{"risk_level": "LOW", "reason": "LLM reason", "precaution": "LLM precaution"}'}]}
    llm_service.set_client(LLMClient("anthropic", _transport(reply, [])))
    clause = {"text": "The Company may terminate at any time by giving seven (7) days' notice.", "category": "Termination",
              "title": None, "summary": ""}
    r = risk_analyzer.analyze(clause, "employment", None, [], {})
    assert r["risk_level"] == "HIGH"          # rule says HIGH, LLM says LOW -> keep the higher level
    assert "llm:anthropic" in r["method"]


def test_llm_failure_falls_back_to_rules(anthropic_key):
    llm_service.set_client(LLMClient("anthropic", _transport({}, [], status=500)))
    text, method = summarizer.explain_clause({"text": "The Tenant shall pay rent of Rs. 10,000 monthly.",
                                              "category": "Payment", "title": "Rent"}, [])
    assert "Rs. 10,000" in text and "LLM error" in method


def test_qa_llm_path_receives_document_context(anthropic_key):
    calls = []
    llm_service.set_client(LLMClient("anthropic", _transport({"content": [{"type": "text", "text": "[Clause 7] 7 days."}]}, calls)))
    import numpy as np
    from backend.services import embedding_service
    clauses = [{"number": 1, "label": "7", "title": "Termination", "category": "Termination", "risk_level": "HIGH",
                "text": "The Company may terminate by giving seven (7) days' notice."}]
    ctx = {"doc_type": "employment", "clauses": clauses,
           "clause_embeddings": embedding_service.encode([clauses[0]["text"]]).tolist()}
    res = qa_service.answer("What is the notice period?", ctx, [])
    assert res["answer"] == "[Clause 7] 7 days."
    prompt = calls[0]["body"]["messages"][0]["content"]
    assert "seven (7) days" in prompt and qa_service.NOT_FOUND in prompt
    assert np.isfinite(res["sources"][0]["score"])
