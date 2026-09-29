"""LLM client supporting Anthropic, OpenAI-compatible APIs (OpenAI, Groq, LM Studio, ...) and Ollama.

When no provider is configured, `available` is False and every generation module uses its deterministic
rule-based method instead; the method used is stored with each result and shown in the UI and report.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from backend.config import settings
from backend.services import cache_service

log = logging.getLogger("legallens.llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, transport: httpx.BaseTransport | None = None):
        self.provider = self._resolve(provider or settings.LLM_PROVIDER)
        self.model = {"anthropic": settings.ANTHROPIC_MODEL, "openai": settings.OPENAI_MODEL,
                      "ollama": settings.OLLAMA_MODEL}.get(self.provider, "")
        self._transport = transport

    @staticmethod
    def _resolve(p: str) -> str:
        if p == "auto":
            if settings.ANTHROPIC_API_KEY:
                return "anthropic"
            if settings.OPENAI_API_KEY:
                return "openai"
            if settings.OLLAMA_MODEL:
                return "ollama"
            return "none"
        return p

    @property
    def available(self) -> bool:
        if self.provider == "anthropic":
            return bool(settings.ANTHROPIC_API_KEY)
        if self.provider == "openai":
            return bool(settings.OPENAI_API_KEY)
        if self.provider == "ollama":
            return bool(settings.OLLAMA_MODEL)
        return False

    @property
    def label(self) -> str:
        return f"llm:{self.provider}/{self.model}" if self.available else "none"

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=settings.LLM_TIMEOUT, transport=self._transport)

    def generate(self, system: str, prompt: str, max_tokens: int = 900, temperature: float = 0.1,
                 use_cache: bool = True) -> str:
        if not self.available:
            raise LLMError("No LLM provider configured")
        key = cache_service.llm_key(self.provider, self.model, system, prompt, str(max_tokens))
        if use_cache:
            cached = cache_service.get_llm(key)
            if cached is not None:
                return cached
        try:
            text = getattr(self, f"_{self.provider}")(system, prompt, max_tokens, temperature)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"{self.provider} HTTP {exc.response.status_code}: {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc
        text = (text or "").strip()
        if not text:
            raise LLMError(f"{self.provider} returned an empty response")
        if use_cache:
            cache_service.set_llm(key, text)
        return text

    def _anthropic(self, system, prompt, max_tokens, temperature) -> str:
        with self._client() as c:
            r = c.post("https://api.anthropic.com/v1/messages", headers={
                "x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                "content-type": "application/json"},
                json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                      "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")

    def _openai(self, system, prompt, max_tokens, temperature) -> str:
        with self._client() as c:
            r = c.post(settings.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
                       headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                       json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                             "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    def _ollama(self, system, prompt, max_tokens, temperature) -> str:
        with self._client() as c:
            r = c.post(settings.OLLAMA_URL.rstrip("/") + "/api/chat",
                       json={"model": self.model, "stream": False, "options": {"temperature": temperature,
                                                                              "num_predict": max_tokens},
                             "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"]


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("LLM did not return JSON")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_client(client: LLMClient) -> None:
    """Used by tests to inject a client with a mock transport."""
    global _client
    _client = client


def status() -> dict:
    c = get_client()
    return {"provider": c.provider, "model": c.model, "available": c.available,
            "mode": "LLM generation" if c.available else "rule-based generation (no LLM configured)"}


SYSTEM_PROMPT = (
    "You are LegalLens, an academic legal document analysis assistant. You are not a lawyer and do not give "
    "legal advice. Use only the clause text and the supplied reference context. Never invent clauses, dates, "
    "amounts, parties or obligations. Say 'potential risk' rather than giving definitive legal judgments. "
    "Clearly distinguish what the document says from your explanation.")
