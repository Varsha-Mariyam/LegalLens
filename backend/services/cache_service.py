"""CAG — Cache-Augmented Generation (guide section 10).

Keeps the uploaded document's reusable context (full text, clauses, clause embeddings, summary, metadata)
so Q&A and analysis do not rebuild it on every request. Also caches LLM responses by prompt hash.
Backend: Redis when REDIS_URL is set, otherwise an in-memory dict mirrored to JSON files in cache/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

from backend.config import settings

log = logging.getLogger("legallens.cache")


class _Backend:
    def __init__(self):
        self.kind = "memory+disk"
        self._mem: dict[str, str] = {}
        self._lock = threading.Lock()
        self._redis = None
        if settings.REDIS_URL:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=3)
                self._redis.ping()
                self.kind = "redis"
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis unavailable (%s); using in-memory + disk cache", exc)
                self._redis = None
        (settings.CACHE_DIR / "documents").mkdir(parents=True, exist_ok=True)
        (settings.CACHE_DIR / "llm").mkdir(parents=True, exist_ok=True)

    def _file(self, key: str):
        ns, _, name = key.partition(":")
        return settings.CACHE_DIR / ns / f"{hashlib.sha1(name.encode()).hexdigest()}.json"

    def get(self, key: str):
        if self._redis is not None:
            return self._redis.get(key)
        with self._lock:
            if key in self._mem:
                return self._mem[key]
        f = self._file(key)
        if f.exists():
            value = f.read_text(encoding="utf-8")
            with self._lock:
                self._mem[key] = value
            return value
        return None

    def set(self, key: str, value: str, ttl: int | None = None):
        if self._redis is not None:
            self._redis.set(key, value, ex=ttl)
            return
        with self._lock:
            self._mem[key] = value
        self._file(key).write_text(value, encoding="utf-8")

    def delete(self, key: str):
        if self._redis is not None:
            self._redis.delete(key)
            return
        with self._lock:
            self._mem.pop(key, None)
        f = self._file(key)
        if f.exists():
            f.unlink()


_backend: _Backend | None = None
stats = {"context_hits": 0, "context_misses": 0, "llm_hits": 0, "llm_misses": 0}


def backend() -> _Backend:
    global _backend
    if _backend is None:
        _backend = _Backend()
    return _backend


def set_document_context(document_id: int, context: dict) -> None:
    context = {**context, "cached_at": time.time()}
    backend().set(f"documents:{document_id}", json.dumps(context))


def get_document_context(document_id: int) -> dict | None:
    raw = backend().get(f"documents:{document_id}")
    if raw is None:
        stats["context_misses"] += 1
        return None
    stats["context_hits"] += 1
    return json.loads(raw)


def update_document_context(document_id: int, **fields) -> None:
    ctx = get_document_context(document_id)
    if ctx is not None:
        ctx.update(fields)
        set_document_context(document_id, ctx)


def delete_document_context(document_id: int) -> None:
    backend().delete(f"documents:{document_id}")


def llm_key(*parts: str) -> str:
    return hashlib.sha256("\u241e".join(parts).encode("utf-8")).hexdigest()


def get_llm(key: str) -> str | None:
    raw = backend().get(f"llm:{key}")
    if raw is None:
        stats["llm_misses"] += 1
        return None
    stats["llm_hits"] += 1
    return json.loads(raw)["text"]


def set_llm(key: str, text: str) -> None:
    backend().set(f"llm:{key}", json.dumps({"text": text}), ttl=60 * 60 * 24 * 30)


def status() -> dict:
    return {"backend": backend().kind, **stats}
