"""RAG: chunk the controlled legal knowledge base and reference documents, embed, store, retrieve (guide section 9)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path

from backend.config import settings
from backend.services import embedding_service, vector_store

log = logging.getLogger("legallens.rag")
META_FILE = settings.VECTOR_DB_DIR / "index_meta.json"
CATEGORY_TOPICS = {
    "Payment": {"payment", "rent", "deposit"},
    "Termination": {"termination"},
    "Confidentiality": {"confidentiality", "non_compete"},
    "Liability": {"liability"},
    "Dispute": {"dispute"},
    "Others": {"general", "registration", "non_compete"},
}
_lock = threading.Lock()
_ready = False


def parse_knowledge_file(path: Path) -> tuple[dict, list[str]]:
    header = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines) and lines[i].strip():
        if ":" in lines[i]:
            k, v = lines[i].split(":", 1)
            header[k.strip().lower()] = v.strip()
        i += 1
    paragraphs = [p.strip() for p in "\n".join(lines[i:]).split("\n\n") if p.strip()]
    return header, paragraphs


def chunk_paragraphs(paragraphs: list[str], target: int = 700) -> list[str]:
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > target:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}".strip()
    if current:
        chunks.append(current)
    return chunks


def knowledge_chunks() -> list[dict]:
    out = []
    for path in sorted(settings.KNOWLEDGE_DIR.rglob("*.txt")):
        header, paragraphs = parse_knowledge_file(path)
        doc_type = path.parent.name
        rel = path.relative_to(settings.KNOWLEDGE_DIR).as_posix()
        for n, chunk in enumerate(chunk_paragraphs(paragraphs)):
            out.append({"id": f"kb::{rel}::{n}", "text": chunk, "metadata": {
                "source": rel, "doc_type": doc_type, "topic": header.get("topic", path.stem),
                "title": header.get("title", path.stem), "citation": header.get("sources", ""),
                "note": header.get("note", ""), "kind": "knowledge"}})
    return out


def load_standard_template(doc_type: str) -> list[dict]:
    """Segment a project reference template; categories come from the manual annotations in
    data/standard_documents/categories.json, falling back to the classifier for unlisted clauses."""
    from backend.services import classifier, clause_service, extraction_service
    path = settings.STANDARD_DIR / f"standard_{doc_type}.txt"
    if not path.exists():
        return []
    text = extraction_service.extract(str(path))["text"]
    clauses = [c for c in clause_service.segment(text) if not (c["title"] or "").startswith("Preamble")]
    ann_file = settings.STANDARD_DIR / "categories.json"
    annotations = json.loads(ann_file.read_text(encoding="utf-8")).get(doc_type, {}) if ann_file.exists() else {}
    for c, cat in zip(clauses, classifier.classify_batch(clauses)):
        c["category"] = annotations.get(c["title"] or "", cat["category"])
    return clauses


def standard_clauses() -> list[dict]:
    out = []
    for path in sorted(settings.STANDARD_DIR.glob("standard_*.txt")):
        doc_type = path.stem.replace("standard_", "")
        for c in load_standard_template(doc_type):
            out.append({"id": f"std::{doc_type}::{c['id']}", "text": c["text"], "metadata": {
                "source": path.name, "doc_type": doc_type, "category": c["category"], "title": c["title"] or "",
                "label": c["label"] or "", "kind": "standard",
                "citation": "Project reference template (not an official legal standard)"}})
    return out


def _fingerprint() -> str:
    h = hashlib.sha256()
    for folder in (settings.KNOWLEDGE_DIR, settings.STANDARD_DIR):
        for p in sorted([*folder.rglob("*.txt"), *folder.glob("*.json")]):
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
    emb = embedding_service.get_embedder()
    h.update(emb.name.encode())
    h.update(str(getattr(emb, "fingerprint", "")).encode())  # a refitted model changes every vector
    return h.hexdigest()


def _collection(kind: str):
    emb = embedding_service.get_embedder()
    return vector_store.get_store(f"{kind}_{emb.name}")


def build_index(force: bool = False) -> dict:
    global _ready
    with _lock:
        fp = _fingerprint()
        meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
        kb, std = _collection("legal_knowledge"), _collection("standard_clauses")
        if not force and meta.get("fingerprint") == fp and kb.count() > 0 and std.count() > 0:
            _ready = True
            return meta
        kb.reset()
        std.reset()
        chunks = knowledge_chunks()
        kb.add([c["id"] for c in chunks], [c["text"] for c in chunks],
               embedding_service.encode([c["text"] for c in chunks]), [c["metadata"] for c in chunks])
        sc = standard_clauses()
        std.add([c["id"] for c in sc], [c["text"] for c in sc],
                embedding_service.encode([c["text"] for c in sc]), [c["metadata"] for c in sc])
        meta = {"fingerprint": fp, "knowledge_chunks": len(chunks), "standard_clauses": len(sc),
                "embedding_backend": embedding_service.get_embedder().name, "vector_store": kb.kind}
        META_FILE.write_text(json.dumps(meta, indent=2))
        _ready = True
        log.info("RAG index built: %s", meta)
        return meta


def ensure_index():
    if not _ready:
        build_index()


def _boost(results: list[dict], category: str | None) -> list[dict]:
    topics = CATEGORY_TOPICS.get(category or "", set())
    for r in results:
        r["adjusted_score"] = round(r["score"] + (0.08 if r["metadata"].get("topic") in topics else 0.0), 4)
    return sorted(results, key=lambda r: -r["adjusted_score"])


def retrieve_knowledge(query: str, doc_type: str, category: str | None = None, k: int = 3) -> list[dict]:
    ensure_index()
    q = embedding_service.encode([query])[0]
    types = [doc_type, "general"] if doc_type != "general" else ["general"]
    res = _collection("legal_knowledge").query(q, k=k + 4, where={"doc_type": {"$in": types}})
    return _boost(res, category)[:k]


def retrieve_standard(query: str, doc_type: str, category: str | None = None, k: int = 1) -> list[dict]:
    ensure_index()
    q = embedding_service.encode([query])[0]
    where = {"doc_type": doc_type}
    if category and category != "Others":
        where = {"$and": [{"doc_type": doc_type}, {"category": category}]}
    res = _collection("standard_clauses").query(q, k=k, where=where)
    if not res:
        res = _collection("standard_clauses").query(q, k=k, where={"doc_type": doc_type})
    return res


def list_sources() -> list[dict]:
    out = []
    for path in sorted(settings.KNOWLEDGE_DIR.rglob("*.txt")):
        header, _ = parse_knowledge_file(path)
        out.append({"file": path.relative_to(settings.KNOWLEDGE_DIR).as_posix(), "title": header.get("title"),
                    "topic": header.get("topic"), "sources": header.get("sources"), "note": header.get("note")})
    return out


def format_reference(r: dict) -> str:
    m = r["metadata"]
    cite = f" [{m.get('citation')}]" if m.get("citation") else ""
    return f"({m.get('title') or m.get('source')}){cite}\n{r['text']}"


def short_citation(r: dict) -> str:
    c = r["metadata"].get("citation", "")
    return re.split(r";", c)[0].strip() if c else r["metadata"].get("source", "")
