"""Vector database wrapper: ChromaDB (persistent, cosine) with a NumPy fallback store."""
from __future__ import annotations

import json
import logging
import re

import numpy as np

from backend.config import settings

log = logging.getLogger("legallens.vectors")


class ChromaStore:
    kind = "chromadb"

    def __init__(self, name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR / "chroma"),
                                                settings=ChromaSettings(anonymized_telemetry=False))
        self.name = name
        self.col = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def reset(self):
        try:
            self.client.delete_collection(self.name)
        except Exception:  # noqa: BLE001  — the collection does not exist yet, nothing to delete
            pass
        self.col = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, texts, embeddings, metadatas):
        for start in range(0, len(ids), 500):
            sl = slice(start, start + 500)
            self.col.upsert(ids=list(ids[sl]), documents=list(texts[sl]),
                            embeddings=[list(map(float, e)) for e in embeddings[sl]], metadatas=list(metadatas[sl]))

    def count(self) -> int:
        return self.col.count()

    def query(self, embedding, k: int = 5, where: dict | None = None) -> list[dict]:
        n = self.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[list(map(float, embedding))], n_results=min(k, n), where=where or None,
                             include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "metadata": meta, "score": round(1.0 - float(dist), 4)})
        return out


class NumpyStore:
    kind = "numpy"

    def __init__(self, name: str):
        self.name = name
        self.dir = settings.VECTOR_DB_DIR / "numpy"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _paths(self):
        return self.dir / f"{self.name}.npy", self.dir / f"{self.name}.json"

    def _load(self):
        vec_p, meta_p = self._paths()
        if vec_p.exists() and meta_p.exists():
            self.vectors = np.load(vec_p)
            data = json.loads(meta_p.read_text(encoding="utf-8"))
            self.ids, self.texts, self.metas = data["ids"], data["texts"], data["metas"]
        else:
            self.vectors, self.ids, self.texts, self.metas = np.zeros((0, 0), dtype=np.float32), [], [], []

    def _save(self):
        vec_p, meta_p = self._paths()
        np.save(vec_p, self.vectors)
        meta_p.write_text(json.dumps({"ids": self.ids, "texts": self.texts, "metas": self.metas}), encoding="utf-8")

    def reset(self):
        self.vectors, self.ids, self.texts, self.metas = np.zeros((0, 0), dtype=np.float32), [], [], []
        self._save()

    def add(self, ids, texts, embeddings, metadatas):
        emb = np.asarray(embeddings, dtype=np.float32)
        self.vectors = emb if self.vectors.size == 0 else np.vstack([self.vectors, emb])
        self.ids += list(ids)
        self.texts += list(texts)
        self.metas += list(metadatas)
        self._save()

    def count(self) -> int:
        return len(self.ids)

    @staticmethod
    def _match(meta: dict, where: dict | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(NumpyStore._match(meta, w) for w in where["$and"])
        for key, cond in where.items():
            if isinstance(cond, dict) and "$in" in cond:
                if meta.get(key) not in cond["$in"]:
                    return False
            elif meta.get(key) != cond:
                return False
        return True

    def query(self, embedding, k: int = 5, where: dict | None = None) -> list[dict]:
        if not self.ids:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            if self._match(self.metas[i], where):
                out.append({"text": self.texts[i], "metadata": self.metas[i], "score": round(float(scores[i]), 4)})
                if len(out) >= k:
                    break
        return out


def get_store(name: str):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60]
    if settings.VECTOR_STORE in ("auto", "chroma"):
        try:
            return ChromaStore(safe)
        except Exception as exc:  # noqa: BLE001
            if settings.VECTOR_STORE == "chroma":
                raise
            log.warning("ChromaDB unavailable (%s); using NumPy vector store", exc)
    return NumpyStore(safe)
