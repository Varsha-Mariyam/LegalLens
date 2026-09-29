"""Sentence embeddings.

Primary backend: Sentence Transformers (all-MiniLM-L6-v2 by default), as recommended in the guide.
Fallback backend: LSA (TF-IDF + truncated SVD) fitted on the project's own legal corpus. The fallback is
used only when the transformer model cannot be loaded (e.g. offline machine without the downloaded model)
and the active backend is always reported by /api/system/status so results are never mislabelled.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from backend.config import settings

log = logging.getLogger("legallens.embeddings")
LSA_PATH = settings.MODELS_DIR / "lsa_embedder.joblib"
LOCAL_SBERT_DIR = settings.MODELS_DIR / "sentence_transformer"


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def build_corpus() -> list[str]:
    """Texts used to fit the LSA fallback: knowledge base, reference docs, clause dataset, raw documents."""
    import csv
    texts: list[str] = []
    for folder in [settings.KNOWLEDGE_DIR, settings.STANDARD_DIR, settings.RAW_DIR]:
        for p in sorted(Path(folder).rglob("*.txt")):
            for para in p.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
                if len(para.strip()) > 30:
                    texts.append(para.strip())
    ds = settings.DATASETS_DIR / "clause_dataset.csv"
    if ds.exists():
        with ds.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                texts.append(row["clause"])
    return texts


class LSAEmbedder:
    name = "lsa-tfidf-svd"

    def __init__(self):
        import joblib
        bundle = None
        if LSA_PATH.exists():
            try:
                bundle = joblib.load(LSA_PATH)
            except Exception as exc:  # noqa: BLE001  (pickle from a different scikit-learn/numpy version)
                log.warning("Could not load %s (%s); refitting the LSA embedder from the local corpus", LSA_PATH.name, exc)
        if bundle is None:
            bundle = self.fit_and_save(build_corpus())
        self.vectorizer, self.svd = bundle["vectorizer"], bundle["svd"]
        self.dim = int(self.svd.n_components)
        # identifies this particular fitted model (a refit changes every vector, so the RAG index must be rebuilt)
        self.fingerprint = f"{bundle.get('n_texts')}:{len(self.vectorizer.vocabulary_)}:{float(self.svd.singular_values_[0]):.6f}"

    @staticmethod
    def fit_and_save(texts: list[str], n_components: int = 256) -> dict:
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        if len(texts) < 10:
            raise RuntimeError("Not enough text to fit the LSA embedder. Run scripts/build_datasets.py first.")
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000,
                              stop_words="english")
        X = vec.fit_transform(texts)
        k = max(2, min(n_components, X.shape[1] - 1, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=k, random_state=42).fit(X)
        bundle = {"vectorizer": vec, "svd": svd, "n_texts": len(texts)}
        joblib.dump(bundle, LSA_PATH)
        log.info("Fitted LSA embedder on %d texts (%d dims)", len(texts), k)
        return bundle

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.svd.transform(self.vectorizer.transform(texts)))


class SbertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        source = str(LOCAL_SBERT_DIR) if (LOCAL_SBERT_DIR / "config.json").exists() else model_name
        self.model = SentenceTransformer(source)
        self.name = "sbert-" + Path(model_name).name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.fingerprint = source

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.model.encode(list(texts), batch_size=32, show_progress_bar=False,
                                            normalize_embeddings=True))


_embedder = None
_lock = threading.Lock()
_load_error: str | None = None


def get_embedder():
    global _embedder, _load_error
    with _lock:
        if _embedder is not None:
            return _embedder
        backend = settings.EMBEDDING_BACKEND
        if backend in ("auto", "sbert"):
            try:
                _embedder = SbertEmbedder(settings.EMBEDDING_MODEL)
                return _embedder
            except Exception as exc:  # noqa: BLE001
                _load_error = f"Sentence Transformers unavailable: {exc}"
                if backend == "sbert":
                    raise
                log.warning("%s — using LSA fallback", _load_error)
        _embedder = LSAEmbedder()
        return _embedder


def status() -> dict:
    emb = get_embedder()
    return {"backend": emb.name, "dim": emb.dim, "fallback_reason": _load_error}


def encode(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize(a) @ _normalize(b).T
