"""Clause retrieval for document Q&A (guide section 14).

Three rankers over the clauses of one document:
  * legacy  — the original LegalLens ranking: 0.6 x embedding cosine + 0.4 x query-term overlap + category-intent boost;
  * dense   — embedding cosine only;
  * hybrid  — BM25 (lexical, rank_bm25) fused with dense cosine (semantic), min-max normalised per query.
The fusion weight, optional category boost and the "not found" (answerability) model are fitted on the validation
contracts of the CUAD retrieval benchmark by scripts/evaluate_models.py and stored in models/retrieval_config.json,
separately for each embedding backend (cosine scales differ between MiniLM and LSA). Without that file, defaults and
the original heuristic are used.
"""
from __future__ import annotations

import json
import math
import re
import threading

import numpy as np

from backend.config import settings

CONFIG_PATH = settings.MODELS_DIR / "retrieval_config.json"
# Original stop list: used by the original ranking and follow-up detection, which must behave exactly as before.
LEGACY_STOP = set("a an the is are was were be of to in on for and or what who whom which when where how does do did "
                  "can could should would will shall this that it its my me i you your there any about with by as at "
                  "from if under per than then clause clauses agreement document contract say says mention mentioned "
                  "there's".split())
# BM25 tokens also drop question words that never carry the answer ("How long...", "How many...", "Does it have...").
STOP = LEGACY_STOP | set("has have had get got much many long keep include included".split())
DEFAULTS = {"fusion": "weighted", "alpha": 0.5, "rrf_k": 60, "category_boost": 0.0, "answerability": None}
# The answerability model uses BM25 statistics, which are degenerate in a tiny collection (in a one-clause document
# every term has a negative idf). It was calibrated on contracts with ~70 clauses; below this size the original rule
# (some query-term overlap or a clearly similar clause) decides instead.
MIN_CLAUSES_FOR_CALIBRATED_NOT_FOUND = 5
_config_cache: dict | None = None
_lock = threading.Lock()


# ------------------------------------------------------------------------------------------ text helpers

def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in LEGACY_STOP}


def stem(w: str) -> str:
    return re.sub(r"(ation|ions|ing|ed|es|s)$", "", w)


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lower-case words and numbers, stop words removed, light suffix stripping."""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and (len(w) > 2 or w.isdigit())]


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over the clauses of one document. Uses rank_bm25 when installed, otherwise the
    same formula (including rank_bm25's epsilon floor for negative idf). Scores are floored at 0: in a tiny document,
    terms present in most clauses get a negative idf, which would otherwise make matches score below non-matches."""

    def __init__(self, docs_tokens: list[list[str]]):
        self.n = len(docs_tokens)
        self.empty = not any(docs_tokens)
        self._bm = None
        if self.empty:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm = BM25Okapi(docs_tokens)
        except ImportError:
            self.docs = docs_tokens
            self.avgdl = sum(map(len, docs_tokens)) / self.n
            df: dict[str, int] = {}
            for d in docs_tokens:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1
            idf = {t: math.log(self.n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
            floor = 0.25 * (sum(idf.values()) / len(idf))
            self.idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}
            self.tf = []
            for d in docs_tokens:
                tf: dict[str, int] = {}
                for t in d:
                    tf[t] = tf.get(t, 0) + 1
                self.tf.append(tf)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        if self.empty or not query_tokens:
            return np.zeros(self.n, dtype=np.float32)
        if self._bm is not None:
            return np.maximum(np.asarray(self._bm.get_scores(query_tokens), dtype=np.float32), 0.0)
        out = np.zeros(self.n, dtype=np.float32)
        for i, (d, tf) in enumerate(zip(self.docs, self.tf)):
            for q in query_tokens:
                f = tf.get(q, 0)
                if f:
                    out[i] += self.idf.get(q, 0.0) * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(d) / self.avgdl))
        return np.maximum(out, 0.0)


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> np.ndarray:
    return BM25Index(docs_tokens).scores(query_tokens)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min(initial=0.0)), float(x.max(initial=0.0))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def fuse(dense: np.ndarray, lexical: np.ndarray, fusion: str = "weighted", alpha: float = 0.5, rrf_k: int = 60) -> np.ndarray:
    if fusion == "rrf":
        out = np.zeros(len(dense), dtype=np.float32)
        for scores in (dense, lexical):
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            out += 1.0 / (rrf_k + ranks)
        return out
    return alpha * minmax(dense) + (1 - alpha) * minmax(lexical)


def answerability_features(q_tokens: list[str], dense: np.ndarray, lexical: np.ndarray, docs_tokens: list[list[str]],
                           best: int) -> list[float]:
    """[top dense cosine, top BM25 / number of query terms, share of query terms found in the best clause]."""
    if len(dense) == 0:
        return [0.0, 0.0, 0.0]
    nq = max(1, len(set(q_tokens)))
    coverage = len(set(q_tokens) & set(docs_tokens[best])) / nq if q_tokens else 0.0
    return [float(dense.max()), float(lexical.max()) / nq, float(coverage)]


# ------------------------------------------------------------------------------------------------ config

def load_config(embedder: str) -> dict:
    global _config_cache
    with _lock:
        if _config_cache is None:
            try:
                _config_cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
            except (OSError, ValueError):
                _config_cache = {}
    return {**DEFAULTS, **_config_cache.get("backends", {}).get(embedder, {})}


def reload_config() -> None:
    global _config_cache
    with _lock:
        _config_cache = None


def p_answerable(features: list[float], model: dict | None) -> float | None:
    if not model:
        return None
    z = model["intercept"] + sum(c * (f - m) / s for c, f, m, s in zip(model["coef"], features, model["mean"], model["scale"]))
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------------------------------------------------------------------------------ rankers

def legacy_scores(query: str, q_emb: np.ndarray, clause_emb: np.ndarray, clauses: list[dict],
                  intent: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scores of the original LegalLens ranking: 0.6 x cosine + 0.4 x query-term overlap + 0.25 category-intent boost."""
    sims = clause_emb @ q_emb
    q_terms = {stem(t) for t in terms(query)}
    scores, overlaps = np.zeros(len(clauses), dtype=np.float32), np.zeros(len(clauses), dtype=np.float32)
    for i, (s, c) in enumerate(zip(sims, clauses)):
        c_terms = {stem(t) for t in terms(f"{c.get('title') or ''} {c['text']} {c.get('category', '')}")}
        overlaps[i] = len(q_terms & c_terms) / (len(q_terms) or 1)
        boost = 0.25 if intent and c.get("category") == intent else 0.0
        scores[i] = round(float(s) * 0.6 + float(overlaps[i]) * 0.4 + boost, 4)
    return scores, overlaps


def legacy_rank(query: str, q_emb: np.ndarray, clause_emb: np.ndarray, clauses: list[dict], k: int = 3,
                intent: str | None = None) -> list[tuple[float, int]]:
    """The original LegalLens ranking, unchanged (kept as the baseline and as QA_RETRIEVAL=legacy)."""
    scores, overlaps = legacy_scores(query, q_emb, clause_emb, clauses, intent)
    order = sorted(range(len(clauses)), key=lambda i: -scores[i])[:k]
    return [(float(scores[i]), i) for i in order if overlaps[i] > 0 or scores[i] > 0.35]


def rank(query: str, q_emb: np.ndarray, clause_emb: np.ndarray, clauses: list[dict], embedder: str, k: int = 3,
         mode: str | None = None, intent: str | None = None, config: dict | None = None) -> dict:
    """Rank the document's clauses for a question.
    Returns {"ranked": [(score, index, {"dense", "bm25"})], "answerable": bool, "p_answerable": float|None, "method"}."""
    mode = (mode or settings.QA_RETRIEVAL).lower()
    if not clauses:
        return {"ranked": [], "answerable": False, "p_answerable": None, "method": f"{mode} retrieval"}
    if mode == "legacy":
        ranked = legacy_rank(query, q_emb, clause_emb, clauses, k, intent)
        return {"ranked": [(s, i, {}) for s, i in ranked], "answerable": bool(ranked), "p_answerable": None,
                "method": f"legacy ranking (embedding {embedder} + keyword overlap)"}
    cfg = config or load_config(embedder)
    dense = np.asarray(clause_emb @ q_emb, dtype=np.float32)
    docs_tokens = [tokenize(f"{c.get('title') or ''} {c['text']}") for c in clauses]
    q_tokens = tokenize(query)
    lexical = BM25Index(docs_tokens).scores(q_tokens)
    fused = combined_scores(dense, lexical, [c.get("category") for c in clauses], intent, cfg, mode)
    method = f"dense retrieval ({embedder})" if mode == "dense" else f"hybrid retrieval (BM25 + {embedder})"
    order = np.argsort(-fused, kind="stable")[:k]
    if len(clauses) < MIN_CLAUSES_FOR_CALIBRATED_NOT_FOUND:
        cfg = {**cfg, "answerability": None}
    answerable, prob = decide_answerable(answerability_features(q_tokens, dense, lexical, docs_tokens, int(order[0])), cfg)
    ranked = [(round(float(fused[i]), 4), int(i), {"dense": round(float(dense[i]), 4), "bm25": round(float(lexical[i]), 3)})
              for i in order]
    return {"ranked": ranked, "answerable": answerable, "p_answerable": None if prob is None else round(prob, 3),
            "method": method}


def combined_scores(dense: np.ndarray, lexical: np.ndarray, categories: list[str | None], intent: str | None,
                    cfg: dict, mode: str = "hybrid") -> np.ndarray:
    """Final ranking scores (shared by the app and scripts/evaluate_models.py)."""
    fused = dense.copy() if mode == "dense" else fuse(dense, lexical, cfg["fusion"], cfg["alpha"], cfg["rrf_k"])
    if cfg.get("category_boost") and intent:
        fused = fused + np.array([cfg["category_boost"] if c == intent else 0.0 for c in categories], dtype=np.float32)
    return fused


def decide_answerable(features: list[float], cfg: dict) -> tuple[bool, float | None]:
    prob = p_answerable(features, cfg.get("answerability"))
    if prob is None:  # uncalibrated: the original rule (some query-term overlap or a clearly similar clause)
        return features[2] > 0 or features[0] > 0.35, None
    return prob >= cfg["answerability"]["threshold"], prob
