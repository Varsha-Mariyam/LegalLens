"""Clause classification (guide section 8).

Three classifiers, as in the guide:
  1. keyword/rule baseline,
  2. embedding similarity against category descriptions,
  3. a trained ML classifier (TF-IDF + Logistic Regression) — trained by scripts/train_classifier.py.
The final label uses the trained model when it is available and confident, and otherwise falls back to
the baseline methods. All three results are kept so they can be shown and evaluated.
"""
from __future__ import annotations

import logging
import re
import threading

import numpy as np

from backend.config import settings
from backend.services import embedding_service

CATEGORIES = settings.CATEGORIES
MODEL_PATH = settings.MODELS_DIR / "clause_classifier.joblib"

KEYWORDS: dict[str, list[str]] = {
    "Payment": ["salary", "payment", "pay", "paid", "fee", "fees", "compensation", "remuneration", "wages", "rent",
                "invoice", "price", "bonus", "reimburse", "royalt", "deposit", "instalment", "installment",
                "consideration", "amount", "rupees", "rs.", "inr", "revenue", "commission", "interest"],
    "Confidentiality": ["confidential", "confidentiality", "non-disclosure", "disclose", "disclosure", "secret",
                        "proprietary", "trade secret", "privacy", "non-public"],
    "Termination": ["terminate", "termination", "notice period", "resign", "resignation", "expiry", "expire",
                    "cancel", "lock-in", "vacate", "notice", "renewal", "renew"],
    "Liability": ["liable", "liability", "damages", "indemnify", "indemnity", "indemnification", "loss", "losses",
                  "penalty", "liquidated", "compensate", "warranty", "consequential"],
    "Dispute": ["dispute", "arbitration", "arbitrator", "jurisdiction", "court", "courts", "governing law",
                "governed by", "mediation", "tribunal", "venue", "seat of arbitration", "laws of"],
}

CATEGORY_DESCRIPTIONS: dict[str, list[str]] = {
    "Payment": ["The employee shall receive a monthly salary paid by bank transfer.",
                "The client shall pay the fees within thirty days of the invoice.",
                "The tenant shall pay monthly rent and a security deposit."],
    "Confidentiality": ["The receiving party shall keep confidential information secret and not disclose it.",
                        "The employee shall not disclose trade secrets or proprietary information."],
    "Termination": ["Either party may terminate this agreement by giving written notice.",
                    "The agreement may be terminated or will expire at the end of the term; notice period for resignation."],
    "Liability": ["A party shall be liable for damages and losses and shall indemnify the other party.",
                  "Total liability shall not exceed the fees paid; no liability for consequential damages."],
    "Dispute": ["Any dispute shall be resolved by arbitration under the Arbitration and Conciliation Act.",
                "This agreement is governed by the laws of India and the courts shall have exclusive jurisdiction."],
    "Others": ["The employee shall report to the manager and perform assigned duties.",
               "This agreement may not be assigned; notices shall be sent to the address; entire agreement.",
               "The parties to this agreement and the definitions used in it."],
}

_lock = threading.Lock()
_model = None
_model_loaded = False
_load_error: str | None = None
_desc_matrix = None
_desc_labels: list[str] = []


def keyword_scores(text: str, title: str | None = None) -> dict[str, float]:
    low = text.lower()
    tlow = (title or "").lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, words in KEYWORDS.items():
        for w in words:
            pattern = r"\b" + re.escape(w)
            scores[cat] += len(re.findall(pattern, low))
            if tlow and re.search(pattern, tlow):
                scores[cat] += 3.0
    return scores


def keyword_classify(text: str, title: str | None = None) -> tuple[str, float]:
    scores = keyword_scores(text, title)
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[best] == 0:
        return "Others", 0.0
    return best, round(scores[best] / total, 3)


def _descriptions():
    global _desc_matrix, _desc_labels
    if _desc_matrix is None:
        labels, texts = [], []
        for cat, descs in CATEGORY_DESCRIPTIONS.items():
            for d in descs:
                labels.append(cat)
                texts.append(d)
        _desc_matrix = embedding_service.encode(texts)
        _desc_labels = labels
    return _desc_matrix, _desc_labels


def embedding_classify_batch(texts: list[str]) -> list[tuple[str, float]]:
    mat, labels = _descriptions()
    sims = embedding_service.encode(texts) @ mat.T
    out = []
    for row in sims:
        per_cat: dict[str, float] = {}
        for lab, s in zip(labels, row):
            per_cat[lab] = max(per_cat.get(lab, -1.0), float(s))
        best = max(per_cat, key=per_cat.get)
        out.append((best, round(per_cat[best], 3)))
    return out


def load_model():
    global _model, _model_loaded
    with _lock:
        if not _model_loaded:
            _model_loaded = True
            if MODEL_PATH.exists():
                import joblib
                try:
                    _model = joblib.load(MODEL_PATH)
                except Exception as exc:  # noqa: BLE001  (e.g. pickle from a different scikit-learn version)
                    global _load_error
                    _load_error = (f"Could not load {MODEL_PATH.name} ({exc}). Re-train with "
                                   "`python scripts/train_classifier.py`; using keyword/embedding baselines meanwhile.")
                    logging.getLogger("legallens.classifier").warning(_load_error)
                    _model = None
    return _model


def reload_model():
    global _model_loaded, _model
    _model_loaded, _model = False, None
    return load_model()


def ml_classify_batch(texts: list[str]) -> list[tuple[str, float]] | None:
    model = load_model()
    if model is None:
        return None
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    return [(classes[int(np.argmax(p))], round(float(np.max(p)), 3)) for p in probs]


def classify_batch(items: list[dict]) -> list[dict]:
    """items: [{"text", "title"}] -> [{"category", "confidence", "method", "details"}]."""
    if not items:
        return []
    texts = [f"{(i.get('title') or '')} {i['text']}".strip() for i in items]
    emb = embedding_classify_batch(texts)
    ml = ml_classify_batch(texts)
    results = []
    for idx, item in enumerate(items):
        kw_label, kw_conf = keyword_classify(item["text"], item.get("title"))
        title_label, _ = keyword_classify(item.get("title") or "", None) if item.get("title") else ("Others", 0.0)
        details = {"keyword": {"label": kw_label, "confidence": kw_conf},
                   "embedding": {"label": emb[idx][0], "similarity": emb[idx][1]}}
        if ml:
            details["ml"] = {"label": ml[idx][0], "confidence": ml[idx][1]}
            label, conf, method = ml[idx][0], ml[idx][1], "ml:tfidf-logreg"
            if conf < 0.45 and item.get("title") and title_label != "Others":
                label, conf, method = title_label, conf, "heading-keyword (low ML confidence)"
        else:
            if kw_conf > 0:
                label, conf, method = kw_label, kw_conf, "keyword-baseline"
            else:
                label, conf, method = emb[idx][0], emb[idx][1], "embedding-similarity"
        title_low = (item.get("title") or "").lower()
        if title_low.startswith("preamble") or title_low == "execution":
            label, method = "Others", "structural (preamble/execution)"
        results.append({"category": label, "confidence": conf, "method": method, "details": details})
    return results


def status() -> dict:
    load_model()
    return {"ml_model": _model is not None, "model_path": str(MODEL_PATH.relative_to(settings.ROOT)),
            "load_error": _load_error}
