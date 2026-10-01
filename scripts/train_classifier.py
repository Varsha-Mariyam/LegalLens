"""Train and evaluate the classical clause classifiers (guide section 8).

Evaluates, on the same held-out test split:
  * keyword baseline
  * embedding similarity to category descriptions (active embedding backend)
  * TF-IDF + Logistic Regression
  * TF-IDF + Linear SVM
Saves the Logistic Regression pipeline (it provides probabilities for confidence scores) to
models/clause_classifier.joblib — the fallback when the fine-tuned Transformer is unavailable — and all metrics to
models/classifier_metrics.json. Metrics are reported overall and on the gold-label subset (CUAD expert labels),
excluding heading-derived weak labels, so the weak labels cannot inflate the reported numbers.

Datasets (--dataset):
  contract_split (default)  data/datasets/contract_split/, split by contract (scripts/prepare_datasets.py)
  legacy                    the original random clause-level split (data/datasets/clause_train.csv / clause_test.csv)
Hyper-parameters are fixed (no tuning on the test split). scripts/evaluate_models.py adds the Transformer to the report.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.config import settings
from backend.services import classifier, embedding_service

LABELS = settings.CATEGORIES


DATASET = "contract_split"


def load(name: str) -> list[dict]:
    folder = settings.DATASETS_DIR / "contract_split" if DATASET == "contract_split" else settings.DATASETS_DIR
    path = folder / f"clause_{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/prepare_datasets.py (or use --dataset legacy)")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
        "n": len(y_true),
    }


def make_pipeline(kind: str) -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")
    else:
        clf = LinearSVC(C=0.8, class_weight="balanced")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main() -> None:
    train, test = load("train"), load("test")
    X_train, y_train = [r["clause"] for r in train], [r["label"] for r in train]
    X_test, y_test = [r["clause"] for r in test], [r["label"] for r in test]
    gold_idx = [i for i, r in enumerate(test) if r["source"] != "cuad_heading_weak"]

    preds: dict[str, list[str]] = {}
    preds["keyword_baseline"] = [classifier.keyword_classify(t)[0] for t in X_test]
    preds["embedding_similarity"] = [lab for lab, _ in classifier.embedding_classify_batch(X_test)]
    models = {}
    for kind in ("logreg", "linear_svm"):
        pipe = make_pipeline(kind).fit(X_train, y_train)
        models[kind] = pipe
        preds[f"tfidf_{kind}"] = list(pipe.predict(X_test))

    report = {"trained_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET, "train_size": len(train),
              "test_size": len(test), "gold_test_size": len(gold_idx),
              "test_contracts": len({r["contract"] for r in test}) if "contract" in test[0] else None,
              "split": ("by contract (filing groups), see data/datasets/contract_split/dataset_card.json"
                        if DATASET == "contract_split" else "random by clause (original)"),
              "embedding_backend": embedding_service.status()["backend"], "methods": {}}
    print(f"train={len(train)} test={len(test)} (gold subset={len(gold_idx)})")
    print(f"{'method':28s} {'acc':>7s} {'macroF1':>8s} | {'gold acc':>8s} {'gold F1':>8s}")
    for name, p in preds.items():
        full = evaluate(y_test, p)
        gold = evaluate([y_test[i] for i in gold_idx], [p[i] for i in gold_idx])
        report["methods"][name] = {"all": full, "gold_only": gold}
        print(f"{name:28s} {full['accuracy']:7.3f} {full['macro_f1']:8.3f} | {gold['accuracy']:8.3f} {gold['macro_f1']:8.3f}")

    final = models["logreg"]
    joblib.dump(final, settings.MODELS_DIR / "clause_classifier.joblib")
    report["deployed_model"] = "tfidf_logreg"
    (settings.MODELS_DIR / "classifier_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved models/clause_classifier.joblib and models/classifier_metrics.json")
    classifier.reload_model()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["contract_split", "legacy"], default="contract_split")
    DATASET = ap.parse_args().dataset
    main()
    from refresh_models import record_versions
    record_versions()
