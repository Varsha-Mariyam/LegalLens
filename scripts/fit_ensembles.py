"""Fit the deployed ensembles (fine-tuned Transformer + TF-IDF Logistic Regression) on the VALIDATION split only.

Why: on validation, averaging the two models' probabilities beats either model alone for both tasks (the lexical
model and the Transformer make different mistakes). This script
  1. trains the TF-IDF + one-vs-rest Logistic Regression risk model on the training contracts
     -> models/risk_logreg.joblib (also the risk detector's fallback when the Transformer is unavailable);
  2. chooses the Transformer weight w in {0, 0.1, ..., 1} for p = w * p_transformer + (1 - w) * p_logreg by validation
     macro F1 (clauses) and validation macro average precision (risk);
  3. tunes per-type risk thresholds on validation for the ensemble and for the Logistic Regression model alone;
  -> models/ensemble_config.json. The test split is not read.
Run after train_classifier.py and train_transformer.py. Usage: python scripts/fit_ensembles.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from backend.config import settings
from backend.services import transformer_models as tm
from train_transformer import risk_metrics, tune_thresholds

DATA = settings.DATASETS_DIR / "contract_split"
GRID = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]


def rows(name):
    with (DATA / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def risk_logreg_pipeline() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)),
                     ("clf", OneVsRestClassifier(LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")))])


def main() -> None:
    labels_c = list(settings.CATEGORIES)
    # ---- clause: deployed Logistic Regression (models/clause_classifier.joblib) + deployed Transformer
    lr = joblib.load(settings.MODELS_DIR / "clause_classifier.joblib")
    val = rows("clause_val.csv")
    Xv, yv = [r["clause"] for r in val], [r["label"] for r in val]
    P_lr = lr.predict_proba(Xv)[:, [list(lr.classes_).index(c) for c in labels_c]]
    clause_model = tm.TransformerModel(tm.model_dir("clause"))
    P_tr = clause_model.predict_proba(Xv, batch_size=64)[:, [clause_model.labels.index(c) for c in labels_c]]
    clause_grid = {w: round(float(f1_score(yv, [labels_c[i] for i in (w * P_tr + (1 - w) * P_lr).argmax(1)],
                                              average="macro")), 4) for w in GRID}
    w_clause = max(clause_grid, key=lambda w: (clause_grid[w], w))  # ties -> more Transformer weight
    print("[clause] validation macro F1 by Transformer weight:", clause_grid, "-> w =", w_clause)

    # ---- risk: train the TF-IDF one-vs-rest model, then weight + thresholds on validation
    labels_r = json.loads((DATA / "dataset_card.json").read_text(encoding="utf-8"))["risk_detection"]["labels"]

    def load(split):
        rs = rows(f"risk_{split}.csv")
        return ([tm.risk_input(r["title"], r["text"]) for r in rs],
                np.array([[int(l in r["labels"].split("|")) for l in labels_r] for r in rs]))

    Xt, Yt = load("train")
    Xr, Yr = load("val")
    pipe = risk_logreg_pipeline().fit(Xt, Yt)
    joblib.dump({"pipeline": pipe, "labels": labels_r}, settings.MODELS_DIR / "risk_logreg.joblib")
    R_lr = pipe.predict_proba(Xr)
    risk_model = tm.TransformerModel(tm.model_dir("risk"))
    R_tr = risk_model.predict_proba(Xr, batch_size=64)[:, [risk_model.labels.index(l) for l in labels_r]]
    have = [j for j in range(len(labels_r)) if Yr[:, j].sum()]

    def macro_ap(R):
        return round(float(np.mean([average_precision_score(Yr[:, j], R[:, j]) for j in have])), 4)

    risk_grid = {w: macro_ap(w * R_tr + (1 - w) * R_lr) for w in GRID}
    w_risk = max(risk_grid, key=lambda w: (risk_grid[w], w))
    R_ens = w_risk * R_tr + (1 - w_risk) * R_lr
    thr_ens, thr_lr = tune_thresholds(Yr, R_ens), tune_thresholds(Yr, R_lr)
    print("[risk] validation macro AP by Transformer weight:", risk_grid, "-> w =", w_risk)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fitted_on": "validation split only (data/datasets/contract_split/*_val.csv)",
        "clause": {"transformer_weight": w_clause, "selection_metric": "validation macro F1",
                   "validation_macro_f1_by_weight": {str(k): v for k, v in clause_grid.items()}},
        "risk": {"transformer_weight": w_risk, "selection_metric": "validation macro average precision",
                 "validation_macro_ap_by_weight": {str(k): v for k, v in risk_grid.items()},
                 "labels": labels_r, "thresholds": {l: float(t) for l, t in zip(labels_r, thr_ens)},
                 "logreg_thresholds": {l: float(t) for l, t in zip(labels_r, thr_lr)},
                 "validation_at_tuned_thresholds": {"ensemble": risk_metrics(Yr, R_ens, thr_ens),
                                                    "logreg": risk_metrics(Yr, R_lr, thr_lr)}},
    }
    (settings.MODELS_DIR / "ensemble_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("[done] models/ensemble_config.json, models/risk_logreg.joblib")


if __name__ == "__main__":
    main()
