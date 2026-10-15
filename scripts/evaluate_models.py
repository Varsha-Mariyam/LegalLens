"""Evaluate every model on the held-out TEST contracts (data/datasets/contract_split/) — run once, after training.

1. Clause classification (6 categories), all methods trained on the same training contracts:
   keyword baseline · embedding similarity (LSA fitted on training contracts; pretrained MiniLM) ·
   TF-IDF + Logistic Regression · TF-IDF + Linear SVM · fine-tuned Transformer(s).
   Accuracy, macro precision/recall/F1, weighted F1, per-class scores, confusion matrix, macro one-vs-rest ROC-AUC;
   95% bootstrap confidence intervals over test contracts and a paired bootstrap test Transformer vs LogReg;
   gold subset (expert labels only) and an out-of-domain check on the project's annotated reference templates.
2. Risk detection (12 CUAD types, multi-label): fine-tuned Transformer vs TF-IDF + one-vs-rest Logistic Regression
   (both with per-label thresholds tuned on validation) vs the original rule engine on the 6 types a rule targets.
3. Q&A retrieval (CUAD questions over the segmented test contracts): original LegalLens ranking (LSA and MiniLM),
   LSA, MiniLM, BM25 and hybrid BM25+MiniLM (fusion settings chosen on VALIDATION contracts). Hit@k, MRR@10,
   nDCG@5; "not found" (answerability) model fitted on validation, evaluated on test.
   Writes models/retrieval_config.json, which the app uses.
4. Inference speed.
Outputs: reports/evaluation/{metrics.json, RESULTS.md, *.png}; models/classifier_metrics.json (System page).
Usage: python scripts/evaluate_models.py [--bootstrap 1000] [--skip-speed]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import report_figures

from backend.config import settings
from backend.services import classifier, retrieval, risk_analyzer
from backend.services import transformer_models as tm
from backend.services.embedding_service import LSAEmbedder, _normalize

DATA = settings.DATASETS_DIR / "contract_split"
OUT = settings.REPORTS_DIR / "evaluation"
LABELS = list(settings.CATEGORIES)
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
ANSWER_RATE_TARGET = 0.90  # Q&A "not found" operating point (see fit_answerability)
# original rules that target a CUAD risk type (approximate correspondences, see docs/EVALUATION.md)
RULE_MAP = {"Uncapped Liability": {"unlimited_liability", "indirect_loss_included", "broad_indemnity"},
            "Non-Compete": {"post_term_restraint"}, "Liquidated Damages": {"penalty_or_forfeiture"},
            "Covenant Not To Sue": {"waiver_of_legal_remedy"}, "Renewal Term": {"auto_renewal"},
            "Termination For Convenience": {"termination_without_notice", "one_sided_termination"}}


def r4(x) -> float:
    return round(float(x), 4)


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cluster_bootstrap(groups: list[str], stat, n: int, seed: int = 42) -> list[float]:
    """Resample whole contracts with replacement; stat(indices) -> value."""
    uniq = sorted(set(groups))
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    idx_lists = [np.array(by[g]) for g in uniq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(stat(np.concatenate([idx_lists[j] for j in pick])))
    return out


def ci(values: list[float]) -> list[float]:
    return [r4(np.percentile(values, 2.5)), r4(np.percentile(values, 97.5))]


# ============================================================================ encoders

class Encoders:
    def __init__(self, train_texts: list[str]):
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.minilm = SentenceTransformer(MINILM)
        # LSA with the app's hyper-parameters, fitted on training-contract text only (no test text leaks in)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000, stop_words="english")
        X = self.vec.fit_transform(train_texts)
        self.svd = TruncatedSVD(n_components=256, random_state=42).fit(X)
        self.lsa_deployed = LSAEmbedder()  # the app's fallback model (used only to calibrate its retrieval settings)

    def encode(self, kind: str, texts: list[str]) -> np.ndarray:
        if kind == "minilm":
            return _normalize(self.minilm.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True))
        if kind == "lsa":
            return _normalize(self.svd.transform(self.vec.transform(texts)))
        return self.lsa_deployed.encode(texts)


def embedding_similarity(enc: Encoders, kind: str, texts: list[str]) -> tuple[list[str], np.ndarray]:
    labs, descs = [], []
    for cat, ds in classifier.CATEGORY_DESCRIPTIONS.items():
        labs += [cat] * len(ds)
        descs += ds
    S = enc.encode(kind, texts) @ enc.encode(kind, descs).T
    scores = np.stack([S[:, [i for i, l in enumerate(labs) if l == c]].max(1) for c in LABELS], 1)
    return [LABELS[i] for i in scores.argmax(1)], scores


# ============================================================================ 1. classification

def cls_metrics(y_true: list[str], y_pred: list[str], scores: np.ndarray | None = None) -> dict:
    """Macro averages run over the classes present in y_true (the expert-labelled subset has no Confidentiality
    examples — CUAD has no such label — and a class with no examples would otherwise add a meaningless 0)."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    present = [c for c in LABELS if c in set(y_true)]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    out = {"n": len(y_true), "classes_in_macro": present, "accuracy": r4(accuracy_score(y_true, y_pred)),
           "macro_precision": r4(p), "macro_recall": r4(r), "macro_f1": r4(f),
           "weighted_f1": r4(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
           "per_class": {k: ({kk: r4(vv) for kk, vv in v.items()} if isinstance(v, dict) else r4(v)) for k, v in
                         classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0).items()},
           "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}}
    if scores is not None:
        Y = np.array([[int(t == c) for c in LABELS] for t in y_true])
        have = [j for j in range(len(LABELS)) if 0 < Y[:, j].sum() < len(Y)]
        out["roc_auc_macro_ovr"] = r4(np.mean([roc_auc_score(Y[:, j], scores[:, j]) for j in have]))
    return out


def evaluate_classification(enc: Encoders, n_boot: int) -> dict:
    from sklearn.metrics import f1_score
    from train_classifier import make_pipeline
    train, val, test = read_csv("clause_train.csv"), read_csv("clause_val.csv"), read_csv("clause_test.csv")
    X_tr, y_tr = [r["clause"] for r in train], [r["label"] for r in train]
    X_va, y_va = [r["clause"] for r in val], [r["label"] for r in val]
    X_te, y_te = [r["clause"] for r in test], [r["label"] for r in test]
    val_scores = {}  # validation macro F1 — the basis for choosing which classifier the app deploys
    groups = [r["contract"] for r in test]
    gold = np.array([r["source"] == "cuad_v1" for r in test])

    preds: dict[str, tuple[list[str], np.ndarray | None]] = {}
    preds["keyword_baseline"] = ([classifier.keyword_classify(t)[0] for t in X_te], None)
    preds["embedding_similarity_lsa"] = embedding_similarity(enc, "lsa", X_te)
    preds["embedding_similarity_minilm"] = embedding_similarity(enc, "minilm", X_te)
    fitted = {}
    for kind in ("logreg", "linear_svm"):
        pipe = make_pipeline(kind).fit(X_tr, y_tr)
        fitted[kind] = pipe
        classes = list(pipe.classes_)
        raw = pipe.predict_proba(X_te) if kind == "logreg" else pipe.decision_function(X_te)
        scores = raw[:, [classes.index(c) for c in LABELS]]
        preds[f"tfidf_{kind}"] = ([LABELS[i] for i in scores.argmax(1)], scores)
        val_scores[f"tfidf_{kind}"] = cls_metrics(y_va, list(pipe.predict(X_va)))
    transformers = {"transformer_minilm": tm.model_dir("clause")}
    for p in sorted((settings.TRANSFORMER_DIR / "experiments").glob("*_clause")):
        transformers[f"transformer_{p.name.removesuffix('_clause')}"] = p
    model_info = {}
    for name, path in transformers.items():
        if not (path / tm.CONFIG_FILE).exists():
            continue
        m = tm.TransformerModel(path)
        probs = m.predict_proba(X_te, batch_size=64)[:, [m.labels.index(c) for c in LABELS]]
        preds[name] = ([LABELS[i] for i in probs.argmax(1)], probs)
        vp = m.predict_proba(X_va, batch_size=64)
        val_scores[name] = cls_metrics(y_va, [m.labels[int(i)] for i in vp.argmax(1)])
        model_info[name] = {"base_model": m.config.get("base_model"), "path": str(path.relative_to(settings.ROOT)),
                            "best_epoch": m.config.get("best_epoch"), "validation": m.config.get("validation_metrics"),
                            "deployed": path == tm.model_dir("clause")}
        del m

    ens_cfg = tm.ensemble_config().get("clause", {})
    if "transformer_minilm" in preds and ens_cfg.get("transformer_weight") is not None:
        import joblib
        w = float(ens_cfg["transformer_weight"])
        lr = joblib.load(settings.MODELS_DIR / "clause_classifier.joblib")  # the deployed fallback model
        cols = [list(lr.classes_).index(c) for c in LABELS]
        m = tm.TransformerModel(tm.model_dir("clause"))
        P_te = w * preds["transformer_minilm"][1] + (1 - w) * lr.predict_proba(X_te)[:, cols]
        P_va = w * m.predict_proba(X_va, batch_size=64)[:, [m.labels.index(c) for c in LABELS]] + (1 - w) * lr.predict_proba(X_va)[:, cols]
        preds["ensemble_minilm_logreg"] = ([LABELS[i] for i in P_te.argmax(1)], P_te)
        val_scores["ensemble_minilm_logreg"] = cls_metrics(y_va, [LABELS[i] for i in P_va.argmax(1)])
        ensemble_deployed = settings.CLASSIFIER_BACKEND == "auto"  # legacy mode; the default deploys MiniLM alone
        model_info["ensemble_minilm_logreg"] = {"transformer_weight": w, "deployed": ensemble_deployed,
                                                "selected_on": ens_cfg.get("selection_metric")}
        model_info["transformer_minilm"]["deployed"] = not ensemble_deployed
        del m
    for name in ("keyword_baseline",):
        val_scores[name] = cls_metrics(y_va, [classifier.keyword_classify(t)[0] for t in X_va])
    y_arr = np.array(y_te)
    results = {}
    for name, (pred, scores) in preds.items():
        pred_arr = np.array(pred)
        entry = {"validation": {k: val_scores[name][k] for k in ("accuracy", "macro_f1")} if name in val_scores else None,
                 "all": cls_metrics(y_te, pred, scores),
                 "gold_only": cls_metrics(list(y_arr[gold]), list(pred_arr[gold]), None if scores is None else scores[gold])}
        boot = cluster_bootstrap(groups, lambda idx: (float((pred_arr[idx] == y_arr[idx]).mean()),
                                                      f1_score(y_arr[idx], pred_arr[idx], labels=LABELS, average="macro",
                                                               zero_division=0)), n_boot)
        entry["all"]["accuracy_ci95"] = ci([b[0] for b in boot])
        entry["all"]["macro_f1_ci95"] = ci([b[1] for b in boot])
        results[name] = entry
        print(f"  {name:32s} acc {entry['all']['accuracy']:.3f} {entry['all']['accuracy_ci95']}  "
              f"macroF1 {entry['all']['macro_f1']:.3f} {entry['all']['macro_f1_ci95']}  gold F1 {entry['gold_only']['macro_f1']:.3f}")

    paired = []
    for x, y in (("transformer_minilm", "tfidf_logreg"), ("ensemble_minilm_logreg", "tfidf_logreg"),
                 ("ensemble_minilm_logreg", "transformer_minilm"), ("ensemble_minilm_logreg", "tfidf_linear_svm")):
        if x not in preds or y not in preds:
            continue
        a, b = np.array(preds[x][0]), np.array(preds[y][0])
        diffs = cluster_bootstrap(groups, lambda idx: f1_score(y_arr[idx], a[idx], labels=LABELS, average="macro", zero_division=0)
                                  - f1_score(y_arr[idx], b[idx], labels=LABELS, average="macro", zero_division=0), n_boot, seed=7)
        paired.append({"comparison": f"{x} - {y} (macro F1, all test clauses)",
                       "observed": r4(results[x]["all"]["macro_f1"] - results[y]["all"]["macro_f1"]),
                       "ci95": ci(diffs), "share_of_resamples_not_better": r4(np.mean(np.array(diffs) <= 0))})
        print(f"  paired {paired[-1]}")
    return {"test_size": len(test), "gold_test_size": int(gold.sum()), "train_size": len(train),
            "test_contracts": len(set(groups)), "methods": results, "paired_bootstrap": paired,
            "transformer_models": model_info, "ood_reference_templates": evaluate_ood(fitted["logreg"]), "_preds": preds,
            "_y": y_te}


def evaluate_ood(logreg) -> dict:
    """Out-of-domain check: the project's Indian reference templates, whose clause categories were annotated by hand."""
    from backend.services import clause_service, extraction_service
    ann = json.loads((settings.STANDARD_DIR / "categories.json").read_text(encoding="utf-8"))
    items = []
    for doc_type, cats in ann.items():
        path = settings.STANDARD_DIR / f"standard_{doc_type}.txt"
        if not path.exists():
            continue
        for c in clause_service.segment(extraction_service.extract(str(path))["text"]):
            if (c["title"] or "") in cats:
                items.append({"title": c["title"], "text": c["text"], "label": cats[c["title"]]})
    if not items:
        return {}
    texts = [f"{i['title']} {i['text']}" for i in items]
    y = [i["label"] for i in items]
    out = {"n": len(items), "note": "clauses of data/standard_documents/standard_*.txt with a manual category in categories.json"}
    out["keyword_baseline"] = r4(np.mean([classifier.keyword_classify(i["text"], i["title"])[0] == l for i, l in zip(items, y)]))
    out["tfidf_logreg"] = r4(np.mean([p == l for p, l in zip(logreg.predict(texts), y)]))
    m = tm.get("clause")
    if m is not None:
        probs = m.predict_proba(texts)[:, [m.labels.index(c) for c in LABELS]]
        out["transformer_minilm"] = r4(np.mean([LABELS[int(p.argmax())] == l for p, l in zip(probs, y)]))
        w = tm.ensemble_weight("clause")
        if w is not None:
            P = w * probs + (1 - w) * logreg.predict_proba(texts)[:, [list(logreg.classes_).index(c) for c in LABELS]]
            out["ensemble_minilm_logreg"] = r4(np.mean([LABELS[int(p.argmax())] == l for p, l in zip(P, y)]))
    return out


# ============================================================================ 2. risk detection

def risk_scores(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None, labels: list[str]) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
    per = {}
    for j, l in enumerate(labels):
        p, r, f, _ = precision_recall_fscore_support(y[:, j], pred[:, j], average="binary", zero_division=0)
        per[l] = {"precision": r4(p), "recall": r4(r), "f1": r4(f), "support": int(y[:, j].sum())}
        if probs is not None and 0 < y[:, j].sum() < len(y):
            per[l]["roc_auc"] = r4(roc_auc_score(y[:, j], probs[:, j]))
            per[l]["average_precision"] = r4(average_precision_score(y[:, j], probs[:, j]))
    out = {"micro_f1": r4(f1_score(y, pred, average="micro", zero_division=0)),
           "macro_f1": r4(f1_score(y, pred, average="macro", zero_division=0)),
           "micro_precision": r4(precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[0]),
           "micro_recall": r4(precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[1]),
           "per_label": per}
    if probs is not None:
        out["macro_roc_auc"] = r4(np.mean([v["roc_auc"] for v in per.values() if "roc_auc" in v]))
        out["macro_average_precision"] = r4(np.mean([v["average_precision"] for v in per.values() if "average_precision" in v]))
    any_true, any_pred = y.max(1), pred.max(1)
    p, r, f, _ = precision_recall_fscore_support(any_true, any_pred, average="binary", zero_division=0)
    out["any_risk_clause"] = {"precision": r4(p), "recall": r4(r), "f1": r4(f), "support": int(any_true.sum())}
    return out


def evaluate_risk(n_boot: int) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from train_transformer import tune_thresholds
    labels = json.loads((DATA / "dataset_card.json").read_text(encoding="utf-8"))["risk_detection"]["labels"]

    def load(split):
        rows = read_csv(f"risk_{split}.csv")
        X = [tm.risk_input(r["title"], r["text"]) for r in rows]
        Y = np.array([[int(l in r["labels"].split("|")) for l in labels] for r in rows])
        return rows, X, Y

    tr_rows, X_tr, Y_tr = load("train")
    va_rows, X_va, Y_va = load("val")
    te_rows, X_te, Y_te = load("test")
    groups = [r["contract"] for r in te_rows]
    methods, preds = {}, {}

    model = tm.TransformerModel(tm.model_dir("risk"))
    P = model.predict_proba(X_te, batch_size=64)[:, [model.labels.index(l) for l in labels]]
    thr = np.array([model.config["thresholds"][l] for l in labels])
    preds["transformer_minilm"] = (P >= thr).astype(int)
    methods["transformer_minilm"] = risk_scores(Y_te, preds["transformer_minilm"], P, labels)
    methods["transformer_minilm"]["thresholds"] = {l: r4(t) for l, t in zip(labels, thr)}
    methods["transformer_minilm"]["deployed"] = settings.RISK_MODEL != "auto"  # MiniLM alone is the default

    lr_model, ens = tm.risk_logreg(), tm.ensemble_config().get("risk", {})
    if lr_model is not None and ens.get("logreg_thresholds"):  # the fallback model (scripts/fit_ensembles.py)
        pipe, lr_labels = lr_model["pipeline"], lr_model["labels"]
        assert lr_labels == labels
        thr_lr = np.array([ens["logreg_thresholds"][l] for l in labels])
    else:
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)),
                         ("clf", OneVsRestClassifier(LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")))])
        pipe.fit(X_tr, Y_tr)
        thr_lr = tune_thresholds(Y_va, pipe.predict_proba(X_va))
    P_lr = pipe.predict_proba(X_te)
    preds["tfidf_logreg_ovr"] = (P_lr >= thr_lr).astype(int)
    methods["tfidf_logreg_ovr"] = risk_scores(Y_te, preds["tfidf_logreg_ovr"], P_lr, labels)
    methods["tfidf_logreg_ovr"]["thresholds"] = {l: r4(t) for l, t in zip(labels, thr_lr)}
    if ens.get("thresholds") and ens.get("transformer_weight") is not None:
        w = float(ens["transformer_weight"])
        P_ens = w * P + (1 - w) * P_lr
        thr_e = np.array([ens["thresholds"][l] for l in labels])
        preds["ensemble_minilm_logreg"] = (P_ens >= thr_e).astype(int)
        methods["ensemble_minilm_logreg"] = risk_scores(Y_te, preds["ensemble_minilm_logreg"], P_ens, labels)
        methods["ensemble_minilm_logreg"].update({"thresholds": {l: r4(t) for l, t in zip(labels, thr_e)},
                                                  "transformer_weight": w, "deployed": settings.RISK_MODEL == "auto"})

    # original rule engine: clause category from the deployed classifier, exactly as in the app
    cats = [c["category"] for c in classifier.classify_batch([{"text": r["text"], "title": r["title"] or None} for r in te_rows])]
    rule_pred = np.zeros_like(Y_te)
    for i, (r, cat) in enumerate(zip(te_rows, cats)):
        fired = {e["rule"] for e in risk_analyzer.rule_signals({"text": r["text"], "title": r["title"], "category": cat},
                                                              "service", None, {})}
        for j, l in enumerate(labels):
            if RULE_MAP.get(l, set()) & fired:
                rule_pred[i, j] = 1
    mapped = [j for j, l in enumerate(labels) if l in RULE_MAP]
    mapped_labels = [labels[j] for j in mapped]
    rule_subset = {}
    for name, pr in (("rules_original", rule_pred), *preds.items()):
        rule_subset[name] = risk_scores(Y_te[:, mapped], pr[:, mapped], None, mapped_labels)

    for name in preds:
        pr = preds[name]
        boot = cluster_bootstrap(groups, lambda idx: f1_score(Y_te[idx], pr[idx], average="micro", zero_division=0), n_boot)
        methods[name]["micro_f1_ci95"] = ci(boot)
    paired = []
    for x, y in (("transformer_minilm", "tfidf_logreg_ovr"), ("ensemble_minilm_logreg", "tfidf_logreg_ovr"),
                 ("ensemble_minilm_logreg", "transformer_minilm")):
        if x not in preds:
            continue
        a, b = preds[x], preds[y]
        diffs = cluster_bootstrap(groups, lambda idx: f1_score(Y_te[idx], a[idx], average="micro", zero_division=0)
                                  - f1_score(Y_te[idx], b[idx], average="micro", zero_division=0), n_boot, seed=7)
        paired.append({"comparison": f"{x} - {y} (micro F1)", "observed": r4(methods[x]["micro_f1"] - methods[y]["micro_f1"]),
                       "ci95": ci(diffs), "share_of_resamples_not_better": r4(np.mean(np.array(diffs) <= 0))})
        print(f"  paired {paired[-1]}")
    for name, m in methods.items():
        print(f"  {name:22s} micro F1 {m['micro_f1']:.3f} {m.get('micro_f1_ci95')}  macro F1 {m['macro_f1']:.3f}  "
              f"macro AUC {m.get('macro_roc_auc')}  macro AP {m.get('macro_average_precision')}")
    for name, m in rule_subset.items():
        print(f"  [6 rule-mapped types] {name:22s} micro F1 {m['micro_f1']:.3f} (P {m['micro_precision']:.3f} R {m['micro_recall']:.3f})")
    return {"labels": labels, "test_clauses": len(te_rows), "test_positive_clauses": int(Y_te.max(1).sum()),
            "test_contracts": len(set(groups)), "methods": methods, "rule_mapped_types": {"labels": mapped_labels,
            "rule_map": {k: sorted(v) for k, v in RULE_MAP.items()}, "methods": rule_subset},
            "paired_bootstrap": paired}


# ============================================================================ 3. retrieval

def retrieval_queries(split: str, enc: Encoders) -> list[dict]:
    from backend.services.qa_service import query_intent
    recs = [json.loads(l) for l in (DATA / f"retrieval_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    texts, spans, segs = [], [], []
    for rec in recs:
        seg_texts = [f"{s['title'] or ''} {s['text']}" for s in rec["segments"]]
        spans.append((len(texts), len(texts) + len(seg_texts)))
        texts += seg_texts
        segs += [{"text": s["text"], "title": s["title"]} for s in rec["segments"]]
    E = {k: enc.encode(k, texts) for k in ("minilm", "lsa", "lsa_deployed")}
    cats = [c["category"] for c in classifier.classify_batch(segs)]  # the app's deployed classifier, as in Q&A
    questions = sorted({q["question"] for rec in recs for q in rec["queries"]})
    Q = {k: dict(zip(questions, enc.encode(k, questions))) for k in E}
    q_tokens = {q: retrieval.tokenize(q) for q in questions}
    intents = {q: query_intent(q) for q in questions}
    out = []
    for rec, (a, b) in zip(recs, spans):
        clauses = [{"title": s["title"], "text": s["text"], "category": c} for s, c in zip(rec["segments"], cats[a:b])]
        docs_tokens = [retrieval.tokenize(t) for t in texts[a:b]]
        bm = retrieval.BM25Index(docs_tokens)
        id_to_idx = {s["id"]: i for i, s in enumerate(rec["segments"])}
        for q in rec["queries"]:
            qt = q["question"]
            item = {"contract": rec["contract"], "label": q["label"], "answerable": q["answerable"],
                    "relevant": {id_to_idx[i] for i in q["relevant"]}, "q_tokens": q_tokens[qt], "docs_tokens": docs_tokens,
                    "categories": [c["category"] for c in clauses], "intent": intents[qt], "lexical": bm.scores(q_tokens[qt])}
            for k in E:
                item[f"dense_{k}"] = E[k][a:b] @ Q[k][qt]
            for k in ("minilm", "lsa"):
                item[f"legacy_{k}"] = retrieval.legacy_scores(qt, Q[k][qt], E[k][a:b], clauses, intents[qt])
            out.append(item)
    return out


def ranking_metrics(items: list[dict], scorer) -> dict:
    hits = {1: [], 3: [], 5: []}
    mrr, ndcg = [], []
    for it in items:
        if not it["relevant"]:
            continue
        order = np.argsort(-scorer(it), kind="stable")
        rel = [int(i in it["relevant"]) for i in order]
        for k in hits:
            hits[k].append(float(any(rel[:k])))
        first = next((r for r, v in enumerate(rel[:10], 1) if v), None)
        mrr.append(1.0 / first if first else 0.0)
        dcg = sum(v / np.log2(r + 1) for r, v in enumerate(rel[:5], 1))
        idcg = sum(1 / np.log2(r + 1) for r in range(1, min(5, len(it["relevant"])) + 1))
        ndcg.append(dcg / idcg)
    return {"queries": len(mrr), "hit@1": r4(np.mean(hits[1])), "hit@3": r4(np.mean(hits[3])), "hit@5": r4(np.mean(hits[5])),
            "mrr@10": r4(np.mean(mrr)), "ndcg@5": r4(np.mean(ndcg))}


def reciprocal_rank(it: dict, scorer, cutoff: int = 10) -> float:
    order = np.argsort(-scorer(it), kind="stable")[:cutoff]
    return next((1.0 / r for r, i in enumerate(order, 1) if i in it["relevant"]), 0.0)


def hybrid_scorer(cfg: dict, dense_key: str):
    return lambda it: retrieval.combined_scores(it[dense_key], it["lexical"], it["categories"], it["intent"], cfg, "hybrid")


def tune_hybrid(val: list[dict], dense_key: str) -> tuple[dict, list[dict]]:
    grid = []
    for fusion, alphas in (("weighted", [round(a, 1) for a in np.arange(0, 1.01, 0.1)]), ("rrf", [0.5])):
        for alpha in alphas:
            for boost in (0.0, 0.1, 0.2, 0.3):
                cfg = {**retrieval.DEFAULTS, "fusion": fusion, "alpha": alpha, "category_boost": boost}
                grid.append((ranking_metrics(val, hybrid_scorer(cfg, dense_key))["mrr@10"], cfg))
    best = max(grid, key=lambda g: g[0])[1]
    return best, [{"fusion": c["fusion"], "alpha": c["alpha"], "category_boost": c["category_boost"], "val_mrr@10": m}
                  for m, c in grid]


def fit_answerability(val: list[dict], cfg: dict, dense_key: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    X, y = answer_features(val, cfg, dense_key), np.array([int(it["answerable"]) for it in val])
    mean, scale = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(max_iter=1000).fit((X - mean) / scale, y)
    model = {"coef": [r4(c) for c in lr.coef_[0]], "intercept": r4(lr.intercept_[0]), "mean": [r4(m) for m in mean],
             "scale": [r4(s) for s in scale], "features": ["top dense cosine", "top BM25 / query terms", "query-term coverage"]}
    probs = np.array([retrieval.p_answerable(list(x), model) for x in X])
    # Operating point: a wrong "not found" (the answer IS in the document) is the costlier error for an assistant, so
    # the threshold is the highest one that still answers >= 90% of answerable validation questions.
    grid = np.round(np.arange(0.01, 0.99, 0.005), 3)
    ok = [t for t in grid if np.mean(probs[y == 1] >= t) >= ANSWER_RATE_TARGET]
    t = max(ok) if ok else float(grid[0])
    model["threshold"] = r4(t)
    model["operating_point"] = f"answer >= {ANSWER_RATE_TARGET:.0%} of answerable validation questions"
    model["val_answered_when_answerable"] = r4(np.mean(probs[y == 1] >= t))
    model["val_not_found_when_unanswerable"] = r4(np.mean(probs[y == 0] < t))
    model["val_balanced_accuracy"] = r4(0.5 * (model["val_answered_when_answerable"] + model["val_not_found_when_unanswerable"]))
    return model


def answer_features(items: list[dict], cfg: dict, dense_key: str) -> np.ndarray:
    scorer = hybrid_scorer(cfg, dense_key)
    return np.array([retrieval.answerability_features(it["q_tokens"], it[dense_key], it["lexical"], it["docs_tokens"],
                                                      int(np.argmax(scorer(it)))) for it in items])


def decision_metrics(items: list[dict], answered: np.ndarray, ranker) -> dict:
    y = np.array([it["answerable"] for it in items])
    nf_pred, nf_true = ~answered, ~y
    tp = float((nf_pred & nf_true).sum())
    prec = tp / nf_pred.sum() if nf_pred.sum() else 0.0
    rec = tp / nf_true.sum() if nf_true.sum() else 0.0
    correct = []
    for it, a in zip(items, answered):
        if not it["answerable"]:
            correct.append(not a)
        elif it["relevant"]:
            top3 = set(np.argsort(-ranker(it), kind="stable")[:3].tolist())
            correct.append(bool(a) and bool(top3 & it["relevant"]))
    return {"accuracy": r4(np.mean(answered == y)), "balanced_accuracy": r4(0.5 * (np.mean(answered[y]) + np.mean(~answered[~y]))),
            "answered_when_answerable": r4(np.mean(answered[y])), "not_found_when_unanswerable": r4(np.mean(~answered[~y])),
            "not_found_precision": r4(prec), "not_found_recall": r4(rec),
            "not_found_f1": r4(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
            "end_to_end_correct": r4(np.mean(correct)), "end_to_end_note":
            "unanswerable: said 'not found'; answerable: answered with a relevant clause in the top 3"}


def evaluate_retrieval(enc: Encoders, n_boot: int) -> dict:
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    val, test = retrieval_queries("val", enc), retrieval_queries("test", enc)
    print(f"  prepared {len(val)} validation and {len(test)} test queries ({time.time() - t0:.0f}s)")
    best_m, grid_m = tune_hybrid(val, "dense_minilm")
    best_l, _ = tune_hybrid(val, "dense_lsa_deployed")
    rankers = {
        "legacy_lsa (original ranking, LSA)": lambda it: it["legacy_lsa"][0],
        "legacy_minilm (original ranking, MiniLM)": lambda it: it["legacy_minilm"][0],
        "lsa_dense": lambda it: it["dense_lsa"],
        "minilm_dense": lambda it: it["dense_minilm"],
        "bm25": lambda it: it["lexical"],
        "hybrid_bm25_minilm (deployed)": hybrid_scorer(best_m, "dense_minilm"),
        "hybrid_bm25_lsa": hybrid_scorer({**best_m}, "dense_lsa"),
    }
    results = {}
    groups = [it["contract"] for it in test]
    answerable_idx = [i for i, it in enumerate(test) if it["relevant"]]
    sub_groups = [groups[i] for i in answerable_idx]
    for name, fn in rankers.items():
        results[name] = {"val": ranking_metrics(val, fn), "test": ranking_metrics(test, fn)}
        rr = np.array([reciprocal_rank(test[i], fn) for i in answerable_idx])  # per-question MRR@10 terms
        boot = cluster_bootstrap(sub_groups, lambda idx: float(rr[idx].mean()), n_boot)
        results[name]["test"]["mrr@10_ci95"] = ci(boot)
        print(f"  {name:42s} test hit@1 {results[name]['test']['hit@1']:.3f} hit@3 {results[name]['test']['hit@3']:.3f} "
              f"MRR {results[name]['test']['mrr@10']:.3f} {results[name]['test']['mrr@10_ci95']}")

    ans_m = fit_answerability(val, best_m, "dense_minilm")
    ans_l = fit_answerability(val, best_l, "dense_lsa_deployed")
    X_te = answer_features(test, best_m, "dense_minilm")
    p_te = np.array([retrieval.p_answerable(list(x), ans_m) for x in X_te])
    y_te = np.array([it["answerable"] for it in test])
    heuristic = np.array([x[2] > 0 or x[0] > 0.35 for x in X_te])
    small = np.array([len(it["docs_tokens"]) < retrieval.MIN_CLAUSES_FOR_CALIBRATED_NOT_FOUND for it in test])
    calibrated = np.where(small, heuristic, p_te >= ans_m["threshold"])  # same rule as retrieval.rank()
    def legacy_answers(key: str) -> np.ndarray:  # legacy_rank's rule: a top-3 clause with term overlap or score > 0.35
        return np.array([any(it[key][1][i] > 0 or it[key][0][i] > 0.35 for i in np.argsort(-it[key][0], kind="stable")[:3])
                         for it in test])
    legacy, legacy_lsa = legacy_answers("legacy_minilm"), legacy_answers("legacy_lsa")
    hyb = hybrid_scorer(best_m, "dense_minilm")
    decisions = {
        "legacy_lsa (original)": decision_metrics(test, legacy_lsa, lambda it: it["legacy_lsa"][0]),
        "legacy_minilm": decision_metrics(test, legacy, lambda it: it["legacy_minilm"][0]),
        "hybrid + uncalibrated rule": decision_metrics(test, heuristic, hyb),
        "hybrid + calibrated answerability (deployed)": decision_metrics(test, calibrated, hyb),
    }
    decisions["hybrid + calibrated answerability (deployed)"]["roc_auc_p_answerable"] = r4(roc_auc_score(y_te, p_te))
    for name, d in decisions.items():
        print(f"  {name:45s} not-found recall {d['not_found_when_unanswerable']:.3f}  answered-when-answerable "
              f"{d['answered_when_answerable']:.3f}  end-to-end {d['end_to_end_correct']:.3f}")

    config = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "fitted_on": "CUAD validation contracts (data/datasets/contract_split/retrieval_val.jsonl)",
              "selection_metric": "validation MRR@10; answerability threshold maximises validation balanced accuracy",
              "backends": {"sbert-all-MiniLM-L6-v2": {k: best_m[k] for k in ("fusion", "alpha", "rrf_k", "category_boost")} | {"answerability": ans_m},
                           "lsa-tfidf-svd": {k: best_l[k] for k in ("fusion", "alpha", "rrf_k", "category_boost")} | {"answerability": ans_l}}}
    retrieval.CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    retrieval.reload_config()
    card = json.loads((DATA / "dataset_card.json").read_text(encoding="utf-8"))["retrieval"]
    return {"dataset": card, "rankers": results, "selected_config": config["backends"], "decisions": decisions,
            "val_grid_minilm": grid_m}


# ============================================================================ 4. speed

def evaluate_speed() -> dict:
    import torch
    texts = [r["clause"] for r in read_csv("clause_test.csv")][:512]
    out = {"clauses": len(texts)}
    from train_classifier import make_pipeline
    train = read_csv("clause_train.csv")
    pipe = make_pipeline("logreg").fit([r["clause"] for r in train], [r["label"] for r in train])
    t = time.perf_counter()
    pipe.predict_proba(texts)
    out["tfidf_logreg_cpu_ms_per_clause"] = r4(1000 * (time.perf_counter() - t) / len(texts))
    for device in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        m = tm.TransformerModel(tm.model_dir("clause"), device)
        m.predict_proba(texts[:32])
        t = time.perf_counter()
        m.predict_proba(texts, batch_size=32)
        out[f"transformer_minilm_{device}_ms_per_clause"] = r4(1000 * (time.perf_counter() - t) / len(texts))
        del m
    for exp in sorted((settings.TRANSFORMER_DIR / "experiments").glob("*_clause")):  # e.g. Legal-BERT (not deployed)
        if not (exp / tm.CONFIG_FILE).exists():
            continue
        sample = texts[:128]
        for device in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            m = tm.TransformerModel(exp, device)
            m.predict_proba(sample[:16])
            t = time.perf_counter()
            m.predict_proba(sample, batch_size=32)
            out[f"transformer_{exp.name.removesuffix('_clause')}_{device}_ms_per_clause"] = r4(1000 * (time.perf_counter() - t) / len(sample))
            del m
        out[f"transformer_{exp.name.removesuffix('_clause')}_encoder_mb"] = r4((exp / "encoder" / "model.safetensors").stat().st_size / 1e6)
    out["transformer_minilm_encoder_mb"] = r4((tm.model_dir("clause") / "encoder" / "model.safetensors").stat().st_size / 1e6)
    out["cpu"] = __import__("platform").processor()
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print("  ", out)
    return out


# ============================================================================ reporting

def training_history(task: str) -> dict:
    d = tm.model_dir(task)
    cfg = json.loads((d / tm.CONFIG_FILE).read_text(encoding="utf-8"))
    steps = [json.loads(l) for l in (d / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"history": cfg.get("history", []), "steps": [s for s in steps if "loss" in s], "best_epoch": cfg.get("best_epoch"),
            "training": cfg.get("training", {}), "validation": cfg.get("validation_metrics")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lsa_corpus = [r["clause"] for r in read_csv("clause_train.csv")] + [r["text"] for r in read_csv("risk_train.csv")]
    enc = Encoders(lsa_corpus)
    print("[1/4] clause classification")
    cls = evaluate_classification(enc, args.bootstrap)
    print("[2/4] risk detection")
    risk = evaluate_risk(args.bootstrap)
    print("[3/4] retrieval")
    ret = evaluate_retrieval(enc, args.bootstrap)
    speed = {} if args.skip_speed else (print("[4/4] speed") or evaluate_speed())
    history = {t: training_history(t) for t in ("clause", "risk")}
    preds, y_true = cls.pop("_preds"), cls.pop("_y")
    card = json.loads((DATA / "dataset_card.json").read_text(encoding="utf-8"))
    metrics = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset_card": card,
               "classification": cls, "risk": risk, "retrieval": ret, "speed": speed, "training": history,
               "bootstrap_resamples": args.bootstrap, "seconds": round(time.time() - t0)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    # System page / backwards-compatible summary
    ui = {"trained_at": metrics["created_at"], "dataset": "contract_split", "split": card["split_method"],
          "train_size": cls["train_size"], "test_size": cls["test_size"], "gold_test_size": cls["gold_test_size"],
          "test_contracts": cls["test_contracts"], "embedding_backend": "see method names",
          "methods": {k: {"all": {kk: v["all"][kk] for kk in ("accuracy", "macro_f1", "macro_precision", "macro_recall",
                                                               "accuracy_ci95", "macro_f1_ci95", "per_class", "confusion_matrix", "n")}
                          | ({"roc_auc_macro_ovr": v["all"]["roc_auc_macro_ovr"]} if "roc_auc_macro_ovr" in v["all"] else {}),
                          "gold_only": {kk: v["gold_only"][kk] for kk in ("accuracy", "macro_f1", "n")}}
                      for k, v in cls["methods"].items()},
          "deployed_model": next((k for k, v in cls["transformer_models"].items() if v.get("deployed")), "tfidf_logreg"),
          "fallback_model": "tfidf_logreg", "paired_bootstrap": cls["paired_bootstrap"],
          "risk_detector": {k: {kk: v[kk] for kk in ("micro_f1", "macro_f1", "macro_roc_auc", "macro_average_precision") if kk in v}
                            for k, v in risk["methods"].items()},
          "retrieval": {report_figures.RANKERS.get(k, k): v["test"] for k, v in ret["rankers"].items()},
          "answerability": ret["decisions"], "report": "reports/evaluation/RESULTS.md"}
    (settings.MODELS_DIR / "classifier_metrics.json").write_text(json.dumps(ui, indent=2), encoding="utf-8")
    report_figures.make_all(metrics, preds, y_true, OUT)
    report_figures.write_markdown(metrics, OUT / "RESULTS.md")
    print(f"[done] {OUT} ({metrics['seconds']}s)")


if __name__ == "__main__":
    main()
