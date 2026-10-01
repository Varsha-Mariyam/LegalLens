"""Figures (PNG) and the Markdown results report for scripts/evaluate_models.py.

Styling follows one fixed palette: categorical series in a fixed order (blue, orange, aqua), a single blue ramp for
magnitudes (confusion matrices), recessive hairline grid, text in ink colours rather than series colours. Every figure
has a table with the same numbers in RESULTS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order; 4th slot only on adjacent bars
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NAMES = {"keyword_baseline": "Keyword baseline", "embedding_similarity_lsa": "Embedding similarity (LSA)",
         "embedding_similarity_minilm": "Embedding similarity (MiniLM)", "tfidf_logreg": "TF-IDF + Logistic Regression",
         "tfidf_linear_svm": "TF-IDF + Linear SVM", "transformer_minilm": "Fine-tuned MiniLM",
         "ensemble_minilm_logreg": "Ensemble MiniLM + LogReg (deployed)",
         "tfidf_logreg_ovr": "TF-IDF + Logistic Regression", "rules_original": "Original rules"}
CLS_MAIN = ("ensemble_minilm_logreg", "transformer_minilm", "tfidf_logreg")
RANKERS = {"legacy_lsa (original ranking, LSA)": "Original ranking + LSA (original system)",
           "legacy_minilm (original ranking, MiniLM)": "Original ranking + MiniLM",
           "lsa_dense": "LSA dense", "minilm_dense": "MiniLM dense", "bm25": "BM25",
           "hybrid_bm25_minilm (deployed)": "Hybrid BM25 + MiniLM (deployed)", "hybrid_bm25_lsa": "Hybrid BM25 + LSA"}


def name(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    if key.startswith("transformer_"):
        return f"Fine-tuned {key.removeprefix('transformer_')} (experiment)"
    return key


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Segoe UI" if "Segoe UI" in fonts else "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "legend.frameon": False, "legend.labelcolor": INK2})
    return plt


def _axes(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def classification_figure(metrics: dict, path: Path) -> None:
    plt = _plt()
    methods = metrics["classification"]["methods"]
    keys = sorted(methods, key=lambda k: methods[k]["all"]["macro_f1"])
    vals = [methods[k]["all"]["macro_f1"] for k in keys]
    lo = [v - methods[k]["all"]["macro_f1_ci95"][0] for k, v in zip(keys, vals)]
    hi = [methods[k]["all"]["macro_f1_ci95"][1] - v for k, v in zip(keys, vals)]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(keys) + 1.4))
    y = np.arange(len(keys))
    ax.barh(y, vals, height=0.5, color=SERIES[0])
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, v, h in zip(y, vals, hi):
        ax.text(v + h + 0.01, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [name(k) for k in keys])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro F1 on held-out test contracts (error bars: 95% bootstrap CI over contracts)")
    ax.set_title(f"Clause classification — {metrics['classification']['test_size']} test clauses, "
                 f"{metrics['classification']['test_contracts']} unseen contracts", loc="left")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_figure(metrics: dict, keys: list[str], path: Path) -> None:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE] + BLUES)
    methods = metrics["classification"]["methods"]
    keys = [k for k in keys if k in methods]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        cm = np.array(methods[k]["all"]["confusion_matrix"]["matrix"], dtype=float)
        labels = methods[k]["all"]["confusion_matrix"]["labels"]
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name(k)}\nmacro F1 {methods[k]['all']['macro_f1']:.3f}", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Confusion matrices (counts; shading = share of the true class)", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def risk_figure(metrics: dict, path: Path) -> None:
    plt = _plt()
    risk = metrics["risk"]
    labels = risk["labels"]
    series = [(k, risk["methods"][k]["per_label"]) for k in ("ensemble_minilm_logreg", "transformer_minilm", "tfidf_logreg_ovr")
              if k in risk["methods"]]
    series.append(("rules_original", risk["rule_mapped_types"]["methods"]["rules_original"]["per_label"]))
    fig, ax = plt.subplots(figsize=(8.5, 9))
    y = np.arange(len(labels))
    h = 0.8 / len(series)
    for s, (key, per) in enumerate(series):
        vals = [per[l]["f1"] if l in per else np.nan for l in labels]
        ax.barh(y + (s - (len(series) - 1) / 2) * h, vals, height=h * 0.88, color=SERIES[s], label=name(key))
    ax.set_yticks(y, [f"{l} (n={risk['methods']['transformer_minilm']['per_label'][l]['support']})" for l in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on test clauses (rules exist for 6 of the 12 types; no bar = no rule, or F1 0 — see table)")
    ax.set_title("Risk detection per clause type — held-out test contracts", loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def retrieval_figure(metrics: dict, path: Path) -> None:
    plt = _plt()
    rankers = metrics["retrieval"]["rankers"]
    keys = list(rankers)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(keys))
    h = 0.26
    for s, m in enumerate(("hit@1", "hit@3", "mrr@10")):
        ax.barh(y + (s - 1) * h, [rankers[k]["test"][m] for k in keys], height=h * 0.9, color=SERIES[s],
                label={"hit@1": "Hit@1", "hit@3": "Hit@3", "mrr@10": "MRR@10"}[m])
    ax.set_yticks(y, [RANKERS.get(k, k) for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    n = rankers[keys[0]]["test"]["queries"]
    ax.set_xlabel(f"Test score ({n} answerable CUAD questions over clauses of unseen contracts)")
    ax.set_title("Q&A clause retrieval", loc="left")
    ax.legend(loc="lower right")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_figure(metrics: dict, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row, (task, metric, label) in enumerate((("clause", "macro_f1", "validation macro F1"),
                                                 ("risk", "macro_ap", "validation macro average precision"))):
        h = metrics["training"][task]
        steps = h["steps"]
        per_epoch = max((s["batches"] for s in steps), default=1)
        x = [(s["epoch"] - 1) + s["batch"] / per_epoch for s in steps]
        ax = axes[row, 0]
        ax.plot(x, [s["loss"] for s in steps], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.set_title(f"{task}: training loss", loc="left")
        ax.set_xlabel("epoch")
        _axes(ax, "y")
        ax = axes[row, 1]
        ep = [e["epoch"] for e in h["history"]]
        vals = [e["val"][metric] for e in h["history"]]
        ax.plot(ep, vals, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        best = h.get("best_epoch")
        if best in ep:
            v = vals[ep.index(best)]
            ax.annotate(f"best: epoch {best}, {v:.3f}", (best, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", color=INK, fontsize=9)
        ax.set_title(f"{task}: {label}", loc="left")
        ax.set_xlabel("epoch")
        ax.set_xticks(ep)
        _axes(ax, "y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(metrics: dict, preds: dict, y_true: list[str], out: Path) -> None:
    classification_figure(metrics, out / "fig_classification_macro_f1.png")
    confusion_figure(metrics, list(CLS_MAIN), out / "fig_confusion_matrices.png")
    risk_figure(metrics, out / "fig_risk_per_label_f1.png")
    retrieval_figure(metrics, out / "fig_retrieval.png")
    training_figure(metrics, out / "fig_training_curves.png")


# ------------------------------------------------------------------------------------------ markdown

def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]" if c else ""


def _paired_lines(pairs, metric: str) -> list[str]:
    if not pairs:
        return []
    if isinstance(pairs, dict):
        pairs = [pairs]
    out = ["", f"Paired bootstrap over test contracts (difference in {metric}; 95% CI; share of resamples where the first "
               "model is not better):", ""]
    for p in pairs:
        out.append(f"- {p['comparison'].split(' (')[0].replace(' - ', ' vs ')}: {p['observed']:+.3f}, CI {_ci(p['ci95'])}, "
                   f"{p['share_of_resamples_not_better']:.3f}")
    return out


def write_markdown(m: dict, path: Path) -> None:
    card, cls, risk, ret, speed = m["dataset_card"], m["classification"], m["risk"], m["retrieval"], m["speed"]
    L = [f"# LegalLens evaluation results", "",
         f"Generated by `scripts/evaluate_models.py` on {m['created_at']} ({m['bootstrap_resamples']} bootstrap resamples). "
         "Every number is measured on the **held-out test contracts**; model selection, thresholds and retrieval "
         "settings used only the training and validation contracts.", "",
         "## Data", "",
         f"CUAD v1 ({card['source']['license']}, sha256 `{card['source']['sha256'][:16]}…`), split by filing company, "
         f"stratified by contract type, seed {card['seed']}.", "",
         "| | Train | Validation | Test |", "|---|---|---|---|",
         f"| Contracts | {card['contracts']['train']} | {card['contracts']['val']} | {card['contracts']['test']} |",
         f"| Clause-classification examples | {card['clause_classification']['sizes']['train']} | "
         f"{card['clause_classification']['sizes']['val']} | {card['clause_classification']['sizes']['test']} |",
         f"| … of which expert (gold) labels | {card['clause_classification']['gold_sizes']['train']} | "
         f"{card['clause_classification']['gold_sizes']['val']} | {card['clause_classification']['gold_sizes']['test']} |",
         f"| Risk-detection clauses (with ≥1 risk label) | {card['risk_detection']['sizes']['train']} "
         f"({card['risk_detection']['positives']['train']}) | {card['risk_detection']['sizes']['val']} "
         f"({card['risk_detection']['positives']['val']}) | {card['risk_detection']['sizes']['test']} "
         f"({card['risk_detection']['positives']['test']}) |", "",
         "## 1. Clause classification", "", "![Macro F1 by method](fig_classification_macro_f1.png)", "",
         "| Method | Val macro F1 | Test accuracy [95% CI] | Macro P | Macro R | Test macro F1 [95% CI] | Weighted F1 | ROC-AUC (macro OvR) | Gold acc. | Gold macro F1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(cls["methods"].items(), key=lambda kv: -kv[1]["all"]["macro_f1"]):
        a, g, val = v["all"], v["gold_only"], v.get("validation") or {}
        L.append(f"| {name(k)} | {_f(val.get('macro_f1'))} | {_f(a['accuracy'])} {_ci(a['accuracy_ci95'])} | {_f(a['macro_precision'])} | "
                 f"{_f(a['macro_recall'])} | **{_f(a['macro_f1'])}** {_ci(a['macro_f1_ci95'])} | {_f(a['weighted_f1'])} | "
                 f"{_f(a.get('roc_auc_macro_ovr'))} | {_f(g['accuracy'])} | {_f(g['macro_f1'])} |")
    L += ["", "Gold macro F1 averages over the five classes present in the expert-labelled subset (CUAD has no "
          "confidentiality label, so that class exists only among heading-derived labels)."]
    L += _paired_lines(cls.get("paired_bootstrap"), "macro F1")
    main = [k for k in CLS_MAIN if k in cls["methods"]]
    L += ["", f"Test: {cls['test_size']} clauses from {cls['test_contracts']} contracts ({cls['gold_test_size']} with expert "
          "labels; the rest are heading-derived weak labels). Gold = expert-labelled subset.", "",
          "Per-class F1 (all test clauses):", "", "| Class | " + " | ".join(name(k) for k in main) + " | Support |",
          "|---|" + "---|" * (len(main) + 1)]
    for c in cls["methods"]["tfidf_logreg"]["all"]["per_class"]:
        if c in ("accuracy", "macro avg", "weighted avg"):
            continue
        row = [_f(cls["methods"][k]["all"]["per_class"][c]["f1-score"]) for k in main]
        L.append(f"| {c} | " + " | ".join(row) + f" | {int(cls['methods']['tfidf_logreg']['all']['per_class'][c]['support'])} |")
    L += ["", "![Confusion matrices](fig_confusion_matrices.png)", ""]
    ood = cls.get("ood_reference_templates") or {}
    if ood:
        L += ["Out-of-domain check — the project's Indian reference templates "
              f"({ood['n']} hand-annotated clauses; accuracy): " + ", ".join(
                  f"{name(k)} {ood[k]:.3f}" for k in (*CLS_MAIN, "keyword_baseline") if k in ood) + ".", ""]

    L += ["## 2. Risk detection (12 CUAD clause types, multi-label)", "", "![Risk F1 per type](fig_risk_per_label_f1.png)", "",
          "| Method | Micro P | Micro R | Micro F1 [95% CI] | Macro F1 | Macro ROC-AUC | Macro avg. precision | Any-risk clause F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in risk["methods"].items():
        L.append(f"| {name(k)} | {_f(v['micro_precision'])} | {_f(v['micro_recall'])} | **{_f(v['micro_f1'])}** "
                 f"{_ci(v.get('micro_f1_ci95'))} | {_f(v['macro_f1'])} | {_f(v.get('macro_roc_auc'))} | "
                 f"{_f(v.get('macro_average_precision'))} | {_f(v['any_risk_clause']['f1'])} |")
    L += _paired_lines(risk["paired_bootstrap"], "micro F1")
    dep = "ensemble_minilm_logreg" if "ensemble_minilm_logreg" in risk["methods"] else "transformer_minilm"
    L += ["", f"Test: {risk['test_clauses']} clauses of {risk['test_contracts']} contracts, {risk['test_positive_clauses']} with at "
          "least one risk label (natural distribution). Thresholds per type were tuned on validation.", "",
          "| Risk type | Support | Deployed P | Deployed R | Deployed F1 | Deployed AUC | MiniLM F1 | LogReg F1 | Rules F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    rules = risk["rule_mapped_types"]["methods"]["rules_original"]["per_label"]
    for l in risk["labels"]:
        t = risk["methods"][dep]["per_label"][l]
        mm = risk["methods"]["transformer_minilm"]["per_label"][l]
        lr = risk["methods"]["tfidf_logreg_ovr"]["per_label"][l]
        L.append(f"| {l} | {t['support']} | {_f(t['precision'])} | {_f(t['recall'])} | {_f(t['f1'])} | {_f(t.get('roc_auc'))} | "
                 f"{_f(mm['f1'])} | {_f(lr['f1'])} | {_f(rules[l]['f1']) if l in rules else 'no rule'} |")
    L += ["", "On the 6 types that an original rule targets (" + ", ".join(risk["rule_mapped_types"]["labels"]) + "):", "",
          "| Method | Micro P | Micro R | Micro F1 |", "|---|---|---|---|"]
    for k, v in risk["rule_mapped_types"]["methods"].items():
        L.append(f"| {name(k)} | {_f(v['micro_precision'])} | {_f(v['micro_recall'])} | {_f(v['micro_f1'])} |")

    L += ["", "## 3. Q&A retrieval", "", "![Retrieval](fig_retrieval.png)", "",
          "| Ranker | Val MRR@10 | Test Hit@1 | Test Hit@3 | Test Hit@5 | Test MRR@10 [95% CI] | Test nDCG@5 |", "|---|---|---|---|---|---|---|"]
    for k, v in ret["rankers"].items():
        t = v["test"]
        L.append(f"| {RANKERS.get(k, k)} | {_f(v['val']['mrr@10'])} | {_f(t['hit@1'])} | {_f(t['hit@3'])} | {_f(t['hit@5'])} | "
                 f"**{_f(t['mrr@10'])}** {_ci(t.get('mrr@10_ci95'))} | {_f(t['ndcg@5'])} |")
    sel = ret["selected_config"]["sbert-all-MiniLM-L6-v2"]
    L += ["", f"Selected on validation: fusion `{sel['fusion']}`, dense weight α = {sel['alpha']}, category boost "
          f"{sel['category_boost']}. Answerability: logistic regression on (top dense cosine, top BM25 per query term, "
          f"query-term coverage) fitted on validation; threshold {sel['answerability']['threshold']} chosen so that the "
          f"assistant still answers ≥ 90% of answerable validation questions (validation: answers "
          f"{sel['answerability']['val_answered_when_answerable']:.3f} of answerable, says \"not found\" for "
          f"{sel['answerability']['val_not_found_when_unanswerable']:.3f} of unanswerable). Documents with fewer than "
          "5 clauses use the original overlap rule (BM25 statistics are degenerate there).", "",
          "\"Not found\" decision on test questions (unanswerable = the clause type is absent from the contract):", "",
          "| System | Says \"not found\" when unanswerable | Answers when answerable | Balanced acc. | Not-found F1 | End-to-end correct |",
          "|---|---|---|---|---|---|"]
    for k, d in ret["decisions"].items():
        L.append(f"| {k} | {_f(d['not_found_when_unanswerable'])} | {_f(d['answered_when_answerable'])} | "
                 f"{_f(d['balanced_accuracy'])} | {_f(d['not_found_f1'])} | {_f(d['end_to_end_correct'])} |")
    auc = ret["decisions"]["hybrid + calibrated answerability (deployed)"].get("roc_auc_p_answerable")
    L += ["", f"ROC-AUC of the calibrated answerability probability on test: {_f(auc)}. End-to-end correct = unanswerable "
          "questions answered \"not found\", answerable questions answered with a relevant clause in the top 3.", ""]
    if speed:
        L += ["## 4. Inference speed", "", "| Model | Device | ms per clause |", "|---|---|---|",
              f"| TF-IDF + Logistic Regression | CPU | {_f(speed['tfidf_logreg_cpu_ms_per_clause'], 2)} |"]
        for key in sorted({k.rsplit("_", 4)[0] for k in speed if k.endswith("_ms_per_clause") and k.startswith("transformer_")}):
            for dev in ("cuda", "cpu"):
                k = f"{key}_{dev}_ms_per_clause"
                if k in speed:
                    L.append(f"| {name(key)} ({_f(speed.get(key + '_encoder_mb'), 0)} MB) | "
                             f"{speed['gpu'] if dev == 'cuda' else 'CPU'} | {_f(speed[k], 2)} |")
        L.append("")
    samples = path.parent / "sample_documents.json"
    if samples.exists():
        import json
        rows = json.loads(samples.read_text(encoding="utf-8"))
        L += ["## Sample documents: fallback vs upgraded configuration", "",
              "Fallback = TF-IDF + Logistic Regression classifier and rule signals only; upgraded = the deployed "
              "configuration (MiniLM + Logistic Regression ensemble classifier, rule signals + learned risk detector) "
              "(`scripts/compare_sample_documents.py`). Levels can also change through classification: a clause "
              "classified differently triggers different category-specific rules.", "",
              "| Document | Type | Clauses | Fallback H / M / L | Upgraded H / M / L | Learned-detector evidence |", "|---|---|---|---|---|---|"]
        for r in rows:
            b, a = r["fallback"], r["upgraded"]
            items = "; ".join(a["detector_evidence"][:4]) + (" …" if len(a["detector_evidence"]) > 4 else "")
            L.append(f"| {r['file']} | {a['doc_type']} | {a['clauses']} | {b['HIGH']} / {b['MEDIUM']} / {b['LOW']} | "
                     f"{a['HIGH']} / {a['MEDIUM']} / {a['LOW']} | {items or '—'} |")
        L.append("")
    L += ["## 5. Training", "", "![Training curves](fig_training_curves.png)", ""]
    for task in ("clause", "risk"):
        h = m["training"][task]
        tr = h["training"]
        L += [f"**{task}** — base `{tr.get('model')}`, {tr.get('epochs')} epochs, lr {tr.get('lr')}, batch {tr.get('batch_size')}, "
              f"max length {tr.get('max_length')}, seed {tr.get('seed')}, device {tr.get('device')}; best epoch "
              f"{h['best_epoch']} (selected on {tr.get('selection_metric')}); {tr.get('total_seconds')} s.", "",
              "| Epoch | Train loss | Validation |", "|---|---|---|"]
        for e in h["history"]:
            L.append(f"| {e['epoch']} | {_f(e['train_loss'], 4)} | " + ", ".join(f"{k} {v:.4f}" for k, v in e["val"].items()) + " |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    """Re-draw the figures and RESULTS.md from reports/evaluation/metrics.json (no models needed)."""
    import json
    out = Path(__file__).resolve().parent.parent / "reports" / "evaluation"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    make_all(metrics, None, None, out)
    write_markdown(metrics, out / "RESULTS.md")
    print(f"regenerated figures and RESULTS.md in {out}")


if __name__ == "__main__":
    main()
