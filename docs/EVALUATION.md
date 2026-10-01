# Evaluation, results and limitations

All numbers below were produced on this machine by the scripts in this repository (Windows 11, Python 3.12, torch 2.13
+ CUDA 12.6 on an NVIDIA GTX 1650 4 GB; 12-thread Intel CPU). The complete, auto-generated report with every table and
figure is [`reports/evaluation/RESULTS.md`](../reports/evaluation/RESULTS.md) (`reports/evaluation/metrics.json` holds the
raw numbers). Nothing here was estimated or copied from elsewhere.

## How the upgraded models were evaluated

* **Data.** CUAD v1 only (510 real contracts, expert annotations, CC BY 4.0) — see [DATASETS.md](DATASETS.md).
  Contracts are grouped by filing company and the groups split 70/15/15 into **train / validation / test**
  (351 / 80 / 79 contracts, seed 42). No test contract — and no sibling exhibit of a test contract — is seen during
  training; identical texts are removed across splits.
* **Selection only on validation.** Best epoch, risk thresholds, ensemble weights, Q&A fusion settings and the
  "not found" threshold were all chosen on the validation contracts. The test contracts were scored once, at the end,
  by `scripts/evaluate_models.py`.
* **Same data for every method.** All classical baselines were retrained on the same training contracts, so every row
  of every table is directly comparable.
* **Uncertainty.** 95% confidence intervals come from a cluster bootstrap over test *contracts* (1,000 resamples),
  because clauses of one contract are not independent. Paired bootstraps test the difference between two models on the
  same resamples.
* **Reproducibility.** Seeds are fixed and Transformer training uses deterministic attention kernels; two identical
  runs produced byte-identical weights (checked). The Legal-BERT experiment was interrupted by the PC sleeping and
  resumed from its epoch-2 checkpoint (optimizer, scheduler and random-number state restored).

### Why the old 0.878 cannot be compared with the new numbers

The original TF-IDF + Logistic Regression scored macro F1 **0.878** on a test set drawn **at random by clause** from a
dataset that also contained 131 project-written clauses; clauses of the same contract were in both train and test,
which inflates scores. The sandbox Transformer's **0.864** was a *validation* score on yet another split. Neither
number is comparable with anything below. The fair comparison is the table in section 1, where every method — including
Logistic Regression, retrained — is trained on the same training contracts and scored on the same unseen test contracts.

RESULTS_PLACEHOLDER

## Automated tests

TESTS_PLACEHOLDER

## Limitations (please state these in the report/viva)

1. **The Transformer alone is not better than TF-IDF on clause classification.** On validation the fine-tuned MiniLM
   (0.870 macro F1) is slightly *below* Logistic Regression (0.878) and the linear SVM (0.886); on test it is slightly
   above Logistic Regression, but the difference is inside the bootstrap noise. CUAD clause types have strong lexical
   cues, which TF-IDF captures well. The models make *different* mistakes, so their probability average is what the app
   deploys (chosen on validation).
2. **Domain shift.** All learned models are trained on US commercial contracts (CUAD). The app's sample documents and
   reference templates are Indian employment, rental, NDA and service agreements. Accuracy on the 31 hand-labelled
   reference-template clauses is reported in section 1 — n is tiny, so treat it as a sanity check only.
3. **Risk labels are CUAD clause types, not risk judgements.** The detector recognises *kinds* of clauses that reviewers
   flag (uncapped liability, non-compete, exclusivity…); the level shown to the user comes from a documented policy
   (`MODEL_RISK_POLICY`), not from the data. Known failure: negation — "This License does **not** include a minimum
   annual royalty" was flagged as *Minimum Commitment* in a CUAD sample contract. Rare types have 10–14 test
   positives, so their per-type scores are noisy.
4. **The original rules were written for Indian consumer/employment contracts**, so their low recall on CUAD types
   is expected; the comparison shows what the learned detector adds, not that the rules are wrong for their purpose.
5. **"Not found" is calibrated on CUAD's questions**, which are long and formulaic; users ask short questions. The
   operating point favours answering (≥ 90% of answerable validation questions are answered), and documents with
   fewer than 5 clauses use the original overlap rule. On the project's sample documents the decision was checked by
   hand on 20 questions; one clear miss was "How long does confidentiality last?" in the NDA, whose answer ("continue
   indefinitely") shares no words with the question.
6. **No summary quality score.** There are no reference summaries for these contracts, so the extractive key sentences
   cannot be scored with ROUGE; they are quoted verbatim (nothing is generated) and shown with links to their clauses.
7. **No LLM was used in testing.** Generation via Anthropic / OpenAI-compatible / Ollama is implemented and tested with
   mocked HTTP responses only; output quality with a real model has not been measured.
8. **Legal-BERT** was trained as an experiment only (see section 1); it is not deployed — see the trade-off below.
9. **Tesseract OCR is not installed on the evaluation machine**, so the two OCR tests are skipped; OCR was verified in
   the original build environment.
10. **Risk rules are tuned to Indian consumer/employment contexts** (e.g. Model Tenancy Act deposit benchmark,
    s. 27 ICA non-competes). They flag *potential* risks for review, not legality. The knowledge base has 15
    educational summaries and does not cover all of Indian law.
11. **Document-type detection** uses keyword profiles over four types; CUAD's US commercial contracts are mostly
    detected as "service". Users can select the type at upload to override it.
12. **Clause segmentation** relies on headings/numbering; documents without structure fall back to paragraphs.
13. **PostgreSQL and Redis** are supported through SQLAlchemy/redis-py but were not used in testing; SQLite and the
    memory+disk cache were.

## Original version (kept for reference)

Results of the first version, measured on the original random clause-level split (4,135 clauses incl. 131
project-written; test 827, gold subset 646). **Not comparable** with the tables above (see the note at the top).

| Method | Accuracy | Macro F1 | Gold accuracy | Gold macro F1 |
|---|---|---|---|---|
| Keyword baseline | 0.658 | 0.699 | 0.605 | 0.604 |
| Embedding similarity (LSA embeddings) | 0.498 | 0.534 | 0.433 | 0.415 |
| TF-IDF + Logistic Regression (deployed then) | 0.877 | 0.878 | 0.882 | 0.812 |
| TF-IDF + Linear SVM | 0.886 | 0.886 | 0.896 | 0.827 |
