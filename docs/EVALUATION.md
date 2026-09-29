# Evaluation, results and limitations

All numbers below were produced in the build environment by the scripts and tests in this repository.

## Clause classification (held-out test split, 827 clauses; gold subset 646)

| Method | Accuracy | Macro F1 | Gold accuracy | Gold macro F1 |
|---|---|---|---|---|
| Keyword baseline | 0.658 | 0.699 | 0.605 | 0.604 |
| Embedding similarity (LSA embeddings) | 0.498 | 0.534 | 0.433 | 0.415 |
| **TF-IDF + Logistic Regression (deployed)** | **0.877** | **0.878** | **0.882** | **0.812** |
| TF-IDF + Linear SVM | 0.886 | 0.886 | 0.896 | 0.827 |

Per class on the gold subset (deployed model): Dispute F1 0.99, Others 0.92, Liability 0.86, Payment 0.83,
Termination 0.78, Confidentiality 0.50 — the gold test set contains only **5** confidentiality clauses (CUAD has no
confidentiality label; most confidentiality data is heading-derived and therefore outside the gold subset), so that
number is unstable. Logistic Regression is deployed instead of the marginally better SVM because it gives probabilities,
used as the confidence score and for the low-confidence heading fallback. Full reports: `models/classifier_metrics.json`,
`notebooks/02_classifier_evaluation.ipynb`.

## Pipeline behaviour on all 32 raw documents

Every sample (all formats, including OCR) and all 12 CUAD contracts processed without errors (0.6–4 s each, rule-based
mode). Different formats of the same sample give identical clause counts and risk profiles.

| Document | Clauses | High / Medium / Low |
|---|---|---|
| One-sided employment agreement | 12 | 6 / 2 / 4 |
| Balanced offer letter | 9 | 0 / 0 / 9 |
| One-sided rental agreement (digital and scanned) | 9 | 4 / 2 / 3 |
| Balanced lease | 8 | 0 / 0 / 8 |
| One-sided NDA | 8 | 3 / 3 / 2 |
| One-sided service agreement | 9 | 3 / 2 / 4 |

The balanced documents produce no risk flags and the one-sided ones flag the clauses they were written to contain
(salary withholding, 7-vs-90-day notice, perpetual confidentiality, two-year non-compete, training penalty, unlimited
liability, unilateral arbitrator, 10-month non-refundable deposit, entry without notice…).

## Automated tests

`python -m pytest backend/tests` — **66 passed**. Covers extraction for every format and OCR, the guide's segmentation
and classification examples, RAG retrieval of the guide's 7-day-notice example, CAG persistence, risk rules, the guide's
comparison table, the LLM client for Anthropic / OpenAI-compatible / Ollama request formats against mocked HTTP
(caching, errors, fallback, "LLM never lowers risk"), and the full API flow including access control and bad uploads.

## Limitations (please state these in the report/viva)

1. **Embeddings in the build environment.** huggingface.co was not reachable where this package was built, so the
   shipped index uses the LSA fallback (TF-IDF + SVD). It works for retrieval over this small controlled corpus but is
   weaker than Sentence Transformers (see the embedding-similarity baseline row). Install `sentence-transformers`, run
   `scripts/download_models.py`, `scripts/build_vector_db.py --force` and optionally `scripts/train_classifier.py`
   to switch.
2. **No LLM was used in testing.** All results in this document come from the rule-based / extractive mode. LLM mode is
   implemented and tested against mocked API responses, but output quality with a real model has not been measured
   here. Prompts restrict the model to the supplied clauses/context.
3. **Rule-based text is plainer than LLM text.** Without an LLM, explanations are rewrites of the clause wording with
   the key facts listed, and Q&A answers quote the most relevant clauses.
4. **Document-type detection** uses keyword profiles over four types; CUAD's US commercial contracts are mostly
   detected as "service" and one maintenance agreement with heavy confidentiality language as "nda". Users can select
   the type at upload to override it.
5. **Risk rules are tuned to Indian consumer/employment contexts** (e.g. Model Tenancy Act deposit benchmark,
   s. 27 ICA non-competes). They flag *potential* risks for review, not legality.
6. **Knowledge base scope** — 15 educational summaries; it does not cover all of Indian law, and the Labour Codes may
   alter some employment positions.
7. **Clause segmentation** relies on headings/numbering; documents without structure fall back to paragraphs.
8. **PostgreSQL and Redis** are supported through SQLAlchemy/redis-py but were not available in the build environment;
   SQLite and the memory+disk cache were used for all tests.
