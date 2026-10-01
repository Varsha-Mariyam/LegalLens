# Datasets and provenance

The guide (section 4) asks for four datasets. Nothing in this project is scraped from unknown sources or invented
and presented as real: every file is either (a) a public dataset with its licence, (b) written for the project and
labelled as such, or (c) generated from (a)/(b) by a script in `scripts/`.

## Dataset A — raw legal documents (`data/raw_documents/`)

| Folder | Content | Origin |
|---|---|---|
| `samples/` | 6 agreements (one-sided employment agreement, balanced offer letter, one-sided rental agreement, balanced lease, one-sided NDA, one-sided service agreement), each as `.txt`, `.pdf`, `.docx`; the rental agreement also as a scanned image (`_scanned.png`, `_scanned.pdf`) | **Synthetic**, written for the project; all names and amounts fictitious. Rendered by `scripts/generate_sample_files.py` |
| `cuad/` | 12 full commercial contracts (endorsement, licensing, maintenance, hosting, sponsorship, outsourcing, consulting, IP, joint venture…) | **Real**, from CUAD v1 (The Atticus Project, CC BY 4.0; Hendrycks et al., NeurIPS 2021). Downloaded by `scripts/download_datasets.py` |

Real Indian employment/rental agreements are private documents and are not published under an open licence, so the
Indian-context samples are synthetic. To test with your own documents, upload them in the app or put them in
`data/raw_documents/` and run `scripts/build_datasets.py` (preprocessing step).

## Dataset B — clause dataset (`data/datasets/`)

`clause_dataset.csv` (4,135 rows: `clause, label, source, origin_label`), built by `scripts/build_datasets.py`:

| Source | Rows | How the label was obtained |
|---|---|---|
| `seed_authored` | 131 | Clauses written and labelled for the project (`seed_clauses_authored.csv`), ~20 per category plus boilerplate "Others" examples |
| `cuad_v1` | 3,052 | CUAD expert annotations, mapped from CUAD's 41 categories to the six LegalLens categories (`CUAD_MAP` in the script; e.g. *Cap On Liability → Liability*, *Governing Law → Dispute*, *Termination For Convenience → Termination*, *Revenue/Profit Sharing → Payment*; capped at 150 per CUAD label) |
| `cuad_heading_weak` | 952 | **Weak labels**: sections of CUAD contracts whose drafter-written heading is e.g. "Confidentiality", "Payment", "Termination" (the method used to build the LEDGAR dataset). Needed because CUAD has no confidentiality category |

Label counts: Others 1,781 · Termination 606 · Liability 600 · Payment 576 · Dispute 353 · Confidentiality 219.

`clause_train.csv` / `clause_test.csv` — stratified 80/20 split (seed 42). The **gold subset** of the test set
(seed + CUAD expert labels, 646 rows) is evaluated separately so weak labels cannot inflate reported scores.

`cuad_annotated_spans.csv` — all 9,358 expert-annotated CUAD spans with their original labels.

These original files are kept so the first version of the project stays reproducible (`python scripts/train_classifier.py
--dataset legacy`). **They are no longer used for training or evaluation**: the split was random *by clause*, so clauses
from one contract appear in both train and test, and it includes 131 project-written clauses.

## Dataset B2 — contract-level split for the Transformer upgrade (`data/datasets/contract_split/`)

Built by `scripts/prepare_datasets.py` from `data/external/cuad/CUADv1.json` (CUAD v1, 510 contracts, 41 labels,
13,823 answer spans — all spans verified against their character offsets; SHA-256 recorded in `dataset_card.json`).
Licence: CC BY 4.0 (Zenodo record, DOI 10.5281/zenodo.4595826) — attribution in the README and here.

**Split.** Several CUAD contracts are exhibits of the same company filing and share boilerplate, so contracts are
grouped by filing company (459 groups) and the *groups* are assigned to train / validation / test (≈70/15/15 of
contracts), stratified by contract type (28 types read from the CUAD title), seed 42. Identical texts are removed across
splits (the test copy is kept), so no test text — and no sibling of a test contract — is seen in training.

| | Train | Validation | Test |
|---|---|---|---|
| Contracts | 351 (68.8%) | 80 (15.7%) | 79 (15.5%) |
| Clause classification examples | 5,933 | 1,514 | 1,463 |
| … expert (CUAD) labels / heading-derived weak labels | 4,886 / 1,047 | 1,158 / 356 | 1,121 / 342 |
| Risk-detection clauses (≥ 1 risk label) | 4,449 (1,483) | 4,327 (326) | 4,034 (347) |
| Retrieval questions (answerable) | — | 2,880 (691) | 2,844 (710) |

**Clause classification** (6 LegalLens categories, same `CUAD_MAP` and heading rules as Dataset B, no caps, no
project-written clauses): gold rows are CUAD expert spans (50–1,500 characters); weak rows are segmented clauses whose
own heading names a category (the only source of Confidentiality — CUAD has no such label). One label per distinct text:
expert labels beat heading labels; 17 texts with two different specific categories were dropped. Label counts —
train: Others 2,935 · Termination 858 · Payment 823 · Liability 715 · Dispute 436 · Confidentiality 166;
test: Others 662 · Termination 228 · Liability 203 · Payment 189 · Dispute 120 · Confidentiality 61.

**Risk detection** (multi-label, 12 CUAD clause types: Uncapped Liability, Non-Compete, Non-Solicitation [= No-Solicit
Of Employees + No-Solicit Of Customers], Exclusivity, Liquidated Damages, Change Of Control, Anti-Assignment,
Termination For Convenience, Ip Ownership Assignment, Covenant Not To Sue, Minimum Commitment, Renewal Term). The unit is
the clause exactly as LegalLens segments it (`clause_service.segment`, ≥ 40 characters). Segments are located in the
original contract text by alphanumeric alignment (98.6% of 26,484 segments located), and a segment receives a label when
it contains at least half of a CUAD answer span, or the span covers at least half of the segment (≥ 25 characters
overlap). Validation and test keep every clause (natural class balance, ~8% positive); the training split keeps all
positives plus two random negatives per positive (seed 42). Test positives per type range from 10 (Non-Solicitation)
to 81 (Anti-Assignment), so per-type scores for rare types are noisy.

**Retrieval benchmark** (`retrieval_{val,test}.jsonl`): for every validation/test contract, its segmented clauses and
the 36 non-metadata CUAD questions (the question's "Details" sentence, e.g. *"Is there a cap on liability upon the
breach of a party's obligation?"*). A clause is relevant when it overlaps an answer span (same rule as above); a
question with no answer span in that contract is *unanswerable* and used to evaluate the "not found" decision.

## Dataset C — standard / reference documents (`data/standard_documents/`)

`standard_employment.txt`, `standard_rental.txt`, `standard_nda.txt`, `standard_service.txt` — **project reference
templates written for LegalLens** with balanced terms (e.g. 30 days' mutual notice, liability capped, two-month rental
deposit, mutually appointed arbitrator). They are *not* official legal standards; the app and report say so wherever
they are used. `categories.json` holds manual category annotations for each template clause.

## Dataset D — legal knowledge base (`data/legal_knowledge/`)

15 short educational summaries organised as `general/`, `employment/`, `rental/`, `nda/`, `service/`. Each file has a
header with Title, Jurisdiction, Topic, **Sources** (statute sections / reported judgments) and a Note. Examples of
cited sources: Indian Contract Act 1872 (ss. 10, 14, 23, 27, 28, 73, 74, 124, 125), Arbitration and Conciliation Act
1996 (ss. 7, 8, 11, 12(5)), Payment of Wages Act 1936, Industrial Disputes Act 1947 (s. 25F), Model Tenancy Act 2021
(s. 11), Transfer of Property Act 1882 (ss. 106, 111), Registration Act 1908, MSMED Act 2006 (ss. 15, 16), and
judgments such as *Perkins Eastman v. HSCC* (2019), *Superintendence Co. v. Krishan Murgai* (1980), *Central Inland
Water Transport Corp. v. Brojo Nath Ganguly* (1986), *Kailash Nath Associates v. DDA* (2015). Statements that are
drafting conventions rather than law are marked "General drafting practice". These are summaries for an academic
project — verify against the primary sources; the Labour Codes (2019–2020) may change some employment positions.

## Derived artefacts

- `data/processed_documents/*.txt` and `data/clauses/*.json` — cleaned text and segmented clauses for every raw document
  (`index.json` lists method, detected type and clause count). Uploaded documents are written here as `upload_doc{id}.*`.
- `models/lsa_embedder.joblib` — LSA embedder fitted on the clause dataset + knowledge base + templates (fallback only).
- `models/clause_classifier.joblib` — TF-IDF + Logistic Regression trained on the contract-split training contracts
  (fallback classifier); `models/classifier_metrics.json` — test-set metrics of every method (written by
  `scripts/evaluate_models.py`).
- `models/transformers/{clause,risk}/` — fine-tuned MiniLM models (encoder + head + `model_config.json` with labels,
  thresholds, validation metrics, training settings and history; `training_log.jsonl`).
- `models/retrieval_config.json` — Q&A fusion settings and "not found" model, fitted on validation contracts.
- `reports/evaluation/` — metrics, tables and figures.
- `vector_db/` — ChromaDB index of the knowledge base and templates.

Not included in the ZIP (re-downloadable): `data/external/cuad/CUADv1.json` (40 MB). Run `scripts/download_datasets.py`,
then `scripts/prepare_datasets.py` reproduces `data/datasets/contract_split/` exactly (fixed seed; the SHA-256 of the
source file is recorded in `dataset_card.json`, so you can confirm a rebuild used the same file).
