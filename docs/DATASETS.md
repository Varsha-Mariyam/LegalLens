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
- `models/lsa_embedder.joblib` — LSA embedder fitted on the clause dataset + knowledge base + templates.
- `models/clause_classifier.joblib`, `models/classifier_metrics.json`.
- `vector_db/` — ChromaDB index of the knowledge base and templates.

Not included in the ZIP (re-downloadable): `data/external/cuad/CUADv1.json` (40 MB). Run `scripts/download_datasets.py`.
