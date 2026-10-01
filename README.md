# LegalLens

AI-assisted legal document analysis — a final-year academic project.

Upload an employment agreement, offer letter, rental/lease agreement, NDA or service agreement (PDF, Word, text, or a
scanned image). LegalLens extracts the text (with OCR when needed), splits it into clauses, classifies each clause with
a fine-tuned Transformer, explains it in plain language using retrieved legal context (RAG), flags **potential** risks
with the evidence behind them (rules + a learned risk detector), compares the document with a reference template,
picks out key sentences, writes persona-specific summaries, answers questions about the document with follow-up support
(CAG + hybrid BM25/semantic retrieval), and exports a PDF report.

**Models** (all trained and evaluated on CUAD v1 with a contract-level split — see
[docs/EVALUATION.md](docs/EVALUATION.md)): clause classifier = fine-tuned `all-MiniLM-L6-v2` averaged with TF-IDF +
Logistic Regression (Logistic Regression alone as fallback); risk detector for 12 CUAD risk types = fine-tuned MiniLM
averaged with a TF-IDF model, on top of the rules; MiniLM sentence embeddings for RAG, comparison and key sentences;
BM25 + dense hybrid retrieval with a calibrated "not found" for Q&A.

| Held-out test contracts | Original approach | Upgraded (deployed) |
|---|---|---|
| Clause classification, macro F1 | 0.873 (TF-IDF + LogReg, retrained on the same split) | **0.897** (ensemble) |
| Risk types, micro F1 on the 6 types the rules target | 0.164 (rules) | **0.596** (rules + detector) |
| Q&A retrieval, MRR@10 | 0.534 (original ranking, LSA) | **0.666** (hybrid) |
| Says "not found" for unanswerable questions | 3% | **41%** (while answering 90% of answerable ones) |

> LegalLens is an educational tool. Its output is not legal advice; have a qualified lawyer review any document before signing.

---

## Quick start

**Requirements:** Python 3.10+ (tested on 3.12), Tesseract OCR (for scanned files), Node.js 18+ (only to rebuild the frontend).

```bash
# Linux / macOS
./setup.sh          # venv, pip install, vector DB, frontend build, tests
./run.sh            # http://localhost:8000
```

```bat
:: Windows
setup.bat
run.bat             :: http://localhost:8000
```

Manual setup:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional; everything has defaults
python scripts/refresh_models.py                       # re-fits saved models if your scikit-learn version differs
python scripts/build_vector_db.py                      # builds the RAG index (also built automatically on first use)
python -m uvicorn backend.main:app --port 8000         # API + built frontend at http://localhost:8000
```

Then register an account, upload a document from `data/raw_documents/samples/`, and open it.

**Frontend development:** `cd frontend && npm install && npm run dev` → http://localhost:5173 (proxies `/api` to port 8000).
A production build is already included in `frontend/dist/` and is served by FastAPI.

**Tesseract:** Ubuntu `sudo apt install tesseract-ocr` · macOS `brew install tesseract` ·
Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and set `TESSERACT_CMD` in `.env`.
Without Tesseract, digital PDFs/DOCX/TXT still work; scanned files report a clear error.

---

## What runs with no configuration

| Component | Default (no keys, no extra services) | Upgrade path |
|---|---|---|
| Text generation (explanations, summaries, persona, Q&A) | Deterministic rule-based / extractive mode | Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (any OpenAI-compatible server) or `OLLAMA_MODEL` (local, free) in `.env` |
| Clause classifier | Fine-tuned MiniLM (`models/transformers/clause`) + TF-IDF Logistic Regression ensemble, CPU or GPU | Logistic Regression alone if torch/transformers are missing; `CLASSIFIER_BACKEND=transformer` or `logreg` to use one model |
| Risk | Rule signals + learned detector (fine-tuned MiniLM `models/transformers/risk` + TF-IDF `models/risk_logreg.joblib`) | `RISK_MODEL=transformer`, `logreg`, or `off` for rules only |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` (downloaded from Hugging Face on first use, ~90 MB) | falls back to LSA (TF-IDF + SVD) offline; `python scripts/download_models.py` stores MiniLM in `models/` for offline use |
| Q&A retrieval | Hybrid BM25 + MiniLM, "not found" calibrated on held-out contracts | `QA_RETRIEVAL=legacy` restores the original ranking |
| Vector DB | ChromaDB (persistent, `vector_db/`) | NumPy fallback if ChromaDB is not installed |
| CAG cache | In-memory + JSON files in `cache/` | Set `REDIS_URL` |
| Database | SQLite `data/legallens.db` | `DATABASE_URL=postgresql+psycopg2://...` (`pip install psycopg2-binary`) |

Every result records which method produced it (e.g. "rule-based plain-language rewrite" or `llm:anthropic/...`), and the
System page in the app shows the status of every component.

---

## Rebuilding data and models from scratch

```bash
python scripts/download_datasets.py     # CUAD v1 from GitHub (~18 MB zip) -> data/external, spans CSV, sample contracts
python scripts/generate_sample_files.py # renders the synthetic samples to PDF/DOCX/scanned PNG+PDF
python scripts/build_datasets.py        # preprocessing of raw documents + LSA fallback embedder (+ original clause dataset)
python scripts/prepare_datasets.py      # contract-level CUAD split -> data/datasets/contract_split/ (seed 42)
python scripts/train_classifier.py      # TF-IDF + Logistic Regression fallback (+ SVM, keyword, embedding baselines)
python scripts/train_transformer.py --task clause   # fine-tuned MiniLM clause classifier -> models/transformers/clause
python scripts/train_transformer.py --task risk     # fine-tuned MiniLM risk detector    -> models/transformers/risk
python scripts/fit_ensembles.py         # TF-IDF risk model + MiniLM/TF-IDF ensemble weights and thresholds (validation only)
python scripts/evaluate_models.py       # all models on the held-out test contracts -> reports/evaluation/ (+ Q&A calibration)
python scripts/compare_sample_documents.py  # risk profile of every raw document, fallback vs upgraded configuration
python scripts/build_vector_db.py --force
python scripts/run_pipeline_cli.py data/raw_documents/samples/nda_sample.pdf --persona business --question "How long does confidentiality last?"
```
(`./setup.sh --rebuild` / `setup.bat --rebuild` run these steps.) Measured on a GTX 1650 (4 GB): clause model ≈ 13 min,
risk model ≈ 20 min, evaluation a few minutes. Any Hugging Face encoder can be trained instead of MiniLM, e.g.
`python scripts/train_transformer.py --task clause --model nlpaueb/legal-bert-base-uncased --tag legal-bert --batch-size 16`
(experiments are saved under `models/transformers/experiments/` and evaluated, but not deployed).

### GPU (optional)

The app runs on CPU. For GPU training install the CUDA build of the *same* torch version, e.g. for torch 2.13 and an
NVIDIA driver supporting CUDA 12.6+:

```bash
pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --no-deps --force-reinstall
```
`python -c "import torch; print(torch.cuda.is_available())"` should print `True`. To go back to CPU-only:
`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu --no-deps --force-reinstall`.

## Tests

```bash
python -m pytest backend/tests -q      # 90 tests: extraction/OCR, segmentation, classifiers + Transformer models,
                                       # fallbacks, RAG, hybrid retrieval, CAG, risk, comparison, LLM client
                                       # (mocked HTTP), full API flow, access control, empty/unreadable files
```

---

## Project structure

```
backend/
  main.py                 FastAPI app (API + serves frontend/dist)
  config.py               settings from environment / .env
  database/               SQLAlchemy models: users, documents, clauses, conversations, persona_summaries, comparisons
  routes/                 auth, documents, analysis (+persona), qa, comparison, report, system
  services/               extraction (PyMuPDF, python-docx, Tesseract), cleaning, clause segmentation, classifier,
                          transformer_models (fine-tuned models), embeddings, vector store, RAG, CAG cache, LLM client,
                          summarizer (+ extractive key sentences), risk analyzer, persona, Q&A + retrieval (hybrid
                          BM25/dense), comparison, PDF report, pipeline orchestration
  tests/
frontend/                 React + Vite app (src/, prebuilt dist/)
data/
  raw_documents/samples/  synthetic sample agreements (txt/pdf/docx + scanned png/pdf)
  raw_documents/cuad/     12 real contracts from CUAD v1 (CC BY 4.0)
  processed_documents/    cleaned text of every raw document
  clauses/                segmented clauses (JSON) of every raw document
  standard_documents/     project reference templates + category annotations
  legal_knowledge/        controlled RAG knowledge base with cited sources
  datasets/               CUAD spans; contract_split/ (train/val/test by contract + dataset_card.json); original clause dataset
models/                   transformers/{clause,risk}/ (fine-tuned MiniLM), clause_classifier.joblib (LogReg fallback),
                          LSA embedder, retrieval_config.json, classifier_metrics.json
vector_db/                ChromaDB index
cache/                    CAG cache (memory+disk backend)
reports/                  generated PDF reports; evaluation/ (RESULTS.md, metrics.json, figures)
notebooks/                dataset exploration, classifier evaluation, RAG/risk walkthrough (original version)
scripts/                  data download, dataset build (prepare_datasets.py), training (train_classifier.py,
                          train_transformer.py), evaluation (evaluate_models.py), index build, CLI
docs/                     architecture, datasets, API, evaluation, guide coverage, viva notes
```

See `docs/` for details: [ARCHITECTURE](docs/ARCHITECTURE.md) · [DATASETS](docs/DATASETS.md) · [API](docs/API.md) ·
[EVALUATION](docs/EVALUATION.md) · [GUIDE_COVERAGE](docs/GUIDE_COVERAGE.md) · [VIVA_NOTES](docs/VIVA_NOTES.md).

## Licences and data

Code: written for this project. CUAD v1 © The Atticus Project, CC BY 4.0 (Hendrycks, Burns, Chen and Ball, *CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021; https://github.com/TheAtticusProject/cuad,
DOI 10.5281/zenodo.4595826) — the fine-tuned models in `models/transformers/` are trained on it. Base model
`sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0). Sample agreements are synthetic and fictitious.
The knowledge base is an educational summary written for the project, citing Indian statutes and judgments — verify against
the primary sources before relying on it.

## Screenshots and sample output

`docs/screenshots/` — login, documents, clause analysis with the risk rail, reference comparison, persona summary,
Q&A, summary & recommendations, system status, mobile view.
`docs/sample_output/` — a PDF report generated by the pipeline (rule-based mode) for the synthetic rental agreement.
