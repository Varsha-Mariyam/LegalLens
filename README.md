# LegalLens

AI-assisted legal document analysis — a final-year academic project.

Upload an employment agreement, offer letter, rental/lease agreement, NDA or service agreement (PDF, Word, text, or a
scanned image). LegalLens extracts the text (with OCR when needed), splits it into clauses, classifies each clause,
explains it in plain language using retrieved legal context (RAG), flags **potential** risks with the evidence behind
them, compares the document with a reference template, writes persona-specific summaries, answers questions about the
document with follow-up support (CAG + RAG), and exports a PDF report.

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
| Embeddings | LSA (TF-IDF + SVD) trained on the project corpus | `pip install sentence-transformers && python scripts/download_models.py && python scripts/build_vector_db.py --force` |
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
python scripts/build_datasets.py        # clause dataset + train/test split + preprocessing + LSA embedder
python scripts/train_classifier.py      # evaluates 4 classifiers, saves models/clause_classifier.joblib + metrics
python scripts/build_vector_db.py --force
python scripts/run_pipeline_cli.py data/raw_documents/samples/nda_sample.pdf --persona business --question "How long does confidentiality last?"
```
(`./setup.sh --rebuild` runs the first five steps.)

## Tests

```bash
python -m pytest backend/tests -q      # 66 tests: extraction/OCR, segmentation, classifier, RAG, CAG, risk,
                                       # comparison, LLM client (mocked HTTP), full API flow, access control
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
                          embeddings, vector store, RAG, CAG cache, LLM client, summarizer, risk analyzer, persona,
                          Q&A, comparison, PDF report, pipeline orchestration
  tests/
frontend/                 React + Vite app (src/, prebuilt dist/)
data/
  raw_documents/samples/  synthetic sample agreements (txt/pdf/docx + scanned png/pdf)
  raw_documents/cuad/     12 real contracts from CUAD v1 (CC BY 4.0)
  processed_documents/    cleaned text of every raw document
  clauses/                segmented clauses (JSON) of every raw document
  standard_documents/     project reference templates + category annotations
  legal_knowledge/        controlled RAG knowledge base with cited sources
  datasets/               clause dataset, CUAD spans, train/test split
models/                   LSA embedder, clause classifier, evaluation metrics
vector_db/                ChromaDB index
cache/                    CAG cache (memory+disk backend)
reports/                  generated PDF reports
notebooks/                dataset exploration, classifier evaluation, RAG/risk walkthrough (executed)
scripts/                  data download, dataset build, training, index build, CLI
docs/                     architecture, datasets, API, evaluation, guide coverage, viva notes
```

See `docs/` for details: [ARCHITECTURE](docs/ARCHITECTURE.md) · [DATASETS](docs/DATASETS.md) · [API](docs/API.md) ·
[EVALUATION](docs/EVALUATION.md) · [GUIDE_COVERAGE](docs/GUIDE_COVERAGE.md) · [VIVA_NOTES](docs/VIVA_NOTES.md).

## Licences and data

Code: written for this project. CUAD v1 © The Atticus Project, CC BY 4.0. Sample agreements are synthetic and fictitious.
The knowledge base is an educational summary written for the project, citing Indian statutes and judgments — verify against
the primary sources before relying on it.

## Screenshots and sample output

`docs/screenshots/` — login, documents, clause analysis with the risk rail, reference comparison, persona summary,
Q&A, summary & recommendations, system status, mobile view.
`docs/sample_output/` — a PDF report generated by the pipeline (rule-based mode) for the synthetic rental agreement.
