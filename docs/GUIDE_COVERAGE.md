# Implementation guide coverage

Section-by-section mapping of `LegalLens_Complete_Implementation_Guide.pdf` to this code base.

| § | Guide section | Where it is implemented | Verified by |
|---|---|---|---|
| 1, 3 | What LegalLens does; input and output | Upload PDF/DOCX/TXT/images → clauses, categories, explanations, risks, persona summaries, Q&A, comparison, PDF report | `tests/test_api.py` |
| 2, 22, 26 | Development plan / order | Built in the guide's order: processing → classification → RAG → CAG → LLM modules → UI/report | — |
| 4 | Dataset collection (A–D) | `data/raw_documents`, `data/datasets`, `data/standard_documents`, `data/legal_knowledge`; `scripts/download_datasets.py`, `build_datasets.py` | `docs/DATASETS.md`, notebook 01 |
| 5 | Folder structure | `backend/ frontend/ data/{raw_documents,processed_documents,clauses,standard_documents,legal_knowledge} models/ vector_db/ cache/ reports/ notebooks/` | — |
| 6 | Document processing (PyMuPDF, python-docx, Tesseract, cleaning) | `services/extraction_service.py`, `pdf_service.py`, `docx_service.py`, `ocr_service.py`, `text_cleaning.py` | `tests/test_extraction.py` (incl. OCR of scanned PNG and PDF) |
| 7 | Clause segmentation (rules, `{id,title,text}`) | `services/clause_service.py` | `tests/test_segmentation.py` (guide's own example) |
| 8 | Classification: keyword → embeddings → ML | `services/classifier.py`, `scripts/train_classifier.py` | `tests/test_classifier.py`, notebook 02, `docs/EVALUATION.md` |
| 9 | RAG (Sentence Transformers + ChromaDB/FAISS, controlled KB) | `services/embedding_service.py`, `vector_store.py`, `rag_service.py`, `scripts/build_vector_db.py` | `tests/test_rag.py` (guide's 7-day-notice example), notebook 03 |
| 10 | CAG (document context cache, Redis or in-memory) | `services/cache_service.py`, `pipeline.get_context` | `tests/test_cache.py` |
| 11 | Summarization (clause + overall) | `services/summarizer.py` | `tests/test_api.py`, `test_llm_client.py` |
| 12 | Risk analysis LOW/MEDIUM/HIGH + reason + precaution, "potential risk" wording | `services/risk_analyzer.py` | `tests/test_risk.py` |
| 13 | Three personas (Individual, Employee, Business) | `services/persona_service.py`, `routes/analysis.py` | `tests/test_api.py::test_persona_summaries` |
| 14 | Q&A with follow-ups; "not available" answer | `services/qa_service.py`, `routes/qa.py` | `tests/test_api.py::test_qa_with_follow_up_and_not_found` |
| 15 | Standard document comparison (table, reference labelled as project template) | `services/comparison_service.py`, `routes/comparison.py` | `tests/test_comparison.py` (guide's table: 30 vs 7 days, Limited vs Unlimited) |
| 16 | Database: users, documents, clauses, conversations, persona_summaries | `database/models.py` (+ `comparisons` table) | — |
| 17 | Backend architecture (routes: auth, documents, analysis, qa, comparison; services) | `backend/routes/*`, `backend/services/*` | `docs/API.md` |
| 18 | Technology stack | React, FastAPI, PyMuPDF, python-docx, Tesseract, Sentence Transformers (optional; LSA fallback), ChromaDB (NumPy fallback), Redis/in-memory, LLM API (Anthropic/OpenAI-compatible/Ollama, optional), SQLite/PostgreSQL via SQLAlchemy, ReportLab | System page |
| 19 | Frontend (login → dashboard → upload → processing → analysis dashboard, clause detail, Q&A chat, persona dropdown) | `frontend/src/pages/*`, `components/*` | Browser run with Playwright (no console errors) |
| 20 | PDF report, 9 sections | `services/report_service.py`, `routes/report.py` | `tests/test_api.py::test_pdf_report` checks all 9 section titles |
| 21, 23 | Complete pipeline / end-to-end example | `services/pipeline.py`, `scripts/run_pipeline_cli.py` | `tests/test_api.py` |
| 24 | What not to do | No LLM training; deterministic segmentation; three personas; controlled KB; LLM prompts restricted to supplied context, LLM never lowers a rule-based risk level, facts (amounts, durations) extracted by rules | — |
| 25 | Minimum viable LegalLens | All 12 stages present | — |
| 27 | Team division | Folder/module split matches the four roles (processing · NLP/ML · generative · application) | — |
| 28 | One-sentence explanation / disclaimer | Disclaimer in UI footer, report section 9, README | — |
