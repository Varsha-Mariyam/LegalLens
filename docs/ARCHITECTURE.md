# Architecture

## Pipeline

```
Upload ──► Extraction ──► Cleaning ──► Clause segmentation ──► Classification
(FastAPI)  PyMuPDF /       unicode,      numbered / ARTICLE /     keyword baseline
           python-docx /   hyphenation,  roman / uppercase        embedding similarity
           Tesseract OCR   page numbers  headings, paragraph      TF-IDF + LogReg (deployed)
                                          fallback
        ──► CAG: cache document context (text, clauses, clause embeddings) — Redis or memory+disk
        ──► per clause: RAG retrieval (legal knowledge + reference clause) ──► plain-language explanation
                                                                            ──► risk analysis (rules + optional LLM)
        ──► comparison with the reference template ──► overall summary + recommendations ──► DB
On demand: persona summary · Q&A (CAG context + clause retrieval + RAG + history) · PDF report
```

`backend/services/pipeline.py` runs these steps in a FastAPI background task and records progress
(`documents.current_step`, `documents.progress`), which the frontend polls.

## Components

| Guide module | Implementation |
|---|---|
| Text extraction | `extraction_service.py` dispatches to `pdf_service.py` (PyMuPDF text layer; pages with < 40 characters are OCR'd at 300 dpi), `docx_service.py` (paragraphs + tables), `ocr_service.py` (Tesseract) |
| Cleaning | `text_cleaning.py` — deterministic, idempotent; rebuilds paragraphs from wrapped lines, splits run-in headings |
| Segmentation | `clause_service.py` — heading hierarchy (1., 1.1, ARTICLE IV, I., UPPERCASE), preamble and execution units, long-paragraph splitting |
| Classification | `classifier.py` — all three methods run; ML result used, with heading-keyword fallback when ML confidence < 0.45; preamble/execution forced to Others |
| Embeddings | `embedding_service.py` — Sentence Transformers if available locally, else LSA (TF-IDF bigrams + 256-d SVD) |
| Vector DB | `vector_store.py` — ChromaDB (cosine, persistent) or NumPy store with the same interface |
| RAG | `rag_service.py` — two collections: `legal_knowledge` (chunked KB, filtered by document type + general) and `standard_clauses` (reference templates, filtered by type and category); category-topic score boost; index fingerprint (files + embedder) triggers rebuilds |
| CAG | `cache_service.py` — document context and LLM responses keyed by prompt hash; `pipeline.get_context` rebuilds from DB on a miss or when the embedder changed |
| LLM | `llm_service.py` — Anthropic Messages API, OpenAI-compatible chat completions, Ollama; httpx with injectable transport (tests) |
| Summaries | `summarizer.py` — LLM prompt grounded in clause + retrieved references, or rule-based rewrite (modal verbs → plain words, key amounts/durations) |
| Risk | `risk_analyzer.py` — ~25 evidence rules, each with a reason, precaution and source; the LLM (if configured) can only raise the level, never lower it |
| Persona | `persona_service.py` — Individual / Employee / Business prompts; rule-based version filters obligations by party role |
| Q&A | `qa_service.py` — follow-up detection (pronouns, "and/what about"), clause ranking = 0.6·cosine + 0.4·term overlap + 0.25 category-intent boost, extractive answer with clause citations, explicit "not in the document" answer |
| Comparison | `comparison_service.py` — best semantic match per reference clause (+ same-category bonus), key-value extraction (amount, notice days, liability limited/unlimited, arbitration appointment, confidentiality duration) → equivalent / differs / missing / additional |
| Report | `report_service.py` — ReportLab, 9 sections as in the guide |

## Data model (SQLAlchemy)

- **users** (user_id, name, email, password_hash [PBKDF2-SHA256, 240k iterations], created_at)
- **documents** (document_id, user_id, filename, stored_path, upload_date, document_type, detected_type, status,
  current_step, progress, error, extraction_method, page_count, char_count, full_text, overall_summary,
  summary_method, recommendations JSON, analysis_meta JSON, processed_at)
- **clauses** (clause_id, document_id, clause_number, label, title, clause_type, category_confidence,
  classifier_details, original_text, summary, summary_method, risk_level, risk_reason, precaution,
  risk_evidence, risk_method, references, standard_match)
- **conversations** (conversation_id, document_id, question, answer, sources, method, timestamp)
- **persona_summaries** (id, document_id, persona, summary, method, created_at)
- **comparisons** (id, document_id, position, category, status, standard/user title+text, user_clause_number,
  user_clause_label, similarity, standard_value, user_value, difference, method)

## Security

JWT bearer tokens (HS256, `SECRET_KEY`), every document route checks ownership (other users get 404), upload type and
size limits, sanitised stored filenames, SPA fallback refuses paths outside `frontend/dist`.

## Frontend

React 18 + React Router + Vite; fonts bundled with Fontsource (works offline). Pages: login/register, documents
(upload + list), document analysis (clauses with a clickable risk rail, summary & recommendations, reference comparison,
persona summary, Q&A chat, PDF download), system status (component health, classifier metrics, knowledge sources).
