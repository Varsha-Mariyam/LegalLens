# Architecture

## Pipeline

```
Upload ──► Extraction ──► Cleaning ──► Clause segmentation ──► Classification
(FastAPI)  PyMuPDF /       unicode,      numbered / ARTICLE /     fine-tuned MiniLM Transformer (deployed)
           python-docx /   hyphenation,  roman / uppercase        TF-IDF + LogReg (fallback)
           Tesseract OCR   page numbers  headings, paragraph      keyword + embedding baselines (recorded)
                                          fallback
        ──► CAG: cache document context (text, clauses, MiniLM clause embeddings) — Redis or memory+disk
        ──► learned risk detector over all clauses in one batch (12 CUAD risk types, fine-tuned MiniLM)
        ──► per clause: RAG retrieval (legal knowledge + reference clause) ──► plain-language explanation
                                                                            ──► risk analysis (rules + detector + optional LLM)
        ──► comparison with the reference template ──► extractive key sentences ──► overall summary + recommendations ──► DB
On demand: persona summary · Q&A (CAG context + hybrid BM25/dense clause retrieval with calibrated "not found"
           + RAG + history) · PDF report
```

## Transformer upgrade (models and training)

| Piece | Where | What it does |
|---|---|---|
| Datasets | `scripts/prepare_datasets.py` → `data/datasets/contract_split/` | CUAD v1 only; contracts grouped by filing company and split 70/15/15 (seed 42, stratified by contract type); every contract segmented with the app's own segmenter and CUAD answer spans aligned to the segments; cross-split duplicates removed |
| Model | `backend/services/transformer_models.py` | pretrained encoder (default `sentence-transformers/all-MiniLM-L6-v2`, any Hugging Face encoder works) → masked mean pooling → dropout → linear head; softmax (clause) or per-label sigmoid with validation-tuned thresholds (risk); float16 safetensors on disk, loaded lazily, thread-safe, failures recorded and reported |
| Training | `scripts/train_transformer.py` | AdamW + linear warm-up/decay, class-weighted loss (balanced CE / BCE with `pos_weight`), fp16 mixed precision on CUDA, deterministic attention + fixed seeds (bit-for-bit reproducible runs), best epoch chosen on validation, resumable checkpoints, JSONL training log |
| Evaluation | `scripts/evaluate_models.py` + `scripts/report_figures.py` → `reports/evaluation/` | every method scored once on the held-out test contracts; bootstrap CIs over contracts; writes `models/classifier_metrics.json` (System page) and `models/retrieval_config.json` (Q&A settings fitted on validation) |

Fallback chain (no step ever stops the analysis): clause classifier Transformer → TF-IDF + Logistic Regression →
keyword / embedding baselines; risk = rules (+ detector when available) (+ LLM when configured); Q&A = hybrid →
(`QA_RETRIEVAL=legacy`) original ranking. Settings: `CLASSIFIER_BACKEND=auto|transformer|logreg`,
`RISK_MODEL=auto|off`, `TRANSFORMER_DEVICE=auto|cpu|cuda`, `QA_RETRIEVAL=hybrid|dense|legacy`.

`backend/services/pipeline.py` runs these steps in a FastAPI background task and records progress
(`documents.current_step`, `documents.progress`), which the frontend polls.

## Components

| Guide module | Implementation |
|---|---|
| Text extraction | `extraction_service.py` dispatches to `pdf_service.py` (PyMuPDF text layer; pages with < 40 characters are OCR'd at 300 dpi), `docx_service.py` (paragraphs + tables), `ocr_service.py` (Tesseract) |
| Cleaning | `text_cleaning.py` — deterministic, idempotent; rebuilds paragraphs from wrapped lines, splits run-in headings |
| Segmentation | `clause_service.py` — heading hierarchy (1., 1.1, ARTICLE IV, I., UPPERCASE), preamble and execution units, long-paragraph splitting |
| Classification | `classifier.py` — all methods run and are stored; the fine-tuned Transformer's label is used (TF-IDF + LogReg when the Transformer is unavailable), with heading-keyword fallback when model confidence < 0.45; preamble/execution forced to Others |
| Embeddings | `embedding_service.py` — Sentence Transformers (all-MiniLM-L6-v2) when installed, else LSA (TF-IDF bigrams + 256-d SVD); the active backend is recorded with every analysis |
| Vector DB | `vector_store.py` — ChromaDB (cosine, persistent) or NumPy store with the same interface |
| RAG | `rag_service.py` — two collections: `legal_knowledge` (chunked KB, filtered by document type + general) and `standard_clauses` (reference templates, filtered by type and category); category-topic score boost; index fingerprint (files + embedder) triggers rebuilds |
| CAG | `cache_service.py` — document context and LLM responses keyed by prompt hash; `pipeline.get_context` rebuilds from DB on a miss or when the embedder changed |
| LLM | `llm_service.py` — Anthropic Messages API, OpenAI-compatible chat completions, Ollama; httpx with injectable transport (tests) |
| Summaries | `summarizer.py` — LLM prompt grounded in clause + retrieved references, or rule-based rewrite (modal verbs → plain words, key amounts/durations); document-level **extractive key sentences**: sentences embedded with MiniLM, ranked by similarity to the document centroid, Maximal Marginal Relevance (λ = 0.7) for diversity, quoted verbatim in document order |
| Risk | `risk_analyzer.py` — ~25 evidence rules, each with a reason, precaution and source, plus one evidence item per learned-detector detection (level from `MODEL_RISK_POLICY`: commonplace types informational, mutual wording lowers termination-for-convenience and non-solicitation, a detection below p = 0.5 can never make a clause HIGH); the final level is the highest of rules, detector and LLM — nothing lowers a rule level |
| Persona | `persona_service.py` — Individual / Employee / Business prompts; rule-based version filters obligations by party role |
| Q&A | `qa_service.py` + `retrieval.py` — follow-up detection (pronouns, "and/what about"); clause ranking = BM25 (rank_bm25, per document) fused with MiniLM cosine (weight and fusion method chosen on validation contracts); "not found" when a logistic model over (top cosine, top BM25 per query term, query-term coverage) says the question is unanswerable — calibrated on validation contracts; extractive answer with clause citations. `QA_RETRIEVAL=legacy` restores the original 0.6·cosine + 0.4·term overlap + 0.25 intent-boost ranking |
| Comparison | `comparison_service.py` — best semantic match per reference clause (+ same-category bonus), key-value extraction (amount, notice days, liability limited/unlimited, arbitration appointment, confidentiality duration) → equivalent / differs / missing / additional; the "equivalent" similarity cut-off is per embedding backend (0.75 LSA, 0.65 MiniLM) |
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
persona summary, Q&A chat, PDF download), system status (component health incl. Transformer models, risk detector and
Q&A retrieval mode; classification / risk / retrieval evaluation tables; knowledge sources). The summary tab shows the
extractive key sentences with links to their clauses; learned-detector evidence is tagged "model" in the clause panel.
