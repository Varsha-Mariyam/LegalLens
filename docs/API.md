# REST API

Base URL `http://localhost:8000`. Interactive documentation: **http://localhost:8000/docs** (Swagger UI).
All routes except `/api/auth/register`, `/api/auth/login`, `/api/health`, `/api/system/status`, `/api/system/knowledge`
and `/api/standards/{type}` need `Authorization: Bearer <token>`. Documents belong to their uploader; other users get 404.

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/register` | `{name, email, password(≥8)}` → `{access_token, user}` (409 if email exists) |
| POST | `/api/auth/login` | `{email, password}` → `{access_token, user}` |
| GET | `/api/auth/me` | Current user |
| POST | `/api/documents` | multipart `file` (+ `document_type`: auto/employment/rental/nda/service) → 202; analysis runs in the background. 415 unsupported type, 413 too large, 422 empty |
| GET | `/api/documents` | Your documents with status, clause and risk counts |
| GET / DELETE | `/api/documents/{id}` | Document metadata / delete document, files and cache |
| GET | `/api/documents/{id}/status` | `{status, current_step, progress, error, steps}` — poll during processing |
| GET | `/api/documents/{id}/text` | Extracted text |
| GET | `/api/documents/{id}/clauses` | Clause list (preview) |
| GET | `/api/documents/{id}/clauses/{clause_id}` | Full clause analysis |
| POST | `/api/documents/{id}/reprocess` | Re-run the pipeline |
| GET | `/api/documents/{id}/analysis` | Everything: summary, recommendations, clauses, comparisons, category counts, methods used (409 until completed) |
| GET / POST | `/api/documents/{id}/persona` | List saved summaries / `{persona: individual\|employee\|business}` generates and saves one |
| GET / POST / DELETE | `/api/documents/{id}/qa` | Conversation history / `{question}` → answer with sources / clear history |
| GET | `/api/documents/{id}/comparison` | Reference-template comparison rows + summary |
| GET | `/api/standards/{doc_type}` | Reference template clauses |
| GET | `/api/documents/{id}/report` | PDF report download |
| GET | `/api/system/status` | LLM, embeddings, RAG index, cache, OCR, classifier (`primary`, `transformer`, fallback), `risk_detector`, `qa_retrieval` and evaluation metrics |
| GET | `/api/system/knowledge` | Knowledge-base files and their sources |
| POST | `/api/system/reindex` | Rebuild the vector index |
| GET | `/api/health` | Liveness |

### Fields added by the Transformer upgrade (all additive — existing fields are unchanged)

| Where | Field | Meaning |
|---|---|---|
| clause `classifier_details` | `transformer: {label, confidence}` | fine-tuned Transformer prediction (present when the model is loaded); `method` is `transformer:<base model> (fine-tuned)`, `ml:tfidf-logreg` in fallback |
| clause `risk_evidence[]` | items with `kind: "model"`, `rule: "model:<type>"`, `probability` | learned risk-detector detections; `risk_method` ends with `+ learned detector` when it ran |
| `analysis_meta` | `risk_detector {used, base_model, labels, detections}`, `qa_retrieval`, `extractive_summary [{clause_number, clause_label, category, sentence, centrality}]`, `classifier {primary, transformer, ...}` | how this analysis was produced; key sentences quoted from the document |
| Q&A `sources[]` | `dense`, `bm25` | the two retrieval scores behind each cited clause; `method` names the retriever, e.g. `extractive answer (no LLM configured) · hybrid retrieval (BM25 + sbert-all-MiniLM-L6-v2)` |
| `/api/system/status` | `risk_detector`, `qa_retrieval {mode, fusion, alpha, category_boost, calibrated_not_found}`, `classifier_metrics.{risk_detector, retrieval, answerability}` | component status and test-set evaluation summary |

Empty or unreadable files: an empty upload is rejected with 422; a file whose text cannot be read (blank text, corrupt
PDF, scanned image without Tesseract) is accepted (202) and its status becomes `failed` with the reason in `error`.

Example:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"password123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -s -H "Authorization: Bearer $TOKEN" -F file=@data/raw_documents/samples/nda_sample.pdf -F document_type=auto localhost:8000/api/documents
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"question":"How long does confidentiality last?"}' localhost:8000/api/documents/1/qa
curl -s -H "Authorization: Bearer $TOKEN" -o report.pdf localhost:8000/api/documents/1/report
```
