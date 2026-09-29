# Viva / presentation notes

**One sentence (guide §28).** LegalLens extracts and classifies contract clauses, uses RAG to retrieve relevant legal
and reference knowledge and CAG to keep the document's context, generates plain-language and persona-based
explanations, identifies potential risks, answers questions about the document, compares it with a reference
template and produces a consolidated report.

## Demo script (≈5 minutes)
1. System page — show which components are active (LLM or rule-based, embeddings, ChromaDB, cache, OCR, classifier metrics).
2. Upload `employment_agreement_sample.pdf` → processing steps.
3. Clauses tab — click the red segments on the risk rail: Termination (7 vs 90 days), Liability (unlimited), Non-compete.
   Open "Show retrieved reference knowledge" to show the RAG chunks and the three classifier outputs.
4. Reference comparison — the guide's table: 30 days vs 7 days, Limited vs Unlimited; expand a row.
5. Persona — Employee, then Business: same document, different focus.
6. Ask LegalLens — "What is the notice period?" → "Is that risky?" (follow-up) → "Is there a clause about pet ownership?" (not found).
7. Download the PDF report (9 sections).
8. Optional: upload `rental_agreement_sample_scanned.png` to show OCR.

## Likely questions

**Why not use an LLM for everything?** The guide (§24) and good engineering: segmentation and fact extraction must be
deterministic and repeatable; the LLM is used only for language generation, grounded in retrieved text. Amounts and
durations are extracted by rules so they cannot be hallucinated.

**RAG vs CAG?** RAG retrieves *external* knowledge (statute summaries, reference clauses) per clause/question from the
vector DB. CAG caches the *uploaded document's* context (clauses + embeddings + summary) once, so each question
reuses it instead of re-extracting and re-embedding. Cache: Redis if configured, else memory + disk; rebuilt from the
DB on a miss.

**How is risk decided?** ~25 evidence rules (short or unequal notice, unlimited liability, penalty clauses, salary
withholding, excessive deposit vs Model Tenancy Act benchmark, post-employment non-compete vs ICA s. 27, one-sided
arbitrator appointment, etc.). Each rule produces a level, reason, precaution and source. With an LLM configured, the
LLM also assesses the clause but can only raise the level. Wording is always "potential risk".

**How good is the classifier?** TF-IDF + Logistic Regression: 0.878 macro F1 overall, 0.812 on expert-labelled gold
data, vs 0.699 for keywords. Weak point: few gold confidentiality examples.

**Where does the data come from?** CUAD v1 (510 real contracts, expert labels, CC BY 4.0) mapped to our six
categories; heading-derived weak labels (LEDGAR method); project-authored seeds. Sample agreements are synthetic and
labelled so. Reference templates are ours and labelled "not an official legal standard".

**Why is the comparison reference "not official"?** No official standard employment/rental agreement exists; we wrote
balanced templates and say so in the UI and report.

**What if a question isn't answered by the document?** Clause ranking finds no relevant clause → fixed "does not appear
to contain" answer; the LLM prompt also instructs that exact reply.

**Security?** PBKDF2-SHA256 passwords, JWT, per-user document isolation (tested), upload type/size limits.

**Limitations / future work.** Sentence-Transformer or legal-domain embeddings (e.g. fine-tuned on CUAD), measured LLM
output quality, more document types, Malayalam/Hindi OCR (`OCR_LANG`), a larger verified knowledge base, lawyer review
of the risk rules.
