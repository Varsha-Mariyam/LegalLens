# Viva / presentation notes

**One sentence (guide §28).** LegalLens extracts and classifies contract clauses, uses RAG to retrieve relevant legal
and reference knowledge and CAG to keep the document's context, generates plain-language and persona-based
explanations, identifies potential risks, answers questions about the document, compares it with a reference
template and produces a consolidated report.

## Demo script (≈5 minutes)
1. System page — show which components are active (LLM or rule-based, embeddings, ChromaDB, cache, OCR, the
   MiniLM + Logistic Regression ensemble classifier, the risk detector, hybrid Q&A) and the three evaluation tables
   (classification, risk detector, retrieval — all on held-out contracts).
2. Upload `employment_agreement_sample.pdf` → processing steps.
3. Clauses tab — click the red segments on the risk rail: Termination (7 vs 90 days), Liability (unlimited), Non-compete.
   Point out evidence tagged "model" (learned detector) next to the rule evidence. Open "Show retrieved reference
   knowledge" to show the RAG chunks and every classifier's output (ensemble, Transformer, LogReg, keyword, embedding).
   The Summary tab shows the extractive key sentences with links to their clauses.
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

**How good is the classifier?** On 1,463 clauses of 79 contracts never seen in training: deployed ensemble
(fine-tuned MiniLM × 0.3 + TF-IDF Logistic Regression × 0.7) macro F1 **0.897** [95% CI 0.867–0.922], Logistic
Regression alone 0.873, keyword baseline 0.694. The ensemble's gain over Logistic Regression (+0.024, CI
+0.008…+0.041) is statistically clear. Legal-BERT reaches 0.909 but is 5× larger and 7× slower on CPU, so it is an
evaluated option, not the default.

**Why is the old 0.878 not in your comparison?** It was measured on a random split by clause, so clauses of the same
contract were in train and test, plus 131 clauses we wrote ourselves — an optimistic setting. The sandbox's 0.864 was a
validation score on another split. We rebuilt the data split *by contract* (grouped by filing company) and retrained
every method on it, so all numbers in the new table are comparable; the old ones are not.

**Did the Transformer beat the simple model?** Honestly: not on its own. Fine-tuned MiniLM alone (0.884 test, 0.870
validation) is within noise of TF-IDF + Logistic Regression — CUAD's categories have strong keywords. The two models
make different errors (MiniLM keeps specific clauses out of "Others"; LogReg is better on Liability), so averaging
their probabilities — weight chosen on validation — is what improves results. A domain-pretrained model (Legal-BERT)
is the most accurate single model.

**How do you know the numbers are not inflated?** Split by contract and filing company; duplicates removed across
splits; best epoch, thresholds, ensemble weights and retrieval settings chosen on validation only; test scored once;
95% confidence intervals by bootstrap over contracts; all baselines retrained on the same data; seeds fixed and
training bit-for-bit reproducible (checked).

**How does the learned risk detector combine with the rules?** It recognises 12 CUAD clause types (uncapped liability,
non-compete, exclusivity, liquidated damages…). Each detection becomes one evidence item; its level comes from a
written policy (commonplace types are LOW, mutual wording lowers some, a low-probability detection cannot make a clause
HIGH). The final level is the highest of rules, detector and LLM — nothing lowers a rule's level. On CUAD test
contracts it reaches micro F1 0.59 vs 0.16 for the rules on the six types the rules target. On our Indian samples it
changes no risk level (balanced documents stay clean); on real US contracts it adds MEDIUM flags the rules never had.

**What changed in Q&A?** Hybrid retrieval: BM25 (exact words) fused with MiniLM embeddings (meaning). On 2,844 CUAD
questions over unseen contracts, MRR@10 rose from 0.534 (original ranking) to 0.666. The original system said
"not found" for only 3% of unanswerable questions; a small calibrated model now says it for 41% while still answering
90% of answerable ones.

**Where does the data come from?** CUAD v1 (510 real contracts, expert labels, CC BY 4.0) mapped to our six
categories, plus heading-derived weak labels (LEDGAR method) for confidentiality, which CUAD lacks; split by contract
for the upgrade. Sample agreements are synthetic and labelled so. Reference templates are ours and labelled "not an
official legal standard".

**Why is the comparison reference "not official"?** No official standard employment/rental agreement exists; we wrote
balanced templates and say so in the UI and report.

**What if a question isn't answered by the document?** A small logistic model (top embedding similarity, top BM25 score
per question word, share of question words found) estimates whether the best clause really answers the question; below
the threshold — set on validation contracts so that ≥ 90% of answerable questions are still answered — LegalLens gives
the fixed "does not appear to contain" answer; the LLM prompt also instructs that exact reply. Documents with fewer than
5 clauses use the original word-overlap rule (BM25 statistics need a few clauses).

**Security?** PBKDF2-SHA256 passwords, JWT, per-user document isolation (tested), upload type/size limits.

**Limitations / future work.** Training data is US commercial contracts (domain shift to Indian agreements); the risk
detector misses negation ("does **not** include a minimum royalty"); "not found" is calibrated on CUAD's formulaic
questions; no reference summaries, so key sentences are not scored; LLM output quality not measured. Future: Legal-BERT
or a Legal-BERT ensemble on a GPU server, an Indian labelled clause set, a cross-encoder re-ranker for Q&A, Malayalam /
Hindi OCR (`OCR_LANG`), a larger verified knowledge base, lawyer review of the risk rules and risk policy.
