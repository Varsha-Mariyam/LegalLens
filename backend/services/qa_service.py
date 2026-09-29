"""Document Q&A (guide section 14) = cached document context (CAG) + clause retrieval + legal RAG + LLM."""
from __future__ import annotations

import re

import numpy as np

from backend.services import embedding_service, rag_service
from backend.services.classifier import keyword_scores
from backend.services import legal_parsing as lp
from backend.services.llm_service import SYSTEM_PROMPT, LLMError, get_client

STOP = set("a an the is are was were be of to in on for and or what who whom which when where how does do did can "
           "could should would will shall this that it its my me i you your there any about with by as at from if "
           "under per than then clause clauses agreement document contract say says mention mentioned there's".split())
RISK_WORDS = re.compile(r"\b(risk|risky|safe|dangerous|concern|worried|problem|fair|unfair|should i sign)\b", re.I)
FOLLOW_UP = re.compile(r"\b(that|it|this|those|they|them|he|she)\b", re.I)
NOT_FOUND = ("The uploaded document does not appear to contain information about this. "
             "I can only answer from the document and the project's reference knowledge.")


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def _stem(w: str) -> str:
    return re.sub(r"(ation|ions|ing|ed|es|s)$", "", w)


def expand_query(question: str, history: list[dict]) -> str:
    continuation = re.match(r"^\s*(and|also|what about|how about)\b", question, re.I)
    if history and (FOLLOW_UP.search(question) or continuation):
        prev = history[-1]
        return f"{question} {prev['question']} {prev.get('answer', '')[:200]}"
    return question


def rank_clauses(query: str, context: dict, k: int = 3) -> list[tuple[float, dict]]:
    clauses = context["clauses"]
    if not clauses:
        return []
    emb = np.asarray(context["clause_embeddings"], dtype=np.float32)
    q = embedding_service.encode([query])[0]
    sims = emb @ q
    q_terms = {_stem(t) for t in _terms(query)}
    kw = keyword_scores(query)
    intent = max(kw, key=kw.get) if kw and max(kw.values()) > 0 else None
    scored = []
    for s, c in zip(sims, clauses):
        c_terms = {_stem(t) for t in _terms(f"{c.get('title') or ''} {c['text']} {c['category']}")}
        overlap = len(q_terms & c_terms) / (len(q_terms) or 1)
        boost = 0.25 if intent and c["category"] == intent else 0.0
        scored.append((round(float(s) * 0.6 + overlap * 0.4 + boost, 4), overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [(score, c) for score, ov, c in scored[:k] if ov > 0 or score > 0.35]


def _extractive(question: str, ranked: list[tuple[float, dict]], asks_risk: bool) -> str:
    q_terms = {_stem(t) for t in _terms(question)}
    best_sents = []
    for score, c in ranked[:2]:
        sents = lp.sentences(c["text"])
        scored = sorted(sents, key=lambda s: -len(q_terms & {_stem(t) for t in _terms(s)}))
        pick = [s for s in scored[:2] if q_terms & {_stem(t) for t in _terms(s)}] or scored[:1]
        best_sents.append((c, pick))
    parts = []
    for c, sents in best_sents:
        label = f"Clause {lp.clause_ref(c)}" + (f" ({c['title']})" if c.get("title") else "")
        parts.append(f"{label} states: \"{' '.join(sents)}\"")
    answer = "According to the document, " + " ".join(parts)
    if asks_risk:
        top = ranked[0][1]
        answer += (f"\n\nPotential risk for clause {lp.clause_ref(top)}: {top.get('risk_level', 'not assessed')}. "
                   f"{top.get('risk_reason', '')} Precaution: {top.get('precaution', '')}")
    return answer


def answer(question: str, context: dict, history: list[dict]) -> dict:
    query = expand_query(question, history)
    asks_risk = bool(RISK_WORDS.search(question))
    topic_terms = _terms(RISK_WORDS.sub(" ", question))
    is_follow_up = bool(history and FOLLOW_UP.search(question) and len(topic_terms) <= 1)
    ranked = [] if is_follow_up else rank_clauses(query, context)
    if is_follow_up and history[-1].get("sources"):
        by_number = {c["number"]: c for c in context["clauses"]}
        prev_nums = [s["clause_number"] for s in history[-1]["sources"] if s.get("clause_number") in by_number]
        ranked = [(1.0, by_number[n]) for n in prev_nums][:2]  # keep the previous answer's ranking order
    refs = rag_service.retrieve_knowledge(query, context["doc_type"], ranked[0][1]["category"] if ranked else None, k=2)
    sources = [{"clause_number": c["number"], "clause_label": lp.clause_ref(c), "title": c.get("title"), "score": s} for s, c in ranked]
    sources += [{"reference": r["metadata"].get("title"), "citation": r["metadata"].get("citation"),
                 "score": r["score"]} for r in refs]
    client = get_client()
    if client.available:
        clause_ctx = "\n\n".join(f"[Clause {lp.clause_ref(c)}] {c.get('title') or ''} (category {c['category']}, "
                                 f"risk {c.get('risk_level')})\n{c['text']}" for _, c in ranked) or "(no matching clause)"
        ref_ctx = "\n\n".join(rag_service.format_reference(r)[:700] for r in refs)
        hist = "\n".join(f"User: {h['question']}\nLegalLens: {h['answer'][:400]}" for h in history[-4:])
        prompt = (f"Conversation so far:\n{hist or '(none)'}\n\nRelevant clauses from the uploaded document:\n{clause_ctx}\n\n"
                  f"Reference knowledge (background, not part of the document):\n{ref_ctx or '(none)'}\n\n"
                  f"Question: {question}\n\nAnswer using only the supplied document clauses and reference context. Cite "
                  f"clauses as [Clause n]. If the answer cannot be found in the supplied context, reply exactly: {NOT_FOUND}")
        try:
            return {"answer": client.generate(SYSTEM_PROMPT, prompt, max_tokens=600), "sources": sources,
                    "method": client.label}
        except LLMError as exc:
            method = f"extractive (LLM error: {str(exc)[:80]})"
    else:
        method = "extractive retrieval (no LLM configured)"
    if not ranked:
        return {"answer": NOT_FOUND, "sources": [s for s in sources if "reference" in s], "method": method}
    return {"answer": _extractive(question, ranked, asks_risk), "sources": sources, "method": method}
