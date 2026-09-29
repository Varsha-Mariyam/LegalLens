"""Clause explanations and overall document summary (guide section 11)."""
from __future__ import annotations

import re
from collections import Counter

from backend.services import legal_parsing as lp
from backend.services.llm_service import SYSTEM_PROMPT, LLMError, get_client

PLAIN = [
    (r"\bshall not\b", "must not"), (r"\bshall\b", "must"), (r"\bhereinafter\b", "from now on"),
    (r"\bnotwithstanding\b", "despite"), (r"\bforthwith\b", "immediately"), (r"\bthereof\b", "of it"),
    (r"\bherein\b", "in this document"), (r"\bhereby\b", ""), (r"\bhereto\b", "to this document"),
    (r"\bin lieu of\b", "instead of"), (r"\bprior to\b", "before"), (r"\bpursuant to\b", "under"),
    (r"\bin the event that\b", "if"), (r"\bin the event of\b", "if there is"), (r"\bwhereas\b", "since"),
    (r"\bindemnify\b", "compensate and protect"), (r"\bforfeited\b", "lost (not returned)"),
    (r"\bforfeit\b", "lose"), (r"\bin perpetuity\b", "forever"), (r"\bsole discretion\b", "own choice, without needing a reason"),
    (r"\bliquidated damages\b", "a fixed compensation amount"), (r"\bconsequential losses\b", "indirect losses"),
    (r"\bthe said\b", "the"), (r"\bsubject to\b", "depending on"), (r"\bcommencement\b", "start"),
    (r"\bterminate\b", "end"), (r"\btermination\b", "ending"), (r"\bvacant possession\b", "the empty property"),
    (r"\bwhatsoever\b", "at all"), (r"\baforesaid\b", "mentioned above"),
]
FACT_LABELS = {"Payment": "Money", "Termination": "Time limits", "Liability": "Money / limits"}


def plain_language(sentence: str) -> str:
    s = sentence
    for pat, rep in PLAIN:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\(\s*Rupees[^)]*\)", "", s, flags=re.IGNORECASE)
    return s[:1].upper() + s[1:] if s else s


def key_facts(text: str) -> dict:
    return {"amounts": [a["text"] for a in lp.extract_amounts(text)],
            "durations": list(dict.fromkeys(d["text"] for d in lp.extract_durations(text) if d["value"] or d["unit"])),
            "percentages": lp.extract_percentages(text)}


def rule_based_explanation(text: str, category: str) -> str:
    sents = lp.sentences(text)[:5]
    simplified = [plain_language(s) for s in sents]
    body = " ".join(simplified)
    if len(body) > 700:
        body = body[:700].rsplit(" ", 1)[0] + "…"
    facts = key_facts(text)
    extras = []
    if facts["amounts"]:
        extras.append("amounts mentioned: " + ", ".join(dict.fromkeys(facts["amounts"])))
    if facts["durations"]:
        extras.append("time periods mentioned: " + ", ".join(facts["durations"][:5]))
    if facts["percentages"]:
        extras.append("percentages mentioned: " + ", ".join(f"{p:g}%" for p in facts["percentages"]))
    out = f"In simple terms: {body}"
    if extras:
        out += "\nKey details — " + "; ".join(extras) + "."
    return out


def explain_clause(clause: dict, references: list[dict]) -> tuple[str, str]:
    client = get_client()
    if client.available:
        ref_text = "\n\n".join(f"- {r['metadata'].get('title')}: {r['text'][:600]}" for r in references[:2])
        prompt = (f"Given the following clause:\n\"\"\"\n{clause['text']}\n\"\"\"\n\nClause category: {clause['category']}\n\n"
                  f"Reference context (for background only):\n{ref_text or '(none)'}\n\n"
                  "Explain the clause in simple language in 2-4 sentences for a non-lawyer. Do not add information "
                  "that is not present in the clause. Start with 'This clause says' and do not quote the whole clause.")
        try:
            return client.generate(SYSTEM_PROMPT, prompt, max_tokens=350), client.label
        except LLMError as exc:
            return rule_based_explanation(clause["text"], clause["category"]), f"rule-based (LLM error: {str(exc)[:80]})"
    return rule_based_explanation(clause["text"], clause["category"]), "rule-based plain-language rewrite"


def rule_based_overall(doc_type: str, clauses: list[dict], full_text: str) -> str:
    counts = Counter(c["category"] for c in clauses)
    risks = Counter(c.get("risk_level") for c in clauses if c.get("risk_level"))
    parties = lp.defined_parties(full_text[:3000]) or lp.party_names(full_text[:3000])
    article = "an" if doc_type[:1] in "aeiou" else "a"
    lines = [f"This appears to be {article} {doc_type} document with {len(clauses)} clauses."]
    if parties:
        lines.append("Parties / roles identified: " + ", ".join(parties[:4]) + ".")
    cat_part = ", ".join(f"{n} {c}" for c, n in counts.most_common())
    lines.append(f"Clause categories: {cat_part}.")
    for cat in ("Payment", "Termination", "Liability", "Confidentiality", "Dispute"):
        cl = [c for c in clauses if c["category"] == cat]
        if not cl:
            continue
        facts = key_facts(" ".join(c["text"] for c in cl))
        details = (facts["amounts"][:2] + facts["durations"][:3])
        first = lp.sentences(cl[0]["text"])[0] if lp.sentences(cl[0]["text"]) else cl[0]["text"]
        lines.append(f"{cat}: {plain_language(first)[:220]}" + (f" (details: {', '.join(details)})" if details else ""))
    lines.append(f"Potential risk levels: {risks.get('HIGH', 0)} high, {risks.get('MEDIUM', 0)} medium, {risks.get('LOW', 0)} low.")
    high = [c for c in clauses if c.get("risk_level") == "HIGH"]
    if high:
        lines.append("Clauses flagged as potentially high risk: " + "; ".join(
            f"clause {lp.clause_ref(c)} ({c.get('title') or c['category']})" for c in high[:6]) + ".")
    return "\n".join(lines)


def overall_summary(doc_type: str, clauses: list[dict], full_text: str) -> tuple[str, str]:
    client = get_client()
    if client.available:
        digest = "\n".join(f"[{lp.clause_ref(c)}] {c['category']} | risk {c.get('risk_level')} | {c.get('summary', '')[:300]}"
                           for c in clauses)
        prompt = (f"Document type: {doc_type}\nClause-by-clause explanations:\n{digest[:12000]}\n\n"
                  "Write an overall summary of the document in one short paragraph followed by 3-6 bullet points of "
                  "the most important terms and potential risks. Refer to clause numbers in [n] form. Use only the "
                  "information above.")
        try:
            return client.generate(SYSTEM_PROMPT, prompt, max_tokens=700), client.label
        except LLMError as exc:
            return rule_based_overall(doc_type, clauses, full_text), f"rule-based (LLM error: {str(exc)[:80]})"
    return rule_based_overall(doc_type, clauses, full_text), "rule-based summary"
