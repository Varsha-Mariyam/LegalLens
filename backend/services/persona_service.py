"""Persona-based summaries (guide section 13): same document + context, different perspective prompt."""
from __future__ import annotations

import re

from backend.services import legal_parsing as lp
from backend.services.llm_service import SYSTEM_PROMPT, LLMError, get_client
from backend.services.summarizer import plain_language

PERSONAS = {
    "individual": {
        "label": "Individual",
        "focus": ["Payment", "Termination", "Liability", "Dispute", "Confidentiality"],
        "roles": ["tenant", "employee", "receiving party", "licensee", "you", "individual", "consultant", "client"],
        "prompt": "Explain this document from the perspective of an individual signing it (for example a tenant, an "
                  "employee or a person receiving confidential information). Focus on what they must pay or do, what "
                  "they receive, how they can leave, what they could be liable for and potential risks.",
    },
    "employee": {
        "label": "Employee",
        "focus": ["Payment", "Termination", "Confidentiality", "Liability", "Dispute", "Others"],
        "roles": ["employee", "staff", "you", "consultant", "worker"],
        "prompt": "Explain this document from an employee's perspective. Focus on salary, duties, termination, "
                  "confidentiality, rights, obligations and potential risks.",
    },
    "business": {
        "label": "Business",
        "focus": ["Payment", "Liability", "Termination", "Dispute", "Confidentiality"],
        "roles": ["company", "employer", "client", "service provider", "landlord", "disclosing party", "licensor",
                  "supplier", "buyer", "distributor"],
        "prompt": "Explain this document from a business's perspective. Focus on commercial obligations, payment "
                  "terms, liability exposure, termination rights, protection of information, dispute resolution and "
                  "compliance risks.",
    },
}


def _obligation_sentences(clauses: list[dict], roles: list[str]) -> list[tuple[int, str]]:
    out = []
    role_re = re.compile(r"^\s*(?:the\s+|each\s+|either\s+)?(" + "|".join(re.escape(r) for r in roles) + r")\b[^.]{0,30}\b(shall|must|will|agrees?)\b",
                         re.IGNORECASE)
    for c in clauses:
        for s in lp.sentences(c["text"]):
            if role_re.search(s) or re.match(r"^\s*(either|each) party", s, re.IGNORECASE):
                out.append((lp.clause_ref(c), plain_language(s)))
    return out


def rule_based(persona: str, doc_type: str, clauses: list[dict]) -> str:
    cfg = PERSONAS[persona]
    focus = [c for c in clauses if c["category"] in cfg["focus"]]
    lines = [f"{cfg['label']} view of this {doc_type} document"]
    lines.append("")
    lines.append("What this document means for you:")
    for cat in cfg["focus"]:
        cl = [c for c in focus if c["category"] == cat]
        if cl:
            lines.append(f"• {cat}: " + " ".join(f"[{lp.clause_ref(c)}] {(c.get('summary') or '').replace('In simple terms: ', '').split(chr(10))[0][:220]}" for c in cl[:2]))
    obligations = _obligation_sentences(clauses, cfg["roles"])
    if obligations:
        lines.append("")
        lines.append("Obligations that apply to your side:")
        for num, s in obligations[:8]:
            lines.append(f"• [{num}] {s[:220]}")
    risky = sorted([c for c in focus if c.get("risk_level") in ("HIGH", "MEDIUM")],
                   key=lambda c: 0 if c["risk_level"] == "HIGH" else 1)
    lines.append("")
    if risky:
        lines.append("Potential risks to watch:")
        for c in risky[:6]:
            lines.append(f"• [{lp.clause_ref(c)}] {c['risk_level']} — {c.get('title') or c['category']}: {c.get('precaution', '')[:220]}")
    else:
        lines.append("Potential risks to watch: the rule checks found no medium or high risk indicators in your focus areas.")
    lines.append("")
    lines.append("Questions to ask before signing:")
    for c in risky[:4]:
        lines.append(f"• Can clause {lp.clause_ref(c)} ({c.get('title') or c['category']}) be changed so that: {c.get('precaution', '').split('.')[0]}?")
    lines.append("• Is anything you were promised verbally missing from the written document?")
    return "\n".join(lines)


def generate(persona: str, doc_type: str, clauses: list[dict], references: list[dict]) -> tuple[str, str]:
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona}'. Choose one of: {', '.join(PERSONAS)}")
    client = get_client()
    if client.available:
        digest = "\n".join(f"[{lp.clause_ref(c)}] {c['category']} | {c.get('title') or ''} | risk {c.get('risk_level')}\n"
                           f"{c['text'][:700]}" for c in clauses)
        refs = "\n".join(f"- {r['metadata'].get('title')}: {r['text'][:400]}" for r in references[:3])
        prompt = (f"{PERSONAS[persona]['prompt']}\nOnly use information supported by the document/context. Refer to "
                  f"clauses as [n].\n\nDocument type: {doc_type}\nDocument clauses:\n{digest[:14000]}\n\n"
                  f"Reference context:\n{refs or '(none)'}\n\nWrite: a 2-3 sentence overview, 'Your obligations' "
                  "bullets, 'What you receive / your rights' bullets, 'Potential risks' bullets, and 'Questions to ask' bullets.")
        try:
            return client.generate(SYSTEM_PROMPT, prompt, max_tokens=900), client.label
        except LLMError as exc:
            return rule_based(persona, doc_type, clauses), f"rule-based (LLM error: {str(exc)[:80]})"
    return rule_based(persona, doc_type, clauses), "rule-based persona template"
