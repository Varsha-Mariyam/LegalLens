"""Standard-document comparison (guide section 15): semantic clause matching + parameter differences."""
from __future__ import annotations

import re

import numpy as np

from backend.services import embedding_service, rag_service
from backend.services import legal_parsing as lp
from backend.services.llm_service import SYSTEM_PROMPT, LLMError, get_client

REFERENCE_LABEL = "Project reference template (not an official legal standard)"
MATCH_THRESHOLD = 0.25
EQUIVALENT_THRESHOLD = 0.75


def load_standard(doc_type: str) -> list[dict]:
    return rag_service.load_standard_template(doc_type)


def key_value(category: str, text: str) -> str:
    low = text.lower()
    if category == "Payment":
        amounts = lp.extract_amounts(text)
        if amounts:
            return ", ".join(dict.fromkeys(lp.format_amount(a["value"], a["currency"]) for a in amounts[:2]))
        periods = [d["text"] for d in lp.extract_durations(text)]
        return f"within {periods[0]}" if periods else "Present"
    if category == "Termination":
        n = lp.notice_periods(text)
        if n:
            vals = sorted({f"{p['days']:g} days" for p in n}, key=lambda v: float(v.split()[0]))
            return " / ".join(vals)
        if re.search(r"without (any )?notice|at any time", low):
            return "No notice"
        return "Present"
    if category == "Liability":
        if re.search(r"unlimited|without (any )?limit|any and all|all losses", low) and not re.search(r"shall not exceed|limited to", low):
            return "Unlimited"
        if re.search(r"shall not exceed|limited to|aggregate liability|only for", low):
            return "Limited"
        return "Not specified"
    if category == "Confidentiality":
        d = [x for x in lp.extract_durations(text) if x["unit"] == "year"]
        if re.search(r"perpetu|indefinite|forever|at any time thereafter", low):
            return "Present (no end date)"
        return f"Present ({d[0]['text']})" if d else "Present"
    if category == "Dispute":
        mode = "Arbitration" if "arbitra" in low else ("Courts" if "court" in low else "Present")
        who = ""
        if re.search(r"appointed (solely )?by the (company|employer|landlord|disclosing party|client)|nominated by the", low):
            who = ", one-sided appointment"
        elif re.search(r"mutual (agreement|consent)", low):
            who = ", mutual appointment"
        return mode + who
    return "Present"


def _rule_difference(category, std_val, user_val, std_text, user_text) -> str:
    if std_val == user_val:
        return f"Same key term ({user_val}); wording differs." if std_val != "Present" else "Both documents contain this clause."
    if category == "Termination":
        return (f"The user's termination clause specifies {user_val} notice, while the reference specifies {std_val}.")
    if category == "Liability":
        return f"The user's liability clause is '{user_val}', while the reference is '{std_val}'."
    if category == "Payment":
        return f"The payment terms differ: user document has {user_val}; reference has {std_val}."
    if category == "Dispute":
        return f"Dispute resolution differs: user document uses {user_val}; reference uses {std_val}."
    if category == "Confidentiality":
        return f"Confidentiality differs: user document — {user_val}; reference — {std_val}."
    return "The clauses cover the same topic with different terms."


def compare(doc_type: str, user_clauses: list[dict], user_embeddings) -> list[dict]:
    std = load_standard(doc_type)
    if not std:
        return []
    std_emb = embedding_service.encode([c["text"] for c in std])
    user_emb = np.asarray(user_embeddings, dtype=np.float32)
    sims = std_emb @ user_emb.T if len(user_clauses) else np.zeros((len(std), 0))
    client = get_client()
    rows, used = [], set()
    for i, sc in enumerate(std):
        best_j, best_s = None, -1.0
        for j, uc in enumerate(user_clauses):
            s = float(sims[i, j]) + (0.25 if uc["category"] == sc["category"] and sc["category"] != "Others" else 0.0)
            if s > best_s:
                best_j, best_s = j, s
        raw = float(sims[i, best_j]) if best_j is not None else 0.0
        same_cat = best_j is not None and user_clauses[best_j]["category"] == sc["category"] and sc["category"] != "Others"
        if best_j is None or (raw < MATCH_THRESHOLD and not same_cat):
            rows.append({"category": sc["category"], "status": "missing", "standard_title": sc["title"],
                         "standard_text": sc["text"], "user_clause_number": None, "user_clause_label": None, "user_title": None, "user_text": None,
                         "similarity": round(raw, 3), "standard_value": key_value(sc["category"], sc["text"]),
                         "user_value": "Not found", "difference": "The reference contains this clause but no matching clause was found in the user document.",
                         "method": "embedding match"})
            continue
        uc = user_clauses[best_j]
        used.add(best_j)
        sv, uv = key_value(sc["category"], sc["text"]), key_value(sc["category"], uc["text"])
        status = "equivalent" if sv == uv and raw >= EQUIVALENT_THRESHOLD else "differs"
        diff, method = _rule_difference(sc["category"], sv, uv, sc["text"], uc["text"]), "embedding match + rule comparison"
        if client.available and status == "differs":
            prompt = (f"Reference clause ({REFERENCE_LABEL}):\n{sc['text']}\n\nUser clause:\n{uc['text']}\n\n"
                      "In 1-3 sentences, explain the meaningful differences and why they could matter. Do not invent terms.")
            try:
                diff, method = client.generate(SYSTEM_PROMPT, prompt, max_tokens=250), f"embedding match + {client.label}"
            except LLMError as exc:  # keep the rule-based difference, record why the LLM was not used
                method = f"embedding match + rule comparison (LLM error: {str(exc)[:80]})"
        rows.append({"category": sc["category"], "status": status, "standard_title": sc["title"], "standard_text": sc["text"],
                     "user_clause_number": uc["number"], "user_clause_label": lp.clause_ref(uc), "user_title": uc.get("title"), "user_text": uc["text"],
                     "similarity": round(raw, 3), "standard_value": sv, "user_value": uv, "difference": diff, "method": method})
    for j, uc in enumerate(user_clauses):
        if j in used or (uc.get("title") or "").startswith(("Preamble", "Execution")):
            continue
        if uc["category"] != "Others" or uc.get("risk_level") in ("HIGH", "MEDIUM"):
            rows.append({"category": uc["category"], "status": "additional", "standard_title": None, "standard_text": None,
                         "user_clause_number": uc["number"], "user_clause_label": lp.clause_ref(uc), "user_title": uc.get("title"), "user_text": uc["text"],
                         "similarity": None, "standard_value": "Not in reference", "user_value": key_value(uc["category"], uc["text"]),
                         "difference": "This clause has no counterpart in the reference template; review it separately.",
                         "method": "embedding match"})
    return rows
