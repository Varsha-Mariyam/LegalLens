"""Detect which document category (employment / rental / nda / service) an uploaded file belongs to."""
from __future__ import annotations

import re

SIGNALS = {
    "employment": ["employee", "employer", "employment", "salary", "probation", "appointment", "designation",
                   "resign", "offer of employment", "working hours", "leave"],
    "rental": ["tenant", "landlord", "rent", "lease", "lessor", "lessee", "premises", "security deposit",
               "tenancy", "vacate"],
    "nda": ["confidential information", "disclosing party", "receiving party", "non-disclosure",
            "nondisclosure", "confidentiality agreement"],
    "service": ["service provider", "services", "client", "statement of work", "deliverables", "supplier",
                "customer", "distributor", "vendor", "purchase order", "license", "licensor"],
}


def detect(text: str) -> tuple[str, dict]:
    low = text.lower()
    scores = {}
    for dtype, words in SIGNALS.items():
        scores[dtype] = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", low)) for w in words)
    if not re.search(r"disclosing party|receiving party|non-disclosure|nondisclosure", low):
        scores["nda"] = scores["nda"] // 3
    head = low[:600]
    if "non-disclosure" in head or "confidentiality agreement" in head:
        scores["nda"] += 25
    if re.search(r"\b(rental|lease|tenancy) agreement\b", head):
        scores["rental"] += 25
    if re.search(r"\bemployment (agreement|contract)\b|\boffer of employment\b|\bappointment letter\b", head):
        scores["employment"] += 25
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "service"
    return best, scores
