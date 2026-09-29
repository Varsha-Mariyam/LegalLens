"""Deterministic extraction of facts from legal text: sentences, durations, amounts, parties."""
from __future__ import annotations

import re

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "eighteen": 18, "twenty": 20,
    "twenty-one": 21, "twenty-four": 24, "thirty": 30, "forty-five": 45, "forty": 40, "sixty": 60,
    "ninety": 90, "hundred": 100, "a": 1, "an": 1,
}
UNIT_DAYS = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
_WORDS = "|".join(sorted((w for w in WORD_NUMBERS if len(w) > 2), key=len, reverse=True))
DURATION = re.compile(
    r"(?:\b(?P<digits>\d{1,4})\b|\b(?P<word>" + _WORDS + r"|a|an)\b)?\s*(?:\((?P<paren>\d{1,4})\)\s*)?"
    r"(?:working\s+|calendar\s+|business\s+|clear\s+)?(?P<unit>hour|day|week|month|year)s?(?:'s|')?\b",
    re.IGNORECASE)
AMOUNT = re.compile(
    r"(?P<cur>Rs\.?|INR|\u20b9|USD|US\$|\$)\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>lakhs?|crores?|million|thousand)?",
    re.IGNORECASE)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s+cent)", re.IGNORECASE)
_ABBREV = ["Mr.", "Mrs.", "Ms.", "Dr.", "Rs.", "No.", "Nos.", "Pvt.", "Ltd.", "Co.", "Inc.", "St.", "v.", "vs.",
           "i.e.", "e.g.", "etc.", "Art.", "Sec.", "cl.", "U.S.", "S.A."]


def sentences(text: str) -> list[str]:
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, ab.replace(".", f"\u0000{i}\u0000"))
    parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z(\"'0-9])", protected)
    out = []
    for p in parts:
        p = re.sub(r"\u0000\d+\u0000", ".", p).strip()
        if p:
            out.append(p)
    return out


def extract_durations(text: str) -> list[dict]:
    results = []
    for m in DURATION.finditer(text):
        value = None
        if m.group("paren"):
            value = int(m.group("paren"))
        elif m.group("digits"):
            value = int(m.group("digits"))
        elif m.group("word"):
            value = WORD_NUMBERS.get(m.group("word").lower())
        if value is None:
            continue
        unit = m.group("unit").lower()
        results.append({"value": value, "unit": unit, "days": round(value * UNIT_DAYS[unit], 2),
                        "text": m.group(0).strip()})
    return results


def _to_number(num: str, scale: str | None) -> float:
    value = float(num.replace(",", ""))
    if scale:
        s = scale.lower()
        if s.startswith("lakh"):
            value *= 100_000
        elif s.startswith("crore"):
            value *= 10_000_000
        elif s == "million":
            value *= 1_000_000
        elif s == "thousand":
            value *= 1_000
    return value


def extract_amounts(text: str) -> list[dict]:
    out = []
    for m in AMOUNT.finditer(text):
        cur = m.group("cur").upper().replace("RS.", "INR").replace("RS", "INR").replace("\u20b9", "INR")
        cur = "USD" if "$" in cur else cur
        try:
            value = _to_number(m.group("num"), m.group("scale"))
        except ValueError:
            continue
        out.append({"value": value, "currency": cur, "text": m.group(0).strip()})
    return out


def extract_percentages(text: str) -> list[float]:
    return [float(x) for x in PERCENT.findall(text)]


def format_amount(value: float, currency: str = "INR") -> str:
    if currency == "INR":
        s = f"{int(round(value)):d}"
        if len(s) > 3:
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            if head:
                groups.insert(0, head)
            s = ",".join(groups + [tail])
        return f"Rs. {s}"
    return f"{currency} {value:,.0f}"


def notice_periods(text: str) -> list[dict]:
    """Durations that appear in sentences about notice, with a guess of which party they bind."""
    found = []
    for sent in sentences(text):
        low = sent.lower()
        if "notice" not in low:
            continue
        subject = re.match(r"^\s*(?:the\s+)?([A-Za-z][A-Za-z ]{1,40}?)\s+(?:may|shall|must|will|can)\b", sent)
        party = subject.group(1).strip() if subject else None
        for d in extract_durations(sent):
            if d["unit"] in ("day", "week", "month") and d["days"] <= 400:
                found.append({**d, "sentence": sent, "party": party})
    return found


def defined_parties(text: str) -> list[str]:
    roles = re.findall(r"\((?:the\s+|hereinafter\s+(?:referred\s+to\s+as\s+)?(?:the\s+)?)?\"([A-Z][A-Za-z ]{1,40})\"\)", text)
    seen = []
    for r in roles:
        if r not in seen:
            seen.append(r)
    return seen


def party_names(text: str) -> list[str]:
    m = re.search(r"\bbetween\s+(.{3,160}?)\s+and\s+(.{3,160}?)(?:\.|$|\n)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    names = []
    for chunk in m.groups():
        chunk = re.split(r",|\(|\bresiding\b|\bhaving\b|\bowner of\b", chunk)[0].strip()
        if chunk:
            names.append(chunk)
    return names


def contains_any(text: str, patterns) -> str | None:
    """Return the first matching regex pattern text found in text (case-insensitive)."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def clause_ref(clause: dict) -> str:
    """How a clause is referred to in generated text: its printed number (e.g. '5' or 'IV'), else its position."""
    return str(clause.get("label") or clause["number"])
