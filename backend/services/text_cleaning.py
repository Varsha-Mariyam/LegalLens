"""Text cleaning: normalise extracted text and rebuild paragraphs from wrapped lines (guide section 6)."""
from __future__ import annotations

import re
import unicodedata

NUMBERED_HEADING = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){1,2})\.?\s+(?P<rest>\S.*)$|^(?P<num1>\d{1,2})[\.\)]\s+(?P<rest1>\S.*)$")
ARTICLE_HEADING = re.compile(
    r"^(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause)\s+(?P<num>[IVXLC]{1,6}|\d{1,3}(?:\.\d{1,2})*)\b[\.:\-\u2013\u2014]?\s*(?P<rest>.*)$")
ROMAN_HEADING = re.compile(r"^(?P<num>[IVX]{1,6})[\.\)]\s+(?P<rest>\S.*)$")
PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
EXECUTION = re.compile(r"^(?:IN\s+WITNESS\s+WHEREOF)", re.IGNORECASE)


def is_uppercase_heading(line: str) -> bool:
    s = line.strip().rstrip(".:")
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3 or len(s) > 80 or len(s.split()) > 10:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.85


def match_numbered_heading(line: str):
    """Return (label, level, rest) if the line starts a numbered clause, else None."""
    s = line.strip()
    m = ARTICLE_HEADING.match(s)
    if m:
        num = m.group("num")
        level = 2 if "." in num else 1
        return num, level, m.group("rest").strip()
    m = NUMBERED_HEADING.match(s)
    if m:
        if m.group("num"):
            return m.group("num"), 2, m.group("rest").strip()
        return m.group("num1"), 1, m.group("rest1").strip()
    m = ROMAN_HEADING.match(s)
    if m:
        return m.group("num"), 1, m.group("rest").strip()
    return None


def _looks_like_heading_line(s: str) -> bool:
    return bool(match_numbered_heading(s)) or is_uppercase_heading(s) or bool(EXECUTION.match(s))


def _is_short_title(s: str) -> bool:
    return len(s) <= 80 and not re.search(r"[.;!?,]$", s) and len(s.split()) <= 10


def clean_text(text: str) -> str:
    """Normalise unicode, drop page numbers, repair hyphenation and join wrapped lines into paragraphs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate across line breaks
    lines = [re.sub(r"[ ]{2,}", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not PAGE_NUMBER.match(ln) or ln == ""]

    paragraphs: list[str] = []
    current = ""
    for s in lines:
        if not s:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if not current:
            current = s
            continue
        if _looks_like_heading_line(s):
            paragraphs.append(current)
            current = s
        elif _is_short_title(current) and (_looks_like_heading_line(current) or s[:1].isupper()):
            paragraphs.append(current)
            current = s
        elif re.search(r"[.;:!?]$", current) and (s[:1].isupper() or s[:1] in "(\""):
            paragraphs.append(current)
            current = s
        else:
            current = f"{current} {s}"
    if current:
        paragraphs.append(current)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    expanded: list[str] = []
    for p in paragraphs:
        expanded.extend(split_inline_headings(p) if len(p) > 1200 else [p])
    return "\n\n".join(expanded)


INLINE_HEADING = re.compile(
    r"(?:(?<=[.:;\"])|(?<=[A-Z]{3}))\s+(?=(?:\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?|\d{1,2}\.|ARTICLE\s+[IVX\d]+|Section\s+\d{1,2}(?:\.\d{1,2})?\.?)\s+[A-Z][A-Za-z])")


def split_inline_headings(paragraph: str) -> list[str]:
    """Long run-on paragraphs (common in flattened PDFs, e.g. CUAD) are split before inline clause numbers."""
    parts = [x.strip() for x in INLINE_HEADING.split(paragraph) if x and x.strip()]
    return parts if len(parts) > 1 else [paragraph]
