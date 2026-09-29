"""Deterministic clause segmentation (guide section 7) — numbering, headings and paragraphs; no LLM."""
from __future__ import annotations

import re

from backend.services.legal_parsing import sentences
from backend.services.text_cleaning import clean_text, EXECUTION, is_uppercase_heading, match_numbered_heading

SPLIT_LARGE_PARENT_CHARS = 1500
MAX_PARAGRAPH_CLAUSE_CHARS = 1800
TITLE_PREFIX = re.compile(r"^(?P<title>[A-Z][A-Za-z0-9/&,'\-\s]{1,60}?)[\.:]\s+(?P<body>\S.*)$", re.DOTALL)


def _split_title(rest: str) -> tuple[str | None, str]:
    rest = rest.strip()
    if not rest:
        return None, ""
    if len(rest) <= 80 and not re.search(r"[;,!?]$", rest) and len(rest.split()) <= 10 and (
            not rest.endswith(".") or len(rest.split()) <= 6):
        return rest.rstrip(".:").strip(), ""
    m = TITLE_PREFIX.match(rest)
    if m:
        title = m.group("title").strip()
        words = title.split()
        if len(words) <= 8 and (title.isupper() or all(w[:1].isupper() or w.lower() in {"and", "of", "to", "the", "for", "in", "or"} for w in words)):
            return title.rstrip(".:"), m.group("body").strip()
    return None, rest


def _pretty(title: str | None) -> str | None:
    if not title:
        return None
    return title.title() if title.isupper() else title


def _paragraph_mode(paragraphs: list[str]) -> list[dict]:
    units: list[dict] = []
    buffer = ""
    for p in paragraphs:
        if is_uppercase_heading(p) and not buffer:
            units.append({"label": None, "title": _pretty(p.rstrip(".:")), "text": ""})
            continue
        text = f"{buffer} {p}".strip() if buffer else p
        if len(text) < 100:
            buffer = text
            continue
        buffer = ""
        if units and units[-1]["text"] == "" and units[-1]["title"]:
            units[-1]["text"] = text
        else:
            units.append({"label": None, "title": None, "text": text})
    if buffer:
        if units:
            units[-1]["text"] = f"{units[-1]['text']} {buffer}".strip()
        else:
            units.append({"label": None, "title": None, "text": buffer})
    out = []
    for u in units:
        if not u["text"]:
            continue
        if len(u["text"]) <= MAX_PARAGRAPH_CLAUSE_CHARS:
            out.append(u)
            continue
        chunk = ""
        part = 1
        for s in sentences(u["text"]):
            if chunk and len(chunk) + len(s) > 1200:
                out.append({"label": None, "title": (f"{u['title']} (part {part})" if u["title"] else None), "text": chunk})
                chunk, part = "", part + 1
            chunk = f"{chunk} {s}".strip()
        if chunk:
            out.append({"label": None, "title": (f"{u['title']} (part {part})" if u["title"] and part > 1 else u["title"]), "text": chunk})
    return out


def segment(text: str) -> list[dict]:
    """Split text into clauses: [{"id", "label", "title", "text"}].
    The text is normalised with clean_text first (idempotent, so already-cleaned text is unchanged)."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    heads = [match_numbered_heading(p) for p in paragraphs]
    level1 = sum(1 for h in heads if h and h[1] == 1)
    level2 = sum(1 for h in heads if h and h[1] == 2)

    if level1 < 2 and level2 < 2:
        units = _paragraph_mode(paragraphs)
    else:
        promote_level2 = level1 < 2
        preamble: list[str] = []
        tops: list[dict] = []
        for p, h in zip(paragraphs, heads):
            if EXECUTION.match(p) and tops:
                tops.append({"label": None, "title": "Execution", "body": [p], "children": []})
                continue
            if h:
                label, level, rest = h
                title, body = _split_title(rest)
                unit = {"label": label, "title": title, "body": [body] if body else [], "children": []}
                if level == 1 or promote_level2 or not tops:
                    tops.append(unit)
                else:
                    tops[-1]["children"].append(unit)
                continue
            target = tops[-1] if tops else None
            if target is None:
                preamble.append(p)
            elif target["children"]:
                target["children"][-1]["body"].append(p)
            elif target["title"] is None and not target["body"] and len(p) <= 80 and not p.endswith("."):
                target["title"] = p
            else:
                target["body"].append(p)
        units = []
        if preamble:
            units.append({"label": None, "title": "Preamble / Parties", "text": "\n".join(preamble)})
        for t in tops:
            own = " ".join(t["body"]).strip()
            child_texts = [f"{c['label']} {(c['title'] + '. ') if c['title'] else ''}{' '.join(c['body'])}".strip()
                           for c in t["children"]]
            combined = " ".join([own] + child_texts).strip()
            if t["children"] and len(combined) > SPLIT_LARGE_PARENT_CHARS and len(t["children"]) >= 2:
                if len(own) > 40:
                    units.append({"label": t["label"], "title": _pretty(t["title"]), "text": own})
                for c in t["children"]:
                    ctext = " ".join(c["body"]).strip()
                    if not ctext and c["title"]:
                        ctext = c["title"]
                    ctitle = _pretty(c["title"]) or (f"{_pretty(t['title'])} ({c['label']})" if t["title"] else None)
                    if ctext:
                        units.append({"label": c["label"], "title": ctitle, "text": ctext})
            elif combined:
                units.append({"label": t["label"], "title": _pretty(t["title"]), "text": combined})
            elif t["title"]:
                units.append({"label": t["label"], "title": _pretty(t["title"]), "text": t["title"]})

    clauses = []
    for i, u in enumerate(units, start=1):
        text = re.sub(r"\s+\n", "\n", u["text"]).strip()
        if len(text) < 3:
            continue
        clauses.append({"id": len(clauses) + 1, "label": u.get("label"), "title": u.get("title"), "text": text})
    return clauses
