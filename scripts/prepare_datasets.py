"""Build the contract-level datasets used by the Transformer upgrade (CUAD v1 expert annotations only).

Why a new builder: the original clause dataset (scripts/build_datasets.py) was split randomly *by clause*, so
clauses from the same contract appear in both train and test, and it mixes in 131 project-authored clauses. Here:

  * only real CUAD v1 contract text is used (the authored seed clauses are excluded);
  * contracts are grouped by filing company (several CUAD contracts are parts/exhibits of one filing with shared
    boilerplate) and the *groups* are split 70/15/15 into train/validation/test, stratified by contract type,
    with a fixed seed -- no test contract (or sibling contract) is ever seen in training;
  * identical texts are removed across splits (test copy kept, then validation, then train);
  * every contract is segmented with the application's own segmenter (backend.services.clause_service), so the
    risk detector and the retrieval benchmark are trained/evaluated on exactly the units LegalLens analyses.

Outputs (data/datasets/contract_split/):
  splits.json                    contract -> split, group and type
  clause_{train,val,test}.csv    6-category clause classification (same label scheme and CUAD mapping as before)
  risk_{train,val,test}.csv      multi-label risk detection over segmented clauses (12 CUAD risk types)
  retrieval_{val,test}.jsonl     CUAD questions over segmented contracts (retrieval / "not found" evaluation)
  dataset_card.json              sizes, label distributions, alignment coverage, licence, source checksum

Usage:  python scripts/download_datasets.py      # once: fetches CUADv1.json
        python scripts/prepare_datasets.py [--seed 42] [--neg-ratio 2.0]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from build_datasets import CUAD_MAP, HEADING_MAP, MAX_CHARS, MIN_CHARS

from backend.config import settings
from backend.services.clause_service import segment

CUAD_JSON = settings.DATA_DIR / "external" / "cuad" / "CUADv1.json"
OUT = settings.DATASETS_DIR / "contract_split"
METADATA_LABELS = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}

# 12 CUAD clause types that a reviewer would flag as potential risks. The two no-solicit types are merged.
RISK_LABELS = ["Uncapped Liability", "Non-Compete", "Non-Solicitation", "Exclusivity", "Liquidated Damages",
               "Change Of Control", "Anti-Assignment", "Termination For Convenience", "Ip Ownership Assignment",
               "Covenant Not To Sue", "Minimum Commitment", "Renewal Term"]
RISK_SOURCE = {label: label for label in RISK_LABELS}
RISK_SOURCE.update({"No-Solicit Of Employees": "Non-Solicitation", "No-Solicit Of Customers": "Non-Solicitation"})
del RISK_SOURCE["Non-Solicitation"]

# Contract type from the CUAD title (used only to stratify the split). First match wins.
CONTRACT_TYPES = [("STRATEGIC ALLIANCE", "Strategic Alliance"), ("JOINT VENTURE", "Joint Venture"),
                  ("JOINT FILING", "Joint Filing"), ("BRANDING", "Co-Branding"), ("SPONSORSHIP", "Sponsorship"),
                  ("ENDORSEMENT", "Endorsement"), ("FRANCHISE", "Franchise"), ("DISTRIBUT", "Distributor"),
                  ("RESELLER", "Reseller"), ("SUPPLY", "Supply"), ("MANUFACTUR", "Manufacturing"),
                  ("OUTSOURCING", "Outsourcing"), ("HOSTING", "Hosting"), ("MAINTENANCE", "Maintenance"),
                  ("DEVELOPMENT", "Development"), ("COLLABORATION", "Collaboration"), ("COOPERATION", "Cooperation"),
                  ("INTELLECTUAL PROPERTY", "IP"), ("LICENS", "License"), ("AGENCY", "Agency"),
                  ("AFFILIATE", "Affiliate"), ("PROMOTION", "Promotion"), ("MARKETING", "Marketing"),
                  ("CONSULTING", "Consulting"), ("TRANSPORTATION", "Transportation"), ("COMPETITION", "Non-Compete"),
                  ("SERVIC", "Service")]
SPLIT_FRACTIONS = {"test": 0.15, "val": 0.15}
MIN_SEGMENT_CHARS = 40


# ----------------------------------------------------------------------------------------------- split

def contract_type(title: str) -> str:
    tail = re.split(r"[-_]", title)[-1].upper()
    for source in (tail, title.upper()):
        for key, name in CONTRACT_TYPES:
            if key in source:
                return name
    return "Other"


def filing_group(title: str) -> str:
    """Contracts filed by the same company share boilerplate; they must land in the same split."""
    return re.sub(r"[^A-Z0-9]", "", title.split("_")[0].upper()) or title


def split_contracts(contracts: list[dict], seed: int) -> dict[str, dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    for c in contracts:
        groups[filing_group(c["title"])].append(c["title"])
    by_type: dict[str, list[str]] = defaultdict(list)
    types = {c["title"]: contract_type(c["title"]) for c in contracts}
    for g, titles in groups.items():
        by_type[Counter(types[t] for t in titles).most_common(1)[0][0]].append(g)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for ctype in sorted(by_type):
        gs = sorted(by_type[ctype])
        rng.shuffle(gs)
        n = sum(len(groups[g]) for g in gs)
        quota = {s: SPLIT_FRACTIONS[s] * n for s in ("test", "val")}
        filled = {"test": 0, "val": 0}
        for g in gs:
            size = len(groups[g])
            # give the group to the split that is furthest below its quota (test first), otherwise train
            gaps = {s: quota[s] - filled[s] for s in ("test", "val")}
            target = max(gaps, key=gaps.get)
            if gaps[target] >= size / 2:
                assignment[g] = target
                filled[target] += size
            else:
                assignment[g] = "train"
    return {t: {"split": assignment[filing_group(t)], "group": filing_group(t), "type": types[t]}
            for t in sorted(types)}


# ------------------------------------------------------------------------------------------- alignment

def _alnum_index(text: str) -> tuple[str, list[int]]:
    """Lower-case alphanumeric characters of `text` and, for each, its index in `text`.
    Makes matching immune to the whitespace/quote/hyphen/page-number normalisation done by clean_text."""
    chars, idx = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if c.isalnum():
                chars.append(c)
                idx.append(i)
    return "".join(chars), idx


def locate_segments(context: str, segments: list[dict]) -> list[tuple[int, int] | None]:
    """Character interval of every segment in the original CUAD context (None when it cannot be located)."""
    comp, idx = _alnum_index(context)
    pos, out = 0, []
    for seg in segments:
        s, _ = _alnum_index(seg["text"])
        if len(s) < 8:
            out.append(None)
            continue
        k = comp.find(s, pos)
        if k >= 0:
            start, end = k, k + len(s)
        else:  # text interrupted by a removed page number / header: anchor on head and tail separately
            head, tail = s[:40], s[-40:]
            k1 = comp.find(head, pos)
            k2 = comp.find(tail, k1) if k1 >= 0 else -1
            if k1 < 0 or k2 < 0 or (k2 + len(tail) - k1) > 1.5 * len(s) + 200:
                out.append(None)
                continue
            start, end = k1, k2 + len(tail)
        out.append((idx[start], idx[end - 1] + 1))
        pos = end
    return out


def overlaps(seg: tuple[int, int], span: tuple[int, int]) -> bool:
    """A segment carries a span's label when it holds most of the span, or the span covers most of the segment."""
    ov = min(seg[1], span[1]) - max(seg[0], span[0])
    if ov < 25:
        return False
    return ov >= 0.5 * (span[1] - span[0]) or ov >= 0.5 * (seg[1] - seg[0]) or ov >= 300


# ---------------------------------------------------------------------------------------------- builders

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def question_text(q: str) -> str:
    return q.split("Details:", 1)[1].strip() if "Details:" in q else q


def process_contract(c: dict) -> dict:
    para = c["paragraphs"][0]
    context = para["context"]
    spans = []  # (cuad_label, start, end, text)
    questions = {}
    for qa in para["qas"]:
        label = qa["id"].split("__")[-1]
        questions[label] = question_text(qa["question"])
        for a in qa["answers"]:
            spans.append((label, a["answer_start"], a["answer_start"] + len(a["text"]), a["text"]))
    segs = segment(context)
    loc = locate_segments(context, segs)
    for s, iv in zip(segs, loc):
        s["interval"] = iv
    return {"title": c["title"], "context": context, "spans": spans, "segments": segs, "questions": questions}


def clause_rows(doc: dict) -> list[dict]:
    """Gold rows = CUAD expert spans mapped to the six categories; weak rows = segments whose own heading names a
    category (the only source of Confidentiality examples, since CUAD has no confidentiality label)."""
    rows = []
    for label, _, _, text in doc["spans"]:
        text = _norm(text)
        if label in CUAD_MAP and MIN_CHARS <= len(text) <= MAX_CHARS:
            rows.append({"clause": text, "label": CUAD_MAP[label], "source": "cuad_v1", "origin_label": label,
                         "contract": doc["title"]})
    for s in doc["segments"]:
        title = (s["title"] or "").strip().lower()
        text = _norm(s["text"])
        if not title or not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        for pattern, label in HEADING_MAP:
            if re.match(pattern, title):
                rows.append({"clause": text, "label": label, "source": "cuad_heading_weak",
                             "origin_label": "section heading", "contract": doc["title"]})
                break
    return rows


def resolve_clause_labels(rows: list[dict]) -> tuple[list[dict], int]:
    """One label per distinct text. Expert labels beat heading labels; among expert labels a single specific
    (non-Others) category wins over Others; texts with two different specific categories are dropped."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[_key(r["clause"])].append(r)
    out, dropped = [], 0
    for key, group in by_key.items():
        gold = [r for r in group if r["source"] == "cuad_v1"]
        pool = gold or group
        specific = sorted({r["label"] for r in pool if r["label"] != "Others"})
        if len(specific) > 1:
            dropped += 1
            continue
        label = specific[0] if specific else "Others"
        keep = dict(next(r for r in pool if r["label"] == label))
        keep["contracts"] = sorted({r["contract"] for r in group})
        out.append(keep)
    return out, dropped


def risk_rows(doc: dict) -> tuple[list[dict], int, int]:
    rows, located, total = [], 0, 0
    risk_spans = [(RISK_SOURCE[l], a, b) for l, a, b, _ in doc["spans"] if l in RISK_SOURCE]
    for s in doc["segments"]:
        if len(s["text"]) < MIN_SEGMENT_CHARS:
            continue
        total += 1
        if s["interval"] is None:
            continue
        located += 1
        labels = sorted({lab for lab, a, b in risk_spans if overlaps(s["interval"], (a, b))})
        rows.append({"contract": doc["title"], "segment_id": s["id"], "title": s["title"] or "",
                     "text": s["text"], "labels": "|".join(labels)})
    return rows, located, total


def retrieval_record(doc: dict) -> dict:
    segs = [{"id": s["id"], "title": s["title"], "text": s["text"]} for s in doc["segments"]]
    by_label = defaultdict(list)
    for l, a, b, _ in doc["spans"]:
        by_label[l].append((a, b))
    queries = []
    for label, question in sorted(doc["questions"].items()):
        if label in METADATA_LABELS:
            continue
        spans = by_label.get(label, [])
        relevant = sorted({s["id"] for s in doc["segments"] if s["interval"] and any(overlaps(s["interval"], sp) for sp in spans)})
        queries.append({"label": label, "question": question, "answerable": bool(spans), "relevant": relevant})
    return {"contract": doc["title"], "segments": segs, "queries": queries}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=float, default=2.0,
                    help="risk TRAIN split: clauses without any risk label kept per positive clause (val/test keep all)")
    args = ap.parse_args()
    if not CUAD_JSON.exists():
        raise SystemExit(f"{CUAD_JSON} not found. Run: python scripts/download_datasets.py")
    raw = CUAD_JSON.read_bytes()
    contracts = json.loads(raw)["data"]
    OUT.mkdir(parents=True, exist_ok=True)

    splits = split_contracts(contracts, args.seed)
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    split_of = {t: v["split"] for t, v in splits.items()}
    print("[split] contracts:", dict(Counter(split_of.values())),
          "| filing groups:", dict(Counter({v["group"]: v["split"] for v in splits.values()}.values())))

    clause_all, risk_by_split, retrieval = [], defaultdict(list), defaultdict(list)
    seg_located = seg_total = 0
    for n, c in enumerate(contracts, 1):
        doc = process_contract(c)
        sp = split_of[c["title"]]
        clause_all += clause_rows(doc)
        rows, located, total = risk_rows(doc)
        seg_located += located
        seg_total += total
        risk_by_split[sp] += rows
        if sp in ("val", "test"):
            retrieval[sp].append(retrieval_record(doc))
        if n % 100 == 0:
            print(f"[segment] {n}/{len(contracts)} contracts")

    # ---- clause classification: resolve labels, assign split, drop cross-split duplicates
    resolved, dropped_conflicts = resolve_clause_labels(clause_all)
    order = {"test": 0, "val": 1, "train": 2}
    clause_split: dict[str, list[dict]] = defaultdict(list)
    cross_split_removed = 0
    for r in resolved:
        splits_present = sorted({split_of[c] for c in r["contracts"]}, key=order.get)
        r["split"] = splits_present[0]  # identical text in several splits: keep only the most held-out copy
        cross_split_removed += len(splits_present) - 1
        r["contract"] = next(c for c in r["contracts"] if split_of[c] == r["split"])
        clause_split[r["split"]].append(r)
    fields = ["clause", "label", "source", "origin_label", "contract"]
    for sp in ("train", "val", "test"):
        rows = sorted(clause_split[sp], key=lambda r: (r["contract"], r["clause"]))
        write_csv(OUT / f"clause_{sp}.csv", rows, fields)

    # ---- risk detection: cross-split dedupe, then negative subsampling on train only
    seen_eval = {_key(r["text"]) for sp in ("test", "val") for r in risk_by_split[sp]}
    seen_test = {_key(r["text"]) for r in risk_by_split["test"]}
    risk_final = {"test": risk_by_split["test"],
                  "val": [r for r in risk_by_split["val"] if _key(r["text"]) not in seen_test],
                  "train": [r for r in risk_by_split["train"] if _key(r["text"]) not in seen_eval]}
    risk_dupes_removed = sum(len(risk_by_split[s]) - len(risk_final[s]) for s in ("val", "train"))
    dedup_train, seen = [], set()
    for r in risk_final["train"]:  # also collapse repeated boilerplate inside train
        k = (_key(r["text"]), r["labels"])
        if k not in seen:
            seen.add(k)
            dedup_train.append(r)
    pos = [r for r in dedup_train if r["labels"]]
    neg = [r for r in dedup_train if not r["labels"]]
    rng = random.Random(args.seed)
    rng.shuffle(neg)
    neg_kept = neg[: int(round(args.neg_ratio * len(pos)))]
    risk_final["train"] = sorted(pos + neg_kept, key=lambda r: (r["contract"], r["segment_id"]))
    rfields = ["contract", "segment_id", "title", "text", "labels"]
    for sp in ("train", "val", "test"):
        write_csv(OUT / f"risk_{sp}.csv", risk_final[sp], rfields)

    for sp in ("val", "test"):
        with (OUT / f"retrieval_{sp}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in retrieval[sp]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- dataset card
    def label_counts(rows):
        return dict(sorted(Counter(r["label"] for r in rows).items()))

    def risk_counts(rows):
        cnt = Counter(l for r in rows for l in r["labels"].split("|") if l)
        return {l: cnt.get(l, 0) for l in RISK_LABELS}

    card = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"name": "CUAD v1 (Contract Understanding Atticus Dataset)", "file": "data/external/cuad/CUADv1.json",
                   "sha256": hashlib.sha256(raw).hexdigest(), "url": "https://github.com/TheAtticusProject/cuad",
                   "doi": "10.5281/zenodo.4595826", "license": "CC BY 4.0",
                   "citation": "Hendrycks, Burns, Chen and Ball. CUAD: An Expert-Annotated NLP Dataset for Legal "
                               "Contract Review. NeurIPS 2021 Datasets and Benchmarks."},
        "seed": args.seed,
        "split_method": "filing-company groups, stratified by contract type, 70/15/15 by contract count",
        "contracts": {sp: sum(1 for v in split_of.values() if v == sp) for sp in ("train", "val", "test")},
        "filing_groups": dict(Counter({v["group"]: v["split"] for v in splits.values()}.values())),
        "contract_types": dict(sorted(Counter(v["type"] for v in splits.values()).items())),
        "segmentation": {"segments_min_chars": MIN_SEGMENT_CHARS, "segments": seg_total, "located_in_source": seg_located,
                         "coverage": round(seg_located / max(1, seg_total), 4)},
        "clause_classification": {
            "labels": settings.CATEGORIES,
            "sizes": {sp: len(clause_split[sp]) for sp in ("train", "val", "test")},
            "label_distribution": {sp: label_counts(clause_split[sp]) for sp in ("train", "val", "test")},
            "gold_sizes": {sp: sum(r["source"] == "cuad_v1" for r in clause_split[sp]) for sp in ("train", "val", "test")},
            "conflicting_texts_dropped": dropped_conflicts, "cross_split_duplicates_removed": cross_split_removed,
            "excluded": "131 project-authored seed clauses (data/datasets/seed_clauses_authored.csv)",
        },
        "risk_detection": {
            "labels": RISK_LABELS, "merged": {"Non-Solicitation": ["No-Solicit Of Employees", "No-Solicit Of Customers"]},
            "unit": "clause as segmented by backend.services.clause_service (>= 40 chars)",
            "sizes": {sp: len(risk_final[sp]) for sp in ("train", "val", "test")},
            "positives": {sp: sum(1 for r in risk_final[sp] if r["labels"]) for sp in ("train", "val", "test")},
            "label_distribution": {sp: risk_counts(risk_final[sp]) for sp in ("train", "val", "test")},
            "train_negative_ratio": args.neg_ratio, "train_negatives_available": len(neg),
            "cross_split_duplicates_removed": risk_dupes_removed,
        },
        "retrieval": {sp: {"contracts": len(retrieval[sp]),
                           "queries": sum(len(r["queries"]) for r in retrieval[sp]),
                           "answerable": sum(q["answerable"] for r in retrieval[sp] for q in r["queries"]),
                           "answerable_with_located_segments": sum(bool(q["relevant"]) for r in retrieval[sp] for q in r["queries"])}
                      for sp in ("val", "test")},
    }
    (OUT / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps({k: card[k] for k in ("contracts", "segmentation")}, indent=1))
    print("[clause]", json.dumps(card["clause_classification"]["sizes"]), card["clause_classification"]["label_distribution"])
    print("[risk]", json.dumps(card["risk_detection"]["sizes"]), "positives", card["risk_detection"]["positives"])
    print("[risk] train label counts", card["risk_detection"]["label_distribution"]["train"])
    print("[retrieval]", json.dumps(card["retrieval"]))
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
