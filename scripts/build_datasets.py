"""Build Dataset B (clause classification) and preprocess Dataset A documents.

1. Merge project-authored seed labels with CUAD expert annotations mapped to the six LegalLens categories.
2. Deduplicate, filter and split into train/test (stratified, seed 42).
3. Preprocess every raw document: extract -> clean -> segment; write processed text and clause JSON.
4. Fit the LSA fallback embedder on the resulting corpus.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict


import _bootstrap  # noqa: F401
from backend.config import settings

# CUAD label -> LegalLens category. Labels not listed are excluded (metadata or ambiguous).
CUAD_MAP = {
    "Revenue/Profit Sharing": "Payment", "Minimum Commitment": "Payment", "Price Restrictions": "Payment",
    "Most Favored Nation": "Payment",
    "Termination For Convenience": "Termination", "Notice Period To Terminate Renewal": "Termination",
    "Post-Termination Services": "Termination",
    "Cap On Liability": "Liability", "Uncapped Liability": "Liability", "Liquidated Damages": "Liability",
    "Governing Law": "Dispute",
    "Anti-Assignment": "Others", "Audit Rights": "Others", "Change Of Control": "Others", "Exclusivity": "Others",
    "Non-Compete": "Others", "No-Solicit Of Employees": "Others", "No-Solicit Of Customers": "Others",
    "Non-Disparagement": "Others", "Insurance": "Others", "Third Party Beneficiary": "Others",
    "Ip Ownership Assignment": "Others", "License Grant": "Others", "Warranty Duration": "Others",
    "Source Code Escrow": "Others", "Rofr/Rofo/Rofn": "Others",
}
MAX_PER_CUAD_LABEL = 150
# Section headings written by the contract drafters, used as weak labels (the method used to build LEDGAR).
HEADING_MAP = [
    (r"^(confidential|confidentiality|non-?disclosure|proprietary information|trade secrets)", "Confidentiality"),
    (r"^(governing law|arbitration|dispute|jurisdiction|choice of law|venue)", "Dispute"),
    (r"^(payment|payments|fees|compensation|price|pricing|invoic|royalt)", "Payment"),
    (r"^(termination|term and termination|effect of termination)", "Termination"),
    (r"^(limitation of liability|limitations of liability|indemnif|indemnity|liability)", "Liability"),
]
MAX_PER_HEADING_CATEGORY = 200
MIN_CHARS, MAX_CHARS = 50, 1500


def build_clause_dataset() -> list[dict]:
    rows: list[dict] = []
    with (settings.DATASETS_DIR / "seed_clauses_authored.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"clause": r["clause"], "label": r["label"], "source": "seed_authored", "origin_label": r["label"]})
    spans = settings.DATASETS_DIR / "cuad_annotated_spans.csv"
    if spans.exists():
        per_label: dict[str, list] = defaultdict(list)
        with spans.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["cuad_label"] in CUAD_MAP and MIN_CHARS <= len(r["text"]) <= MAX_CHARS:
                    per_label[r["cuad_label"]].append(r["text"])
        rng = random.Random(42)
        for label, texts in sorted(per_label.items()):
            texts = sorted(set(texts))
            rng.shuffle(texts)
            for t in texts[:MAX_PER_CUAD_LABEL]:
                rows.append({"clause": t, "label": CUAD_MAP[label], "source": "cuad_v1", "origin_label": label})
        rows += heading_weak_labels()
    else:
        print("[dataset] CUAD spans not found — run scripts/download_datasets.py for the full dataset. Using seed only.")
    seen, dedup = set(), []
    for r in rows:
        key = r["clause"].lower().strip()
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    out = settings.DATASETS_DIR / "clause_dataset.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clause", "label", "source", "origin_label"])
        w.writeheader()
        w.writerows(dedup)
    print(f"[dataset] {len(dedup)} labelled clauses -> {out}")
    print("[dataset] label distribution:", dict(Counter(r["label"] for r in dedup)))
    by_label = defaultdict(list)
    for r in dedup:
        by_label[r["label"]].append(r)
    rng = random.Random(42)
    train, test = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_test = max(2, int(round(len(items) * 0.2)))
        test += items[:n_test]
        train += items[n_test:]
    for name, part in (("train", train), ("test", test)):
        with (settings.DATASETS_DIR / f"clause_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["clause", "label", "source", "origin_label"])
            w.writeheader()
            w.writerows(part)
    print(f"[dataset] train={len(train)} test={len(test)}")
    return dedup


def heading_weak_labels() -> list[dict]:
    """Segment every CUAD contract and label clauses whose own heading clearly names a category."""
    import re
    from backend.services.clause_service import segment
    from backend.services.text_cleaning import clean_text
    src = settings.DATA_DIR / "external" / "cuad" / "CUADv1.json"
    if not src.exists():
        # Reuse the weak labels already in the shipped dataset so a rebuild without the 40 MB download is identical.
        existing = settings.DATASETS_DIR / "clause_dataset.csv"
        if existing.exists():
            with existing.open(encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if r["source"] == "cuad_heading_weak"]
            print(f"[dataset] CUADv1.json not found — reusing {len(rows)} heading-derived weak labels from clause_dataset.csv")
            return rows
        print("[dataset] CUADv1.json not found — no heading-derived weak labels (run scripts/download_datasets.py)")
        return []
    contracts = json.loads(src.read_text(encoding="utf-8"))["data"]
    buckets: dict[str, list[str]] = defaultdict(list)
    for c in contracts:
        for clause in segment(clean_text(c["paragraphs"][0]["context"])):
            title = (clause["title"] or "").strip().lower()
            if not title or not (MIN_CHARS <= len(clause["text"]) <= MAX_CHARS):
                continue
            for pattern, label in HEADING_MAP:
                if re.match(pattern, title):
                    buckets[label].append(clause["text"])
                    break
    rng = random.Random(7)
    out = []
    for label, texts in sorted(buckets.items()):
        texts = sorted(set(texts))
        rng.shuffle(texts)
        for t in texts[:MAX_PER_HEADING_CATEGORY]:
            out.append({"clause": t, "label": label, "source": "cuad_heading_weak", "origin_label": "section heading"})
    print("[dataset] heading-derived weak labels:", {k: min(len(set(v)), MAX_PER_HEADING_CATEGORY) for k, v in buckets.items()})
    return out


def preprocess_documents() -> None:
    from backend.services import clause_service, document_type, extraction_service
    files = [p for p in sorted(settings.RAW_DIR.rglob("*"))
             if p.suffix.lower() in extraction_service.SUPPORTED and p.name.lower() != "readme.md"]
    index = []
    for p in files:
        try:
            result = extraction_service.extract(str(p))
        except extraction_service.ExtractionError as exc:
            print(f"[preprocess] skip {p.name}: {exc}")
            continue
        clauses = clause_service.segment(result["text"])
        dtype, _ = document_type.detect(result["text"])
        rel = p.relative_to(settings.RAW_DIR)
        stem = str(rel.with_suffix("")).replace("/", "__").replace("\\", "__") + "__" + p.suffix.lstrip(".").lower()
        (settings.PROCESSED_DIR / f"{stem}.txt").write_text(result["text"], encoding="utf-8")
        (settings.CLAUSES_DIR / f"{stem}.json").write_text(json.dumps(
            {"source": str(rel), "document_type": dtype, "extraction": result["methods"], "clauses": clauses},
            indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"source": str(rel), "document_type": dtype, "clauses": len(clauses), "methods": result["methods"]})
        print(f"[preprocess] {rel}: {len(clauses)} clauses ({dtype}, {'/'.join(result['methods'])})")
    (settings.PROCESSED_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def fit_embedder() -> None:
    from backend.services.embedding_service import LSAEmbedder, build_corpus
    corpus = build_corpus()
    LSAEmbedder.fit_and_save(corpus)
    print(f"[embeddings] LSA fallback embedder fitted on {len(corpus)} texts -> models/lsa_embedder.joblib")


if __name__ == "__main__":
    build_clause_dataset()
    preprocess_documents()
    fit_embedder()
    from refresh_models import record_versions
    record_versions()
