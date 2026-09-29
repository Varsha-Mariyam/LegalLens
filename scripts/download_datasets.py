"""Download external datasets.

CUAD v1 (Contract Understanding Atticus Dataset, The Atticus Project, CC BY 4.0) — 510 real commercial
contracts with expert annotations for 41 clause types. Source: https://github.com/TheAtticusProject/cuad

Outputs
  data/external/cuad/CUADv1.json                 full dataset (not committed; re-download any time)
  data/datasets/cuad_annotated_spans.csv         every expert-annotated clause span with its CUAD label
  data/raw_documents/cuad/*.txt                  a selection of full contracts for pipeline testing
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
import zipfile

import _bootstrap  # noqa: F401
from backend.config import settings

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
EXTERNAL = settings.DATA_DIR / "external" / "cuad"
METADATA_LABELS = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}


def download_cuad(force: bool = False) -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    target = EXTERNAL / "CUADv1.json"
    if target.exists() and not force:
        print(f"[cuad] already present: {target}")
        return
    print(f"[cuad] downloading {CUAD_URL} ...")
    try:
        with urllib.request.urlopen(CUAD_URL, timeout=120) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"[cuad] download failed: {exc}\n"
              f"Manual step: download {CUAD_URL}, unzip it and place CUADv1.json in {EXTERNAL}")
        sys.exit(1)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("CUADv1.json") as src:
            target.write_bytes(src.read())
    print(f"[cuad] saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def export_spans() -> int:
    data = json.loads((EXTERNAL / "CUADv1.json").read_text(encoding="utf-8"))["data"]
    out = settings.DATASETS_DIR / "cuad_annotated_spans.csv"
    rows = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["contract", "cuad_label", "text"])
        for contract in data:
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    label = qa["id"].split("__")[-1]
                    if label in METADATA_LABELS:
                        continue
                    for ans in qa["answers"]:
                        text = re.sub(r"\s+", " ", ans["text"]).strip()
                        if text:
                            w.writerow([contract["title"], label, text])
                            rows += 1
    print(f"[cuad] wrote {rows} annotated spans -> {out}")
    return rows


def export_contracts(limit: int = 12) -> None:
    data = json.loads((EXTERNAL / "CUADv1.json").read_text(encoding="utf-8"))["data"]
    dest = settings.RAW_DIR / "cuad"
    dest.mkdir(parents=True, exist_ok=True)
    wanted = ["Service", "Consulting", "Maintenance", "Outsourcing", "Supply", "Distributor", "License"]
    picked = []
    for kw in wanted:
        for c in data:
            ctx = c["paragraphs"][0]["context"]
            if kw.lower() in c["title"].lower() and 8_000 < len(ctx) < 45_000 and c not in picked:
                picked.append(c)
                break
    for c in data:
        if len(picked) >= limit:
            break
        ctx = c["paragraphs"][0]["context"]
        if c not in picked and 8_000 < len(ctx) < 30_000:
            picked.append(c)
    for c in picked[:limit]:
        name = re.sub(r"[^A-Za-z0-9]+", "_", c["title"])[:90].strip("_") + ".txt"
        (dest / name).write_text(c["paragraphs"][0]["context"], encoding="utf-8")
    (dest / "SOURCE.md").write_text(
        "# CUAD v1 contracts\n\nFull-text commercial contracts from the Contract Understanding Atticus Dataset (CUAD v1),\n"
        "The Atticus Project, licensed CC BY 4.0. https://github.com/TheAtticusProject/cuad\n\n"
        "Hendrycks, Burns, Chen and Ball, *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021.\n",
        encoding="utf-8")
    print(f"[cuad] exported {min(limit, len(picked))} contracts -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--contracts", type=int, default=12)
    args = ap.parse_args()
    download_cuad(args.force)
    export_spans()
    export_contracts(args.contracts)
