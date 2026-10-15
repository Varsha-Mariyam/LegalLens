"""Fine-tune a Transformer encoder for clause classification or multi-label risk detection.

Data: data/datasets/contract_split/ (built by scripts/prepare_datasets.py, split by contract).
Model selection (best epoch) and risk-threshold tuning use the VALIDATION split only. The test split is not read
here; scripts/evaluate_models.py evaluates the saved models on it once.

Examples
  python scripts/train_transformer.py --task clause
  python scripts/train_transformer.py --task risk
  python scripts/train_transformer.py --task clause --model nlpaueb/legal-bert-base-uncased --tag legal-bert --batch-size 16
Runs with --tag are experiments: saved to models/transformers/experiments/<tag>_<task>/ and not used by the app.
Training can be resumed after an interruption with --resume (a checkpoint is written after every epoch).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must be set before CUDA starts

import _bootstrap  # noqa: F401
import numpy as np

from backend.config import settings
from backend.services import transformer_models as tm

DATA = settings.DATASETS_DIR / "contract_split"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------------------------------------ data

def load_split(task: str, split: str):
    path = DATA / f"{task}_{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/prepare_datasets.py")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if task == "clause":
        return [r["clause"] for r in rows], [r["label"] for r in rows]
    return [tm.risk_input(r["title"], r["text"]) for r in rows], [[l for l in r["labels"].split("|") if l] for r in rows]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def length_grouped_batches(lengths: list[int], batch_size: int, rng: np.random.Generator) -> list[list[int]]:
    """Random batches of similar length (less padding): shuffle, sort inside chunks of 20 batches, shuffle batches."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * 20
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: lengths[i])
        batches += [list(part[i:i + batch_size]) for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------------------------- metrics

def clause_metrics(y_true: list[int], probs: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    pred = probs.argmax(1)
    return {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
            "macro_f1": round(float(f1_score(y_true, pred, average="macro", labels=range(len(labels)), zero_division=0)), 4)}


def risk_metrics(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, f1_score
    have = [j for j in range(y.shape[1]) if y[:, j].sum() > 0]
    pred = (probs >= thresholds).astype(int)
    return {"macro_ap": round(float(np.mean([average_precision_score(y[:, j], probs[:, j]) for j in have])), 4),
            "micro_f1": round(float(f1_score(y, pred, average="micro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y[:, have], pred[:, have], average="macro", zero_division=0)), 4)}


def tune_thresholds(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per label, the threshold that maximises validation F1 (ties: closest to 0.5); 0.5 when no positives.
    The grid is wide because training over-samples positives, which pushes calibrated thresholds well above 0.5."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    out = np.full(y.shape[1], 0.5, dtype=np.float32)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        best = (-1.0, 0.0)
        for t in grid:
            p = probs[:, j] >= t
            tp = float((p & (y[:, j] == 1)).sum())
            f1 = 2 * tp / (p.sum() + y[:, j].sum()) if (p.sum() + y[:, j].sum()) else 0.0
            key = (round(f1, 6), -abs(t - 0.5))
            if key > best:
                best, out[j] = key, t
    return out


def retune(out_dir: Path, device: str) -> None:
    """Re-tune the risk thresholds of a saved model on the validation split (the test split is not used)."""
    saved = tm.TransformerModel(out_dir, device)
    if not saved.multi_label:
        raise SystemExit("--retune-only applies to the multi-label risk model")
    X_va, y_raw = load_split("risk", "val")
    y_va = np.array([[int(l in ls) for l in saved.labels] for ls in y_raw], dtype=np.float32)
    probs = saved.predict_proba(X_va, batch_size=64)
    thr = tune_thresholds(y_va, probs)
    config = json.loads((out_dir / tm.CONFIG_FILE).read_text(encoding="utf-8"))
    config["thresholds"] = {l: float(t) for l, t in zip(saved.labels, thr)}
    config["validation_metrics"]["at_tuned_thresholds"] = risk_metrics(y_va, probs, thr)
    config["threshold_grid"] = "0.01-0.99 step 0.01"
    (out_dir / tm.CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[retune] thresholds {config['thresholds']}\n[retune] validation {config['validation_metrics']['at_tuned_thresholds']}")


# ------------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["clause", "risk"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="any Hugging Face encoder name or local path")
    ap.add_argument("--tag", default=None, help="experiment name; output goes to models/transformers/experiments/")
    ap.add_argument("--out", default=None, help="output directory (default: models/transformers/<task>)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps with linear warm-up")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-amp", action="store_true", help="disable float16 mixed precision on CUDA")
    ap.add_argument("--fast-attention", action="store_true",
                    help="use fused attention kernels (faster on GPU, but training is then not bit-for-bit reproducible)")
    ap.add_argument("--save-fp32", action="store_true", help="store encoder weights as float32 (default float16)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only N training examples (smoke test)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--retune-only", action="store_true",
                    help="risk task: re-tune the per-label thresholds of the saved model on validation (no training)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress is visible when output is redirected to a log file

    import torch
    from torch import nn
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    import transformers

    out_dir = Path(args.out) if args.out else (
        settings.TRANSFORMER_DIR / "experiments" / f"{args.tag}_{args.task}" if args.tag else tm.model_dir(args.task))
    ckpt_dir = out_dir / "_checkpoint"
    if args.retune_only:
        retune(out_dir, args.device)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    set_seed(args.seed)
    device = tm.resolve_device(args.device)
    use_amp = device == "cuda" and not args.no_amp

    X_tr, y_tr_raw = load_split(args.task, "train")
    X_va, y_va_raw = load_split(args.task, "val")
    if args.limit:
        keep = np.random.default_rng(args.seed).permutation(len(X_tr))[: args.limit]
        X_tr, y_tr_raw = [X_tr[i] for i in keep], [y_tr_raw[i] for i in keep]
    multi = args.task == "risk"
    if multi:
        card = json.loads((DATA / "dataset_card.json").read_text(encoding="utf-8"))
        labels = card["risk_detection"]["labels"]
        y_tr = np.array([[int(l in ls) for l in labels] for ls in y_tr_raw], dtype=np.float32)
        y_va = np.array([[int(l in ls) for l in labels] for ls in y_va_raw], dtype=np.float32)
    else:
        labels = list(settings.CATEGORIES)
        y_tr = np.array([labels.index(l) for l in y_tr_raw])
        y_va = np.array([labels.index(l) for l in y_va_raw])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc_tr = tokenizer(X_tr, truncation=True, max_length=args.max_length)
    enc_va = tokenizer(X_va, truncation=True, max_length=args.max_length)
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc_tr]
    lengths = [len(x) for x in enc_tr["input_ids"]]
    truncated = float(np.mean([len(x) >= args.max_length for x in enc_tr["input_ids"]]))

    def collate(enc, idx):
        batch = tokenizer.pad({k: [enc[k][i] for i in idx] for k in keys}, return_tensors="pt")
        return {k: v.to(device) for k, v in batch.items()}

    model = tm.encoder_classifier_class().from_encoder(
        args.model, len(labels), args.dropout, attn_implementation=None if args.fast_attention else "eager").to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if multi:
        pos = y_tr.sum(0)
        pos_weight = np.clip((len(y_tr) - pos) / np.maximum(pos, 1), 1.0, 10.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
        class_weights = {l: round(float(w), 3) for l, w in zip(labels, pos_weight)}
    else:
        counts = np.bincount(y_tr, minlength=len(labels)).astype(np.float64)
        w = len(y_tr) / (len(labels) * np.maximum(counts, 1))  # "balanced", as for the Logistic Regression baseline
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        class_weights = {l: round(float(x), 3) for l, x in zip(labels, w)}

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(math.ceil(len(X_tr) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, history = 1, -1.0, []
    if args.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["rng_cpu"].cpu())  # map_location moved it to `device`; RNG states must be CPU tensors
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"[resume] continuing after epoch {ck['epoch']} (best validation {best:.4f})")
    elif log_path.exists() and not args.resume:
        log_path.unlink()

    def log(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def predict(enc) -> np.ndarray:
        model.eval()
        n = len(enc["input_ids"])
        order = sorted(range(n), key=lambda i: len(enc["input_ids"][i]))
        out = np.zeros((n, len(labels)), dtype=np.float32)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            for s in range(0, n, 64):
                idx = order[s:s + 64]
                out[idx] = model(**collate(enc, idx)).float().cpu().numpy()
        model.train()
        return 1 / (1 + np.exp(-out)) if multi else np.exp(out - out.max(1, keepdims=True)) / np.exp(out - out.max(1, keepdims=True)).sum(1, keepdims=True)

    def val_metrics(probs):
        return risk_metrics(y_va, probs, np.full(len(labels), 0.5)) if multi else clause_metrics(list(y_va), probs, labels)

    select_key = "macro_ap" if multi else "macro_f1"
    base_config = {
        "task": args.task, "multi_label": multi, "labels": labels, "base_model": args.model, "max_length": args.max_length,
        "pooling": "mean", "dropout": args.dropout, "input": "'<heading>: <clause text>' (heading optional)" if multi else "clause text",
        "training": {**{k: v for k, v in vars(args).items() if k not in ("resume",)}, "device": device, "amp": use_amp,
                     "train_examples": len(X_tr), "val_examples": len(X_va), "parameters": n_params,
                     "steps_per_epoch": steps_per_epoch, "truncated_train_fraction": round(truncated, 4),
                     "loss": "BCE with pos_weight (clipped to [1, 10])" if multi else "cross-entropy, balanced class weights",
                     "class_weights": class_weights, "selection_metric": f"validation {select_key}",
                     "data_sha256": {s: sha256(DATA / f"{args.task}_{s}.csv") for s in ("train", "val")}},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None},
    }
    print(f"[train] task={args.task} model={args.model} device={device} amp={use_amp} params={n_params:,}")
    print(f"[train] train={len(X_tr)} val={len(X_va)} steps/epoch={steps_per_epoch} truncated@{args.max_length}={truncated:.1%}")

    t_start = time.time()
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        batches = length_grouped_batches(lengths, args.batch_size, rng)
        running, t_epoch = [], time.time()
        optimizer.zero_grad(set_to_none=True)
        for b, idx in enumerate(batches, 1):
            inputs = collate(enc_tr, idx)
            target = torch.tensor(y_tr[idx], device=device) if multi else torch.tensor(y_tr[idx], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**inputs)
            loss = loss_fn(logits.float(), target)
            scaler.scale(loss / args.grad_accum).backward()
            if b % args.grad_accum == 0 or b == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running.append(loss.item())
            if b % args.log_every == 0 or b == len(batches):
                rec = {"epoch": epoch, "batch": b, "batches": len(batches), "loss": round(float(np.mean(running[-args.log_every:])), 4),
                       "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - t_start, 1)}
                log(rec)
                print(f"  epoch {epoch} batch {b}/{len(batches)} loss {rec['loss']:.4f} lr {rec['lr']:.2e}", flush=True)
        probs = predict(enc_va)
        m = val_metrics(probs)
        improved = m[select_key] > best
        rec = {"epoch": epoch, "train_loss": round(float(np.mean(running)), 4), "val": m, "epoch_seconds": round(time.time() - t_epoch, 1),
               "best": improved}
        history.append(rec)
        log({"eval": rec})
        print(f"[epoch {epoch}] train loss {rec['train_loss']:.4f} | validation {m} | {rec['epoch_seconds']}s{' | best' if improved else ''}")
        if improved:
            best = m[select_key]
            tm.save_model(model, tokenizer, out_dir, {**base_config, "best_epoch": epoch}, half=not args.save_fp32)
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None},
                   ckpt_dir / "last.pt")

    # Re-load the saved best model through the inference class used by the app, and measure *that* on validation.
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    saved = tm.TransformerModel(out_dir, device)
    probs = saved.predict_proba(X_va, batch_size=64)
    config = json.loads((out_dir / tm.CONFIG_FILE).read_text(encoding="utf-8"))
    if multi:
        thr = tune_thresholds(y_va, probs)
        config["thresholds"] = {l: float(t) for l, t in zip(labels, thr)}
        config["validation_metrics"] = {"at_0.5": risk_metrics(y_va, probs, np.full(len(labels), 0.5)),
                                        "at_tuned_thresholds": risk_metrics(y_va, probs, thr),
                                        "note": "thresholds are tuned on this split, so tuned-threshold scores are optimistic"}
    else:
        config["validation_metrics"] = clause_metrics(list(y_va), probs, labels)
    config["history"] = history
    config["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config["training"]["total_seconds"] = round(time.time() - t_start, 1)
    (out_dir / tm.CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"[done] best epoch {config['best_epoch']} | validation {config['validation_metrics']} | saved {out_dir}")
    if not args.tag and not args.out:
        tm.reload(args.task)


if __name__ == "__main__":
    main()
