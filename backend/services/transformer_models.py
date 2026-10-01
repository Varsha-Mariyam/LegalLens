"""Fine-tuned Transformer models: clause classifier (6 categories) and multi-label risk detector (12 CUAD risk types).

Architecture (shared by training and inference): a pretrained Hugging Face encoder (default
sentence-transformers/all-MiniLM-L6-v2) -> attention-masked mean pooling -> dropout -> linear head.
Single-label tasks use softmax, multi-label tasks use per-label sigmoids with thresholds tuned on the validation split.

Saved layout (models/transformers/<task>/, written by scripts/train_transformer.py):
    encoder/             encoder config, weights (safetensors) and tokenizer
    head.safetensors     linear classification head
    model_config.json    task, labels, max_length, base model, thresholds, validation metrics, training settings

torch/transformers are imported lazily, so LegalLens still starts without them. Loading failures are recorded
(status()) and every caller falls back to the classical models, so a missing or broken model never stops analysis.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.config import settings

log = logging.getLogger("legallens.transformers")
TASKS = ("clause", "risk")
CONFIG_FILE = "model_config.json"


def model_dir(task: str) -> Path:
    return settings.TRANSFORMER_DIR / task


def risk_input(title: str | None, text: str) -> str:
    """Input text of the risk detector — identical in training (scripts/train_transformer.py) and in the app."""
    return f"{title}: {text}" if title else text


def resolve_device(requested: str = "auto") -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; using CPU")
        return "cpu"
    return requested


@lru_cache(maxsize=1)
def encoder_classifier_class():
    """The nn.Module, created on first use so that importing this file never imports torch."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class EncoderClassifier(nn.Module):
        def __init__(self, encoder, num_labels: int, dropout: float = 0.1):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(encoder.config.hidden_size, num_labels)

        @classmethod
        def from_encoder(cls, name_or_path: str, num_labels: int, dropout: float = 0.1, dtype=None,
                         attn_implementation: str | None = None):
            """attn_implementation="eager" gives bit-for-bit reproducible training on CUDA (the default fused
            attention kernels have a non-deterministic backward pass); inference uses the faster default."""
            kwargs = {}
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
            try:
                encoder = AutoModel.from_pretrained(name_or_path, **kwargs, **({"dtype": dtype} if dtype is not None else {}))
            except TypeError:  # transformers < 4.56 names the argument torch_dtype
                encoder = AutoModel.from_pretrained(name_or_path, torch_dtype=dtype, **kwargs)
            return cls(encoder, num_labels, dropout)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**kwargs).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return self.head(self.dropout(pooled))

    return EncoderClassifier


def save_model(model, tokenizer, out_dir: Path, config: dict, half: bool = True) -> None:
    """Write encoder (optionally as float16 to halve the file size), tokenizer, head and config."""
    import copy

    import torch
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = copy.deepcopy(model.encoder).cpu()
    if half:
        encoder = encoder.to(torch.float16)
    encoder.save_pretrained(str(out_dir / "encoder"), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir / "encoder"))
    head = {k: v.detach().float().cpu().contiguous() for k, v in model.head.state_dict().items()}
    save_file(head, str(out_dir / "head.safetensors"))
    config = {**config, "encoder_dtype": "float16" if half else "float32"}
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")


class TransformerModel:
    """A saved EncoderClassifier ready for inference."""

    def __init__(self, path: Path, device: str = "auto"):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        self.path = Path(path)
        self.config = json.loads((self.path / CONFIG_FILE).read_text(encoding="utf-8"))
        self.task = self.config["task"]
        self.labels: list[str] = self.config["labels"]
        self.multi_label = bool(self.config.get("multi_label"))
        self.max_length = int(self.config.get("max_length", 256))
        self.thresholds = np.array([self.config.get("thresholds", {}).get(l, 0.5) for l in self.labels], dtype=np.float32)
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path / "encoder"))
        cls = encoder_classifier_class()
        try:  # fused attention for fast inference, whatever implementation was used in training
            self.model = cls.from_encoder(str(self.path / "encoder"), len(self.labels), dtype=torch.float32,
                                          attn_implementation="sdpa")
        except (ValueError, ImportError):  # architecture without SDPA support
            self.model = cls.from_encoder(str(self.path / "encoder"), len(self.labels), dtype=torch.float32)
        self.model.head.load_state_dict(load_file(str(self.path / "head.safetensors")))
        self.model.to(self.device).eval()
        self._lock = threading.Lock()

    def logits(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        with self._lock, torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer([texts[i] or " " for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
                out[idx] = logits.float().cpu().numpy()
        return out

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        z = self.logits(texts, batch_size)
        if self.multi_label:
            return 1.0 / (1.0 + np.exp(-z))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def info(self) -> dict:
        c = self.config
        return {"available": True, "task": self.task, "base_model": c.get("base_model"), "labels": self.labels,
                "max_length": self.max_length, "device": self.device, "trained_at": c.get("trained_at"),
                "path": str(self.path.relative_to(settings.ROOT)) if self.path.is_relative_to(settings.ROOT) else str(self.path),
                "validation": c.get("validation_metrics"), "load_seconds": c.get("_load_seconds")}


_models: dict[str, TransformerModel | None] = {}
_errors: dict[str, str | None] = {}
_lock = threading.Lock()
_extras: dict = {}  # ensemble config and the TF-IDF risk model, loaded once
ENSEMBLE_FILE = "ensemble_config.json"
RISK_LOGREG_FILE = "risk_logreg.joblib"


def ensemble_config() -> dict:
    """models/ensemble_config.json (written by scripts/fit_ensembles.py from validation data), or {}."""
    if "ensemble" not in _extras:
        path = settings.MODELS_DIR / ENSEMBLE_FILE
        try:
            _extras["ensemble"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", path.name, exc)
            _extras["ensemble"] = {}
    return _extras["ensemble"]


def ensemble_weight(task: str) -> float | None:
    """Transformer weight in the probability average for `task`, or None when no ensemble is configured."""
    w = ensemble_config().get(task, {}).get("transformer_weight")
    return float(w) if w is not None else None


def risk_logreg():
    """TF-IDF + one-vs-rest Logistic Regression risk model (ensemble partner and fallback), or None."""
    if "risk_logreg" not in _extras:
        path = settings.MODELS_DIR / RISK_LOGREG_FILE
        _extras["risk_logreg"] = None
        if path.exists() and settings.RISK_MODEL not in ("off", "none", "rules"):
            try:
                import joblib
                _extras["risk_logreg"] = joblib.load(path)
            except Exception as exc:  # noqa: BLE001  (e.g. pickled by an incompatible scikit-learn)
                log.warning("could not load %s: %s", path.name, exc)
    return _extras["risk_logreg"]


def enabled(task: str) -> bool:
    flag = settings.CLASSIFIER_BACKEND if task == "clause" else settings.RISK_MODEL
    return flag not in ("logreg", "off", "none", "rules")


def get(task: str) -> TransformerModel | None:
    """The loaded model for `task`, or None (disabled, not trained, or failed to load — see status())."""
    if task not in TASKS:
        raise ValueError(task)
    with _lock:
        if task in _models:
            return _models[task]
        model, error = None, None
        path = model_dir(task)
        if not enabled(task):
            error = "disabled by configuration"
        elif not (path / CONFIG_FILE).exists():
            error = f"no trained model in {path.relative_to(settings.ROOT) if path.is_relative_to(settings.ROOT) else path}" \
                    f" — run scripts/train_transformer.py --task {task}"
        else:
            try:
                t0 = time.perf_counter()
                model = TransformerModel(path, settings.TRANSFORMER_DEVICE)
                model.config["_load_seconds"] = round(time.perf_counter() - t0, 2)
                log.info("Loaded %s transformer from %s on %s", task, path, model.device)
            except Exception as exc:  # noqa: BLE001  (torch missing, corrupt files, incompatible versions...)
                error = f"could not load the {task} transformer ({type(exc).__name__}: {exc})"
                log.warning(error)
        _models[task], _errors[task] = model, error
        return model


def reload(task: str | None = None) -> None:
    with _lock:
        for t in ([task] if task else TASKS):
            _models.pop(t, None)
            _errors.pop(t, None)
        _extras.clear()


def set_model(task: str, model) -> None:
    """Inject a model (tests); None simulates a missing model."""
    with _lock:
        _models[task], _errors[task] = model, (None if model is not None else "not available (injected)")


def classify(texts: list[str]) -> list[dict] | None:
    """[{"label", "confidence", "probs"}] from the clause transformer, or None when it is unavailable or fails."""
    model = get("clause")
    if model is None:
        return None
    try:
        probs = model.predict_proba(texts)
    except Exception as exc:  # noqa: BLE001
        log.warning("clause transformer inference failed: %s", exc)
        _errors["clause"] = f"inference failed ({type(exc).__name__}: {exc})"
        return None
    return [{"label": model.labels[int(p.argmax())], "confidence": round(float(p.max()), 3),
             "probs": {l: round(float(v), 4) for l, v in zip(model.labels, p)}} for p in probs]


def risk_detector_mode() -> str | None:
    """"ensemble" (Transformer + TF-IDF, weight/thresholds from validation), "transformer", "logreg", or None."""
    if settings.RISK_MODEL in ("off", "none", "rules"):
        return None
    tr, lr, cfg = get("risk"), risk_logreg(), ensemble_config().get("risk", {})
    if tr is not None and lr is not None and cfg.get("thresholds") and settings.RISK_MODEL == "auto":
        return "ensemble"
    if tr is not None:
        return "transformer"
    if lr is not None and cfg.get("logreg_thresholds"):
        return "logreg"
    return None


def risk_probabilities(texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, str] | None:
    """(labels, probabilities, thresholds, mode) from the configured risk detector, or None when unavailable.
    If the Transformer fails at inference time, the TF-IDF model alone is used when available."""
    mode = risk_detector_mode()
    if mode is None:
        return None
    cfg = ensemble_config().get("risk", {})
    lr = risk_logreg()
    p_tr = None
    if mode in ("ensemble", "transformer"):
        model = get("risk")
        try:
            p_tr = model.predict_proba(texts)
            labels = model.labels
        except Exception as exc:  # noqa: BLE001
            log.warning("risk transformer inference failed: %s", exc)
            _errors["risk"] = f"inference failed ({type(exc).__name__}: {exc})"
            if lr is None or not cfg.get("logreg_thresholds"):
                return None
            mode = "logreg"
        if mode == "transformer":
            return labels, p_tr, model.thresholds, mode
    lr_labels = lr["labels"]
    p_lr = lr["pipeline"].predict_proba(texts) if texts else np.zeros((0, len(lr_labels)))
    if mode == "logreg":
        return lr_labels, p_lr, np.array([cfg["logreg_thresholds"][l] for l in lr_labels]), mode
    w = float(cfg["transformer_weight"])
    p_tr = p_tr[:, [labels.index(l) for l in lr_labels]]
    return lr_labels, w * p_tr + (1 - w) * p_lr, np.array([cfg["thresholds"][l] for l in lr_labels]), mode


def detect_risks(texts: list[str]) -> list[list[dict]] | None:
    """Per text, the risk types whose probability reaches the type's validation-tuned threshold (highest first)."""
    res = risk_probabilities(texts)
    if res is None:
        return None
    labels, probs, thresholds, _ = res
    out = []
    for p in probs:
        hits = [{"label": l, "probability": round(float(v), 3), "threshold": round(float(t), 3)}
                for l, v, t in zip(labels, p, thresholds) if v >= t]
        out.append(sorted(hits, key=lambda h: -h["probability"]))
    return out


def status() -> dict:
    out = {}
    for task in TASKS:
        model = get(task)
        out[task] = model.info() if model else {"available": False}
        out[task]["error"] = _errors.get(task)
        out[task]["ensemble_weight"] = ensemble_weight(task)
    out["risk"]["detector"] = risk_detector_mode()
    out["risk"]["logreg_available"] = risk_logreg() is not None
    return out
