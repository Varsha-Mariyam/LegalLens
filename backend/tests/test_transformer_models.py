"""Fine-tuned Transformer models: loading, predictions, persistence, fallbacks and the risk-merge policy."""
import numpy as np
import pytest

from backend.config import settings
from backend.services import classifier, risk_analyzer
from backend.services import transformer_models as tm

torch = pytest.importorskip("torch")
TRAINED = all((tm.model_dir(t) / tm.CONFIG_FILE).exists() for t in tm.TASKS)
needs_models = pytest.mark.skipif(not TRAINED, reason="run scripts/train_transformer.py first")


@pytest.fixture(autouse=True)
def _fresh_models():
    tm.reload()
    yield
    tm.reload()


@needs_models
def test_deployed_models_load_with_expected_labels():
    clause, risk = tm.get("clause"), tm.get("risk")
    assert clause.labels == settings.CATEGORIES and not clause.multi_label
    assert risk.multi_label and len(risk.labels) == 12 and set(risk.config["thresholds"]) == set(risk.labels)
    assert all(0.0 < t < 1.0 for t in risk.thresholds)
    st = tm.status()
    assert st["clause"]["available"] and st["risk"]["available"] and st["clause"]["error"] is None


@needs_models
def test_clause_probabilities_are_valid_and_primary():
    out = tm.classify(["The Licensee shall pay royalties of 5% of net sales within 30 days of each quarter end.",
                       "This Agreement shall be governed by the laws of the State of New York."])
    assert [o["label"] for o in out] == ["Payment", "Dispute"]
    for o in out:
        assert abs(sum(o["probs"].values()) - 1) < 1e-3 and 0 <= o["confidence"] <= 1
    r = classifier.classify_batch([{"text": "Either party may terminate this Agreement on 60 days' written notice.", "title": None}])[0]
    assert r["method"].startswith(("transformer:", "ensemble:")) and r["details"]["transformer"]["label"] == "Termination"
    assert {"keyword", "embedding", "transformer"} <= set(r["details"]) and r["category"] == "Termination"


@needs_models
def test_transformer_alone_is_deployed_by_default(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFIER_BACKEND", "transformer")
    monkeypatch.setattr(settings, "RISK_MODEL", "transformer")
    tm.reload()
    r = classifier.classify_batch([{"text": "The Distributor shall pay the Company a royalty of 4% of net sales.", "title": None}])[0]
    assert r["method"].startswith("transformer:") and r["category"] == "Payment"
    assert "ml" not in r["details"] and "ensemble" not in r["details"]  # Logistic Regression is not consulted
    assert r["confidence"] == r["details"]["transformer"]["confidence"]
    assert tm.risk_detector_mode() == "transformer"
    assert tm.detect_risks(["The term renews automatically for one year."]) is not None
    assert "risk_logreg" not in tm._extras  # the TF-IDF risk model was never loaded for inference
    st = classifier.status()
    assert st["primary"] == "transformer" and st["transformer"]["ensemble_weight"] is None
    assert tm.status()["risk"]["ensemble_weight"] is None


@needs_models
def test_legacy_ensemble_and_logreg_modes(monkeypatch):
    w = tm.ensemble_weight("clause")
    if w is None:
        pytest.skip("run scripts/fit_ensembles.py")
    monkeypatch.setattr(settings, "CLASSIFIER_BACKEND", "auto")
    monkeypatch.setattr(settings, "RISK_MODEL", "auto")
    tm.reload()
    r = classifier.classify_batch([{"text": "The Distributor shall pay the Company a royalty of 4% of net sales.", "title": None}])[0]
    assert r["method"].startswith("ensemble:") and r["details"]["ensemble"]["transformer_weight"] == w
    tr, lr = r["details"]["transformer"]["confidence"], r["details"]["ml"]["confidence"]
    assert r["category"] == "Payment" and min(tr, lr) - 1e-3 <= r["confidence"] <= max(tr, lr) + 1e-3
    assert classifier.status()["primary"] == "ensemble" and tm.risk_detector_mode() == "ensemble"
    monkeypatch.setattr(settings, "RISK_MODEL", "logreg")
    tm.reload()
    assert tm.risk_detector_mode() == "logreg" and tm.detect_risks(["The term renews automatically for one year."]) is not None


@needs_models
def test_risk_detector_flags_unambiguous_clauses():
    texts = [tm.risk_input("Non-Competition", "During the Term and for two (2) years thereafter, Distributor shall not "
                           "manufacture, sell or promote any product that competes with the Products in the Territory."),
             tm.risk_input("Notices", "All notices shall be in writing and delivered by hand or registered mail.")]
    hits = tm.detect_risks(texts)
    assert "Non-Compete" in [h["label"] for h in hits[0]]
    assert hits[1] == []
    probs = tm.get("risk").predict_proba(texts)
    assert probs.shape == (2, 12) and np.all((probs >= 0) & (probs <= 1))


def test_save_and_load_round_trip(tmp_path):
    """save_model -> TransformerModel reproduces the in-memory model's predictions (float16 storage)."""
    from transformers import AutoTokenizer
    base = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        tok = AutoTokenizer.from_pretrained(base)
        model = tm.encoder_classifier_class().from_encoder(base, 3).eval()
    except OSError:
        pytest.skip("base encoder not downloadable here")
    texts = ["The fee is payable monthly.", "Any dispute shall go to arbitration in Mumbai."]
    with torch.inference_mode():
        enc = tok(texts, padding=True, return_tensors="pt")
        expected = torch.softmax(model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")), -1).numpy()
    tm.save_model(model, tok, tmp_path, {"task": "clause", "multi_label": False, "labels": ["a", "b", "c"], "max_length": 64})
    loaded = tm.TransformerModel(tmp_path, "cpu")
    assert loaded.config["encoder_dtype"] == "float16"
    assert np.allclose(loaded.predict_proba(texts), expected, atol=5e-3)


def test_missing_model_falls_back_to_logistic_regression(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "TRANSFORMER_DIR", tmp_path / "nothing-here")
    tm.reload()
    assert tm.get("clause") is None and "no trained model" in tm.status()["clause"]["error"]
    r = classifier.classify_batch([{"text": "The Company shall pay a monthly salary of Rs. 50,000.", "title": None}])[0]
    assert r["method"] == "ml:tfidf-logreg" and "transformer" not in r["details"]
    assert r["category"] == "Payment"
    assert classifier.status()["primary"] == "tfidf-logreg"


def test_disabled_by_configuration(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFIER_BACKEND", "logreg")
    monkeypatch.setattr(settings, "RISK_MODEL", "off")
    tm.reload()
    assert tm.get("clause") is None and tm.status()["clause"]["error"] == "disabled by configuration"
    assert tm.detect_risks(["anything"]) is None


class _Broken:
    labels = settings.CATEGORIES
    config = {"base_model": "broken"}
    thresholds = np.full(6, 0.5)

    def predict_proba(self, texts, batch_size=32):
        raise RuntimeError("CUDA out of memory (simulated)")

    def info(self):
        return {"available": True, "base_model": "broken"}


def test_inference_failure_falls_back(caplog):
    tm.set_model("clause", _Broken())
    r = classifier.classify_batch([{"text": "Either party may terminate on thirty days' notice.", "title": None}])[0]
    assert r["method"] == "ml:tfidf-logreg" and r["category"] == "Termination"
    assert "inference failed" in tm.status()["clause"]["error"]
    tm.set_model("risk", _Broken())
    res = risk_analyzer.analyze({"text": "The Employee shall be liable for any and all losses.", "category": "Liability",
                                 "title": None}, "employment", None, [], {})
    assert res["risk_level"] == "HIGH"  # the rule result is unaffected by the detector failure
    fallback = tm.risk_probabilities(["The Employee shall be liable for any and all losses."])
    if tm.risk_logreg() is not None and tm.ensemble_config().get("risk", {}).get("logreg_thresholds"):
        assert fallback[3] == "logreg" and "learned detector" in res["method"]  # TF-IDF model takes over
    else:
        assert fallback is None and "learned detector" not in res["method"]
    assert "inference failed" in tm.status()["risk"]["error"]


def _det(label, p=0.9):
    return {"label": label, "probability": p, "threshold": 0.5}


def test_model_signal_policy():
    mutual = {"text": "Either party may terminate this Agreement for convenience on 30 days' notice."}
    one_sided = {"text": "The Company may terminate this Agreement for convenience at any time."}
    assert risk_analyzer.model_signals(mutual, [_det("Termination For Convenience")])[0]["level"] == "LOW"
    assert risk_analyzer.model_signals(one_sided, [_det("Termination For Convenience")])[0]["level"] == "MEDIUM"
    assert risk_analyzer.model_signals(one_sided, [_det("Uncapped Liability", 0.9)])[0]["level"] == "HIGH"
    assert risk_analyzer.model_signals(one_sided, [_det("Uncapped Liability", 0.3)])[0]["level"] == "MEDIUM"
    assert risk_analyzer.model_signals(one_sided, [_det("Anti-Assignment")])[0]["level"] == "LOW"
    assert risk_analyzer.model_signals(one_sided, [_det("Unknown type")]) == []
    ev = risk_analyzer.model_signals(one_sided, [_det("Exclusivity")])[0]
    assert ev["kind"] == "model" and ev["rule"] == "model:Exclusivity" and ev["source"] == risk_analyzer.MODEL_SOURCE


def test_detector_never_lowers_rule_level_and_mutual_notice_stays_low():
    high = {"text": "The Company may terminate this Agreement at any time by giving seven (7) days' notice.",
            "category": "Termination", "title": None}
    r = risk_analyzer.analyze(high, "employment", None, [], {}, model_risks=[_det("Anti-Assignment")])
    assert r["risk_level"] == "HIGH" and "learned detector" in r["method"]
    mutual = {"text": "Either party may terminate this Agreement by giving thirty (30) days' written notice to the other.",
              "category": "Termination", "title": None}
    r = risk_analyzer.analyze(mutual, "employment", None, [], {}, model_risks=[_det("Termination For Convenience")])
    assert r["risk_level"] == "LOW"
    r = risk_analyzer.analyze(mutual, "employment", None, [], {}, model_risks=None)
    assert r["method"] == "rule-based risk signals"
