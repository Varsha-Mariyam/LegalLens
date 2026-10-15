"""Clause classification: keyword baseline, embedding similarity and the trained ML classifier."""
import json

import pytest

from backend.config import settings
from backend.services import classifier

GUIDE_EXAMPLES = [
    ("Employee shall receive a monthly salary of Rs. 50,000 payable on the last day of the month.", "Payment"),
    ("The employee shall maintain confidentiality of all proprietary information of the company.", "Confidentiality"),
    ("Either party may terminate this agreement by giving thirty days written notice.", "Termination"),
    ("The employee shall be liable for damages caused by gross negligence.", "Liability"),
    ("Any dispute shall be resolved through arbitration under the Arbitration and Conciliation Act, 1996.", "Dispute"),
    ("The employee shall report to the engineering manager and perform assigned duties.", "Others"),
]


@pytest.mark.parametrize("text,label", GUIDE_EXAMPLES)
def test_final_classifier_on_guide_examples(text, label):
    assert classifier.classify_batch([{"text": text, "title": None}])[0]["category"] == label


@pytest.mark.parametrize("text,label", GUIDE_EXAMPLES[:5])
def test_keyword_baseline(text, label):
    assert classifier.keyword_classify(text)[0] == label


def test_result_contains_all_three_methods():
    r = classifier.classify_batch([{"text": GUIDE_EXAMPLES[0][0], "title": "Salary"}])[0]
    assert {"keyword", "embedding"} <= set(r["details"])
    assert 0 <= r["confidence"] <= 1


def test_structural_units_are_others():
    r = classifier.classify_batch([{"text": "IN WITNESS WHEREOF the parties have signed.", "title": "Execution"}])[0]
    assert r["category"] == "Others"


def test_system_page_metrics_are_the_deployed_transformers_only():
    """models/classifier_metrics.json (System page) holds the fine-tuned MiniLM models' own test-set metrics."""
    metrics = json.loads((settings.MODELS_DIR / "classifier_metrics.json").read_text())
    assert metrics["model"] == "transformer_minilm" and "methods" not in metrics
    assert metrics["dataset"] == "contract_split" and metrics["test_contracts"] > 50
    c = metrics["classifier"]
    assert c["n"] == metrics["test_size"] and c["gold_only"]["n"] == metrics["gold_test_size"]
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.5 < c[k] <= 1.0
    lo, hi = c["macro_f1_ci95"]
    assert lo <= c["macro_f1"] <= hi
    assert set(c["per_class"]) == set(settings.CATEGORIES)
    cm = c["confusion_matrix"]
    assert cm["labels"] == settings.CATEGORIES and sum(map(sum, cm["matrix"])) == c["n"]
    assert sum(cm["matrix"][i][i] for i in range(len(cm["labels"]))) / c["n"] == pytest.approx(c["accuracy"], abs=1e-3)
    r = metrics["risk_detector"]
    assert {"micro_precision", "micro_recall", "micro_f1", "macro_f1"} <= set(r) and 0 < r["micro_f1"] <= 1
