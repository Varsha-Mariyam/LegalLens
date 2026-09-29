"""Test configuration: isolated SQLite DB, uploads, cache and reports in a temp directory; no LLM (deterministic)."""
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="legallens_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.db').as_posix()}"
os.environ["LLM_PROVIDER"] = "none"
os.environ["REDIS_URL"] = ""
os.environ["SECRET_KEY"] = "test-secret"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from backend.config import settings  # noqa: E402

for name in ("UPLOAD_DIR", "PROCESSED_DIR", "CLAUSES_DIR", "CACHE_DIR", "REPORTS_DIR"):
    d = TMP / name.lower()
    d.mkdir(parents=True, exist_ok=True)
    setattr(settings, name, d)

SAMPLES = ROOT / "data" / "raw_documents" / "samples"


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES


@pytest.fixture(autouse=True)
def _no_llm():
    """Every test starts with the rule-based mode; LLM tests inject their own mock client."""
    from backend.services import llm_service
    llm_service.set_client(llm_service.LLMClient(provider="none"))
    yield
    llm_service.set_client(llm_service.LLMClient(provider="none"))
