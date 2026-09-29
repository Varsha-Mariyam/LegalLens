"""Central configuration. Values come from environment variables or a `.env` file in the project root."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Settings:
    ROOT = ROOT
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw_documents"
    PROCESSED_DIR = DATA_DIR / "processed_documents"
    CLAUSES_DIR = DATA_DIR / "clauses"
    STANDARD_DIR = DATA_DIR / "standard_documents"
    KNOWLEDGE_DIR = DATA_DIR / "legal_knowledge"
    DATASETS_DIR = DATA_DIR / "datasets"
    UPLOAD_DIR = DATA_DIR / "uploads"
    MODELS_DIR = ROOT / "models"
    VECTOR_DB_DIR = ROOT / "vector_db"
    CACHE_DIR = ROOT / "cache"
    REPORTS_DIR = ROOT / "reports"
    FRONTEND_DIST = ROOT / "frontend" / "dist"

    DATABASE_URL = _env("DATABASE_URL", f"sqlite:///{(ROOT / 'data' / 'legallens.db').as_posix()}")
    SECRET_KEY = _env("SECRET_KEY", "change-this-secret-key-in-production")
    TOKEN_EXPIRE_HOURS = int(_env("TOKEN_EXPIRE_HOURS", "24"))
    MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "20"))
    CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

    # LLM: auto | anthropic | openai | ollama | none
    LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()
    ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    OPENAI_API_KEY = _env("OPENAI_API_KEY")
    OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = _env("OLLAMA_MODEL")
    LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "90"))

    # Embeddings: auto | sbert | lsa
    EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "auto").lower()
    EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store: auto | chroma | numpy
    VECTOR_STORE = _env("VECTOR_STORE", "auto").lower()

    # CAG cache: Redis if REDIS_URL is set, otherwise in-memory + disk
    REDIS_URL = _env("REDIS_URL")

    TESSERACT_CMD = _env("TESSERACT_CMD")
    OCR_LANG = _env("OCR_LANG", "eng")

    CATEGORIES = ["Payment", "Confidentiality", "Termination", "Liability", "Dispute", "Others"]
    DOCUMENT_TYPES = ["employment", "rental", "nda", "service"]
    PERSONAS = ["individual", "employee", "business"]


settings = Settings()

for _d in [settings.PROCESSED_DIR, settings.CLAUSES_DIR, settings.UPLOAD_DIR, settings.MODELS_DIR,
           settings.VECTOR_DB_DIR, settings.CACHE_DIR, settings.REPORTS_DIR, settings.DATASETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
