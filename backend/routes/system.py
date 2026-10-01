"""/api/system — component status, knowledge sources, re-indexing."""
import json

from fastapi import APIRouter, Depends

from backend.config import settings
from backend.database.models import User
from backend.routes.security import get_current_user
from backend.services import (cache_service, classifier, embedding_service, llm_service, ocr_service, rag_service,
                              retrieval, transformer_models)

router = APIRouter(prefix="/api/system", tags=["system"])


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@router.get("/status")
def system_status():
    metrics = _json(settings.MODELS_DIR / "classifier_metrics.json")
    index = _json(rag_service.META_FILE)
    emb = embedding_service.status()
    ret_cfg = retrieval.load_config(emb["backend"])
    return {"llm": llm_service.status(), "embedding": emb, "rag_index": index,
            "cache": cache_service.status(), "ocr": ocr_service.status(), "classifier": classifier.status(),
            "risk_detector": transformer_models.status()["risk"],
            "qa_retrieval": {"mode": settings.QA_RETRIEVAL, "fusion": ret_cfg["fusion"], "alpha": ret_cfg["alpha"],
                             "category_boost": ret_cfg["category_boost"],
                             "calibrated_not_found": bool(ret_cfg.get("answerability"))},
            "classifier_metrics": metrics, "document_types": settings.DOCUMENT_TYPES, "personas": settings.PERSONAS,
            "categories": settings.CATEGORIES}


@router.get("/knowledge")
def knowledge_sources():
    return rag_service.list_sources()


@router.post("/reindex")
def reindex(user: User = Depends(get_current_user)):
    return rag_service.build_index(force=True)
