"""/api/system — component status, knowledge sources, re-indexing."""
import json

from fastapi import APIRouter, Depends

from backend.config import settings
from backend.database.models import User
from backend.routes.security import get_current_user
from backend.services import cache_service, classifier, embedding_service, llm_service, ocr_service, rag_service

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
def system_status():
    metrics_file = settings.MODELS_DIR / "classifier_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None
    index = json.loads(rag_service.META_FILE.read_text()) if rag_service.META_FILE.exists() else None
    return {"llm": llm_service.status(), "embedding": embedding_service.status(), "rag_index": index,
            "cache": cache_service.status(), "ocr": ocr_service.status(), "classifier": classifier.status(),
            "classifier_metrics": metrics, "document_types": settings.DOCUMENT_TYPES, "personas": settings.PERSONAS,
            "categories": settings.CATEGORIES}


@router.get("/knowledge")
def knowledge_sources():
    return rag_service.list_sources()


@router.post("/reindex")
def reindex(user: User = Depends(get_current_user)):
    return rag_service.build_index(force=True)
