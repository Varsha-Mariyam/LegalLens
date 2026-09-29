"""LegalLens FastAPI application. Run: uvicorn backend.main:app --reload"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.db import init_db
from backend.routes import analysis, auth, comparison, documents, qa, report, system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="LegalLens API", version="1.0.0", lifespan=lifespan,
              description="AI-assisted legal document analysis (academic project; not legal advice).")
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
for r in (auth, documents, analysis, qa, comparison, report, system):
    app.include_router(r.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


if settings.FRONTEND_DIST.exists() and (settings.FRONTEND_DIST / "index.html").exists():
    assets = settings.FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (settings.FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and settings.FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(settings.FRONTEND_DIST / "index.html")
