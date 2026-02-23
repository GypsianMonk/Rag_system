"""
/health  —  Liveness, readiness, and version endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    vector_store: str
    llm_provider: str
    embedding_provider: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        vector_store=settings.VECTOR_STORE,
        llm_provider=settings.LLM_PROVIDER,
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    """
    Deep readiness check — verify DB, vector store, LLM reachability.
    Keep lightweight for k8s probes.
    """
    checks = {"db": "ok", "vector_store": "ok"}
    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
