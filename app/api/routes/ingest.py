"""
/api/v1/ingest  —  Document ingestion endpoints
"""

import json

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.pipeline import IngestionPipeline
from app.utils.dependencies import get_ingestion_pipeline

logger = structlog.get_logger(__name__)
router = APIRouter()

MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    status: str
    message: str


class URLIngestRequest(BaseModel):
    url: str
    tenant_id: str | None = None
    metadata: dict | None = None


@router.post("/ingest/file", response_model=IngestResponse, summary="Ingest a document file")
async def ingest_file(
    file: UploadFile = File(...),
    tenant_id: str = Form(default="default"),
    metadata: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{ext}' not supported. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_FILE_SIZE_MB}MB",
        )

    meta_dict = json.loads(metadata) if metadata else {}

    result = await pipeline.ingest_file(
        file_bytes=content,
        filename=file.filename,
        tenant_id=tenant_id,
        metadata=meta_dict,
    )
    return IngestResponse(**result, message="Document ingested successfully")


@router.post("/ingest/url", response_model=IngestResponse, summary="Ingest content from a URL")
async def ingest_url(
    req: URLIngestRequest,
    db: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    result = await pipeline.ingest_url(
        url=str(req.url),
        tenant_id=req.tenant_id or "default",
        metadata=req.metadata,
    )
    return IngestResponse(**result, message="URL ingested successfully")


@router.delete("/ingest/{doc_id}", summary="Delete an ingested document")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    from app.core.database import Chunk, Document  # noqa: PLC0415
    from sqlalchemy import delete as sql_delete  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    chunks = await db.execute(select(Chunk.id).where(Chunk.document_id == doc_id))
    chunk_ids = [str(r) for r in chunks.scalars().all()]

    if not chunk_ids:
        raise HTTPException(status_code=404, detail="Document not found")

    await pipeline.vector_store.adelete(chunk_ids)
    await db.execute(sql_delete(Chunk).where(Chunk.document_id == doc_id))
    await db.execute(sql_delete(Document).where(Document.id == doc_id))

    return {"message": f"Deleted document {doc_id} ({len(chunk_ids)} chunks)"}
