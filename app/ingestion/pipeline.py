"""
Ingestion Pipeline
==================
Orchestrates: load → chunk → embed → index → persist metadata

Supports: PDF, DOCX, TXT, CSV, Markdown, plain URLs
"""

import asyncio
import io
import uuid
from pathlib import Path
from typing import List, Optional

import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document as LCDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import Chunk, Document
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStoreClient

logger = structlog.get_logger(__name__)


# ── Loader factory ───────────────────────────────────────────────────────────
def _get_loader(file_path: str, file_type: str):
    loaders = {
        "pdf": PyMuPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "csv": CSVLoader,
        "md": UnstructuredMarkdownLoader,
    }
    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type}. Supported: {list(loaders)}")
    return loaders[file_type](file_path)


def load_from_url(url: str) -> List[LCDocument]:
    loader = WebBaseLoader(url)
    return loader.load()


# ── Chunker ──────────────────────────────────────────────────────────────────
class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: List[LCDocument]) -> List[LCDocument]:
        chunks = self.splitter.split_documents(documents)
        # Inject chunk index into metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        logger.debug("chunked_documents", input_docs=len(documents), output_chunks=len(chunks))
        return chunks


# ── Ingestion Orchestrator ────────────────────────────────────────────────────
class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreClient,
        db: AsyncSession,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.db = db
        self.chunker = DocumentChunker()

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        tenant_id: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        doc_id = str(uuid.uuid4())
        file_ext = Path(filename).suffix.lstrip(".").lower()
        log = logger.bind(doc_id=doc_id, filename=filename, tenant_id=tenant_id)

        # Persist Document record
        db_doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            filename=filename,
            doc_type=file_ext,
            status="processing",
            metadata_=metadata or {},
        )
        self.db.add(db_doc)
        await self.db.flush()

        try:
            # 1. Write temp file (loaders need fs path)
            tmp_path = f"/tmp/{doc_id}_{filename}"
            Path(tmp_path).write_bytes(file_bytes)

            # 2. Load
            loader = _get_loader(tmp_path, file_ext)
            raw_docs = loader.load()
            log.info("loaded_document", pages=len(raw_docs))

            # 3. Chunk
            chunks = self.chunker.chunk(raw_docs)

            # 4. Embed (batch)
            texts = [c.page_content for c in chunks]
            embeddings = await self.embedder.aembed_batch(texts)

            # 5. Index in vector store
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    **c.metadata,
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "filename": filename,
                    **(metadata or {}),
                }
                for c in chunks
            ]
            await self.vector_store.aadd(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)

            # 6. Persist chunks to DB
            db_chunks = [
                Chunk(
                    id=ids[i],
                    document_id=doc_id,
                    chunk_index=i,
                    text=texts[i],
                    token_count=len(texts[i].split()),
                    metadata_=metadatas[i],
                )
                for i in range(len(chunks))
            ]
            self.db.add_all(db_chunks)

            # Update document status
            db_doc.status = "ready"
            db_doc.chunk_count = len(chunks)
            await self.db.flush()

            log.info("ingestion_complete", chunks=len(chunks))
            return {"doc_id": doc_id, "chunks": len(chunks), "status": "ready"}

        except Exception as e:
            db_doc.status = "error"
            db_doc.error_message = str(e)
            await self.db.flush()
            log.error("ingestion_failed", error=str(e))
            raise

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def ingest_url(self, url: str, tenant_id: str, metadata: Optional[dict] = None) -> dict:
        raw_docs = await asyncio.to_thread(load_from_url, url)
        chunks = self.chunker.chunk(raw_docs)
        texts = [c.page_content for c in chunks]
        embeddings = await self.embedder.aembed_batch(texts)

        doc_id = str(uuid.uuid4())
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**c.metadata, "doc_id": doc_id, "tenant_id": tenant_id, "source_url": url, **(metadata or {})}
            for c in chunks
        ]
        await self.vector_store.aadd(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
        logger.info("url_ingested", url=url, chunks=len(chunks))
        return {"doc_id": doc_id, "chunks": len(chunks), "status": "ready"}
