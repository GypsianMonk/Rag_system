"""
SQLAlchemy async models for metadata, documents, tenants, and query logs.
"""

import uuid
from collections.abc import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    metadata_ = Column("metadata", JSONB, default=dict)

    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    filename = Column(String(512), nullable=False)
    doc_type = Column(String(50))
    source_url = Column(Text)
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default="pending")
    error_message = Column(Text)
    metadata_ = Column("metadata", JSONB, default=dict)

    tenant = relationship("Tenant", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer)
    metadata_ = Column("metadata", JSONB, default=dict)

    document = relationship("Document", back_populates="chunks")
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)


class QueryLog(Base, TimestampMixin):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)
    retrieved_chunks = Column(JSONB, default=list)
    faithfulness_score = Column(Float)
    relevancy_score = Column(Float)
    latency_ms = Column(Float)
    llm_tokens_used = Column(Integer)
    model_name = Column(String(100))
    user_id = Column(String(255))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
