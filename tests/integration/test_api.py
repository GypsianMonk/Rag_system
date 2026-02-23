"""
Integration tests — hit real FastAPI endpoints with test database.
Requires: PostgreSQL, Redis, FAISS (configured via .env.test or env vars)
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness(self, client):
        resp = await client.get("/health/ready")
        assert resp.status_code == 200


class TestIngestEndpoints:
    @pytest.mark.asyncio
    async def test_ingest_txt_file(self, client):
        content = b"This is a test document about machine learning and AI systems."
        resp = await client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.txt", content, "text/plain")},
            data={"tenant_id": "test-tenant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["chunks"] >= 1
        assert "doc_id" in data

    @pytest.mark.asyncio
    async def test_ingest_unsupported_type_rejected(self, client):
        resp = await client.post(
            "/api/v1/ingest/file",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
            data={"tenant_id": "test-tenant"},
        )
        assert resp.status_code == 415


class TestQueryEndpoints:
    @pytest.mark.asyncio
    async def test_query_returns_answer(self, client):
        resp = await client.post(
            "/api/v1/query",
            json={
                "query": "What is machine learning?",
                "top_k": 3,
                "tenant_id": "test-tenant",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "faithfulness_score" in data
        assert data["faithfulness_score"] >= 0.0

    @pytest.mark.asyncio
    async def test_query_empty_string_rejected(self, client):
        resp = await client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, client):
        resp = await client.post(
            "/api/v1/query",
            json={
                "query": "Can you elaborate?",
                "top_k": 3,
                "conversation_history": [
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
                ],
            },
        )
        assert resp.status_code == 200
