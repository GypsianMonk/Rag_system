# Enterprise RAG System

**Production-ready Retrieval-Augmented Generation with Hybrid Search, Multi-Tenancy, and Full Observability**

[![CI/CD](https://github.com/your-org/rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rag-system/actions)
[![Coverage](https://codecov.io/gh/your-org/rag-system/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/rag-system)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client / API                            │
│               POST /query     POST /ingest/file                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    FastAPI Application                           │
│         Rate Limiting · Auth (JWT) · Request Tracing            │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├──── Retrieval Layer ────────────────────────────────────┐
       │     │                                                    │
       │     ├─ Embedder (BGE / OpenAI)                          │
       │     ├─ Dense ANN Search (FAISS / Chroma / Pinecone)     │
       │     ├─ BM25 Keyword Scoring (rank-bm25)                 │
       │     ├─ RRF Hybrid Fusion                                 │
       │     └─ Cross-Encoder Reranker (ms-marco)                │
       │                                                          │
       ├──── Generation Layer ──────────────────────────────────┐ │
       │     │                                                   │ │
       │     ├─ Context Window Builder + Citation Injector       │ │
       │     ├─ LLM (OpenAI / Anthropic / Ollama)               │ │
       │     └─ Hallucination Guard (cosine threshold)           │ │
       │                                                          │ │
       └──── Infrastructure ────────────────────────────────────┘ │
             │                                                      │
             ├─ PostgreSQL  (metadata, chunks, audit logs)          │
             ├─ Redis       (embedding cache, rate limiting)        │
             ├─ Prometheus  (metrics scraping)                      │
             └─ Grafana     (dashboards)                            │
```

---

## Features

**Retrieval**
- Semantic dense search with FAISS, Chroma, or Pinecone
- BM25 keyword search fused via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking (ms-marco-MiniLM)
- Configurable `alpha` parameter for dense/sparse trade-off

**Ingestion**
- Supports PDF, DOCX, TXT, CSV, Markdown, and URLs
- Configurable chunking with overlap
- Async batch embedding with Redis caching
- Deduplication via content hashing

**Generation**
- OpenAI, Anthropic, and Ollama LLM support
- Citation-backed responses `[Source N]`
- Hallucination detection via answer-context cosine similarity
- Conversational memory (multi-turn RAG)
- Streaming token responses

**Infrastructure**
- Multi-tenant data isolation
- JWT authentication + RBAC
- Prometheus metrics + Grafana dashboards
- Rate limiting per API key
- Docker + Docker Compose + Kubernetes manifests
- GitHub Actions CI with coverage gating and Trivy security scan

**Evaluation**
- RAGAS-compatible metrics: faithfulness, context precision, context recall, answer relevancy
- Embedding-based offline evaluation (no LLM calls needed)
- Benchmark runner script with JSONL dataset support

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/your-org/rag-system.git
cd rag-system
make env          # copies .env.example → .env

# 2. Add your OpenAI key to .env
#    OPENAI_API_KEY=sk-...

# 3. Start infra
make docker-up    # PostgreSQL + Redis + Prometheus + Grafana

# 4. Install Python deps and start API
make install
make dev
```

API docs → http://localhost:8000/docs  
Grafana → http://localhost:3000 (admin/admin)  
Prometheus → http://localhost:9090

---

## API Reference

### POST /api/v1/query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key risks in the Q3 financial report?",
    "top_k": 5,
    "alpha": 0.6,
    "tenant_id": "acme-corp"
  }'
```

```json
{
  "query": "What are the key risks in the Q3 financial report?",
  "answer": "According to [Source 1], the key risks include...",
  "citations": [
    {
      "index": 1,
      "id": "doc-abc_3",
      "filename": "Q3_Report.pdf",
      "score": 0.921,
      "text_preview": "The following risk factors were identified..."
    }
  ],
  "faithfulness_score": 0.87,
  "latency_ms": 342.5
}
```

### POST /api/v1/ingest/file

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@report.pdf" \
  -F "tenant_id=acme-corp" \
  -F 'metadata={"department":"finance","year":"2024"}'
```

### POST /api/v1/ingest/url

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.example.com/api", "tenant_id": "acme-corp"}'
```

---

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `VECTOR_STORE` | `faiss` | `faiss`, `chroma`, `pinecone` |
| `RETRIEVAL_TOP_K` | `5` | Documents returned per query |
| `HYBRID_ALPHA` | `0.5` | 0=pure BM25, 1=pure dense |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `HALLUCINATION_THRESHOLD` | `0.5` | Min faithfulness score |
| `CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `CACHE_ENABLED` | `true` | Redis embedding cache |

---

## Running Tests

```bash
make test              # full suite with coverage
make test-unit         # unit tests only (fast, no infra)
make test-integration  # integration tests (requires Docker infra)
```

---

## Evaluation

Prepare a JSONL benchmark file:
```jsonl
{"question": "What is RAG?", "ground_truth": "RAG stands for Retrieval-Augmented Generation..."}
{"question": "How does hybrid search work?", "ground_truth": "Hybrid search combines..."}
```

Run evaluation:
```bash
python scripts/run_evaluation.py --dataset data/eval_set.jsonl --output results/eval.json
```

Example results:
```
==================================================
Evaluation Results
==================================================
  n_samples                 50
  faithfulness              0.823
  context_precision         0.791
  context_recall            0.756
  answer_relevancy          0.844
  composite                 0.804
```

---

## Scaling Strategy

**Vertical scaling** — FAISS + single FastAPI process handles ~200 QPS on 8 vCPUs.

**Horizontal scaling** — Run multiple API replicas behind a load balancer. FAISS index must be on shared storage (EFS/NFS) or replaced with Pinecone/Weaviate for true distributed search.

**Ingestion at scale** — Move ingestion to async Celery workers with Redis broker. The `/ingest` endpoint enqueues a task and returns immediately.

**Caching** — Redis caches embeddings (TTL configurable). Identical queries resolved in <5ms.

**Vector store migration** — FAISS (dev/small) → Chroma (medium) → Pinecone/Weaviate (production/millions of vectors).

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | OpenAI / Anthropic / Ollama |
| Embeddings | BGE-base / OpenAI |
| Vector DB | FAISS / Chroma / Pinecone |
| Keyword Search | rank-bm25 |
| Reranker | sentence-transformers CrossEncoder |
| Metadata DB | PostgreSQL + SQLAlchemy (async) |
| Cache | Redis |
| Observability | Prometheus + Grafana + structlog |
| CI/CD | GitHub Actions + Trivy |
| Containers | Docker + Docker Compose |

---

## Project Structure

```
rag-system/
├── app/
│   ├── api/routes/        # FastAPI route handlers
│   ├── core/              # Config, DB models, logging
│   ├── ingestion/         # Document loading, chunking, pipeline
│   ├── retrieval/         # Embedder, vector store, hybrid retriever
│   ├── generation/        # LLM integration, prompt building, citations
│   ├── evaluation/        # RAGAS-compatible evaluation framework
│   └── utils/             # Dependency injection, helpers
├── tests/
│   ├── unit/              # Pure unit tests (no infra)
│   └── integration/       # API + DB integration tests
├── docker/                # Dockerfile, Prometheus, Grafana config
├── scripts/               # Evaluation runner, data scripts
├── .github/workflows/     # CI/CD pipeline
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## License

MIT — see [LICENSE](LICENSE)
