"""
Vector Store Client — unified interface for FAISS, Chroma, and Pinecone.
Supports async add and search with tenant-level metadata filtering.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStoreClient:
    def __init__(self):
        self._store = self._build_store()

    def _build_store(self):
        if settings.VECTOR_STORE == "faiss":
            return _FAISSStore()
        elif settings.VECTOR_STORE == "chroma":
            return _ChromaStore()
        elif settings.VECTOR_STORE == "pinecone":
            return _PineconeStore()
        raise ValueError(f"Unknown vector store: {settings.VECTOR_STORE}")

    async def aadd(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict],
    ) -> None:
        await asyncio.to_thread(self._store.add, ids, embeddings, texts, metadatas)
        logger.debug("vectors_added", count=len(ids))

    async def asearch(
        self,
        query_embedding: List[float],
        top_k: int = settings.RETRIEVAL_TOP_K,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        results = await asyncio.to_thread(self._store.search, query_embedding, top_k, filter)
        return results

    async def adelete(self, ids: List[str]) -> None:
        await asyncio.to_thread(self._store.delete, ids)


# ── FAISS Backend ─────────────────────────────────────────────────────────────
class _FAISSStore:
    def __init__(self):
        import faiss, numpy as np, pickle, os
        self._dim = settings.EMBEDDING_DIMENSION
        self._index = faiss.IndexFlatIP(self._dim)  # inner product (cosine w/ normalized vecs)
        self._id_map: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._path = settings.FAISS_INDEX_PATH
        self._load()

    def _load(self):
        import faiss, pickle, os
        idx_file = f"{self._path}.index"
        meta_file = f"{self._path}.meta"
        if os.path.exists(idx_file):
            self._index = faiss.read_index(idx_file)
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                self._id_map = data["ids"]
                self._texts = data["texts"]
                self._metadatas = data["metadatas"]
            logger.info("faiss_index_loaded", vectors=self._index.ntotal)

    def _save(self):
        import faiss, pickle, os
        os.makedirs(os.path.dirname(self._path), exist_ok=True) if "/" in self._path else None
        faiss.write_index(self._index, f"{self._path}.index")
        with open(f"{self._path}.meta", "wb") as f:
            pickle.dump({"ids": self._id_map, "texts": self._texts, "metadatas": self._metadatas}, f)

    def add(self, ids, embeddings, texts, metadatas):
        import numpy as np
        vecs = np.array(embeddings, dtype="float32")
        self._index.add(vecs)
        self._id_map.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._save()

    def search(self, query_embedding, top_k, filter=None):
        import numpy as np
        if self._index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype="float32")
        scores, indices = self._index.search(q, min(top_k * 3, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadatas[idx]
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            results.append(SearchResult(
                id=self._id_map[idx],
                text=self._texts[idx],
                score=float(score),
                metadata=meta,
            ))
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids):
        # FAISS flat index doesn't support in-place deletion;
        # rebuild without deleted ids
        keep = [i for i, id_ in enumerate(self._id_map) if id_ not in ids]
        import numpy as np, faiss
        self._id_map = [self._id_map[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        new_index = faiss.IndexFlatIP(self._dim)
        if keep:
            vecs = np.array([self._index.reconstruct(i) for i in keep], dtype="float32")
            new_index.add(vecs)
        self._index = new_index
        self._save()


# ── Chroma Backend ────────────────────────────────────────────────────────────
class _ChromaStore:
    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._col = self._client.get_or_create_collection("rag", metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, texts, metadatas):
        self._col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def search(self, query_embedding, top_k, filter=None):
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        if filter:
            kwargs["where"] = filter
        res = self._col.query(**kwargs)
        return [
            SearchResult(id=res["ids"][0][i], text=res["documents"][0][i],
                         score=1 - res["distances"][0][i], metadata=res["metadatas"][0][i])
            for i in range(len(res["ids"][0]))
        ]

    def delete(self, ids):
        self._col.delete(ids=ids)


# ── Pinecone Backend ──────────────────────────────────────────────────────────
class _PineconeStore:
    def __init__(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
        self._index = pc.Index(settings.PINECONE_INDEX_NAME)

    def add(self, ids, embeddings, texts, metadatas):
        vectors = [(ids[i], embeddings[i], {**metadatas[i], "_text": texts[i]}) for i in range(len(ids))]
        self._index.upsert(vectors=vectors)

    def search(self, query_embedding, top_k, filter=None):
        res = self._index.query(vector=query_embedding, top_k=top_k, filter=filter, include_metadata=True)
        return [
            SearchResult(id=m.id, text=m.metadata.pop("_text", ""), score=m.score, metadata=m.metadata)
            for m in res.matches
        ]

    def delete(self, ids):
        self._index.delete(ids=ids)
