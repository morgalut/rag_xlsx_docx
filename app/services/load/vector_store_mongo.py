"""
MongoDB Vector Store (Improved & File-Safe)
===========================================

Enhanced persistence layer for RAG chunks:
- Handles XLSX and DOCX files safely under the same doc_id
- Automatically offsets chunk_id to prevent overwrites
- Namespaces per file automatically (doc_id + file stem)
- Supports duplicate recovery and local vector search fallback
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import numpy as np
import heapq
import logging
from pymongo import ASCENDING
from pymongo.collection import Collection
from langchain_core.documents import Document
from app.core.config import COLLECTION, USE_VECTOR_SEARCH
from app.db.mongo_client import MongoDBManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------
class BaseVectorStore(ABC):
    """Abstract interface for any vector storage backend."""

    def __init__(self, collection: Any):
        self.collection = collection

    @abstractmethod
    def insert_chunks(
        self,
        doc_id: str,
        chunks: List[str],
        vectors: List[List[float]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def insert_segments(
        self,
        doc_id: str,
        texts: List[str],
        vectors: List[List[float]],
        metas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def query_knn(
        self,
        query_vec: List[float],
        k: int = 5,
        filter_: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def as_retriever(self, embedding_fn, k: int = 5):
        """LangChain-compatible retriever adapter."""
        def _retrieve(query: str):
            vec = embedding_fn([query])[0]
            hits = self.query_knn(vec, k=k)
            return [Document(page_content=h["text"], metadata=h) for h in hits]
        return _retrieve


# ---------------------------------------------------------------------
# Local Vector Store (in-memory cosine)
# ---------------------------------------------------------------------
class LocalVectorStore(BaseVectorStore):
    """Lightweight local cosine similarity vector store."""

    def __init__(self, collection: Collection):
        super().__init__(collection)

    def insert_chunks(self, doc_id, chunks, vectors, meta=None):
        metas = [meta or {} for _ in chunks]
        return self.insert_segments(doc_id, chunks, vectors, metas)

    def insert_segments(self, doc_id, texts, vectors, metas=None):
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have same length.")
        metas = metas or [{} for _ in texts]

        docs = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            docs.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": text,
                "vector": vec,
                "meta": metas[i] or {},
            })

        result = self.collection.insert_many(docs, ordered=False)
        logger.info("✅ Inserted %d local chunks for %s", len(result.inserted_ids), doc_id)
        return {"inserted": len(result.inserted_ids)}

    def query_knn(self, query_vec, k=5, filter_=None):
        """Manual cosine similarity fallback search."""
        items = self.collection.find(
            filter_ or {},
            {"text": 1, "vector": 1, "doc_id": 1, "chunk_id": 1, "meta": 1},
        )
        q = np.array(query_vec, dtype=float)
        heap = []
        for i, it in enumerate(items):
            v = np.array(it.get("vector", []), dtype=float)
            if v.size != len(q):
                continue
            score = float(np.dot(q, v)) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)
            heapq.heappush(heap, (score, i, it))
            if len(heap) > max(k, 50):
                heapq.heappop(heap)
        results = sorted(heap, key=lambda x: x[0], reverse=True)[:k]
        return [
            {
                "text": it["text"],
                "doc_id": it["doc_id"],
                "chunk_id": it["chunk_id"],
                "score": score,
                "meta": it.get("meta", {}),
            }
            for score, _, it in results
        ]


# ---------------------------------------------------------------------
# Mongo Vector Store (primary backend)
# ---------------------------------------------------------------------
class MongoVectorStore(BaseVectorStore):
    """Primary MongoDB persistence backend for RAG chunks."""

    def __init__(self, coll_name: str = COLLECTION):
        mongo_manager = MongoDBManager()
        collection = mongo_manager.get_collection(coll_name)
        super().__init__(collection)
        self._ensure_schema()
        logger.info("✅ MongoVectorStore initialized (collection: %s)", coll_name)

    # --------------------------------------------------------------
    # Schema / Indexing
    # --------------------------------------------------------------
    def _ensure_schema(self) -> None:
        try:
            self.collection.create_index(
                [("doc_id", ASCENDING), ("chunk_id", ASCENDING)], unique=True
            )
            self.collection.create_index("vector")
            logger.debug("✅ MongoDB indexes ensured.")
        except Exception as e:
            logger.warning("⚠️ Could not ensure MongoDB indexes: %s", e)

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------
    def delete_all_for_doc(self, doc_id: str):
        res = self.collection.delete_many({"doc_id": doc_id})
        logger.info("🧹 Removed %d chunks for doc_id=%s", res.deleted_count, doc_id)
        return res.deleted_count

    # --------------------------------------------------------------
    # Insert Logic
    # --------------------------------------------------------------
    def insert_chunks(self, doc_id, chunks, vectors, meta=None):
        metas = [meta or {} for _ in chunks]
        return self.insert_segments(doc_id, chunks, vectors, metas)

    def insert_segments(self, doc_id, texts, vectors, metas=None):
        """
        Safe insertion logic supporting multiple file types (XLSX, DOCX).
        Each file is namespaced internally as doc_id + filename stem.
        """
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have same length.")
        metas = metas or [{} for _ in texts]

        # derive per-file safe namespace
        file_name = metas[0].get("filename") or metas[0].get("file_name") or None
        if file_name:
            safe_doc_id = f"{doc_id}_{Path(file_name).stem}"
        else:
            safe_doc_id = doc_id

        last_doc = self.collection.find_one(
            {"doc_id": safe_doc_id}, sort=[("chunk_id", -1)]
        )
        start_index = (last_doc["chunk_id"] + 1) if last_doc else 0

        docs = []
        for i, (text, vec) in enumerate(zip(texts, vectors), start=start_index):
            docs.append({
                "doc_id": safe_doc_id,
                "chunk_id": i,
                "text": text,
                "vector": vec,
                "meta": metas[i - start_index] or {},
            })

        if not docs:
            logger.warning("⚠️ No chunks to insert for %s", safe_doc_id)
            return {"inserted": 0}

        try:
            result = self.collection.insert_many(docs, ordered=False)
            logger.info("✅ Inserted %d chunks for %s", len(result.inserted_ids), safe_doc_id)
            return {"inserted": len(result.inserted_ids)}

        except Exception as e:
            if "E11000" in str(e):
                logger.warning("⚠️ Duplicate key for %s — recalculating chunk IDs", safe_doc_id)
                max_doc = self.collection.find_one(
                    {"doc_id": safe_doc_id}, sort=[("chunk_id", -1)]
                ) or {"chunk_id": -1}
                offset = max_doc["chunk_id"] + 1
                for idx, d in enumerate(docs):
                    d["chunk_id"] = offset + idx
                result = self.collection.insert_many(docs, ordered=False)
                logger.info("✅ Duplicate recovery succeeded for %s", safe_doc_id)
                return {"inserted": len(docs), "recovered": True}

            logger.error("❌ Insert failed for %s: %s", safe_doc_id, e)
            raise

    # --------------------------------------------------------------
    # Vector Search
    # --------------------------------------------------------------
    def query_knn(self, query_vec, k=5, filter_=None):
        mode = (USE_VECTOR_SEARCH or "auto").lower()
        use_vsearch = mode in ("true", "auto")

        if use_vsearch:
            try:
                pipeline = []
                if filter_:
                    pipeline.append({"$match": filter_})
                pipeline.extend([
                    {
                        "$vectorSearch": {
                            "index": "vec_idx",
                            "path": "vector",
                            "queryVector": query_vec,
                            "numCandidates": max(100, k * 25),
                            "limit": k,
                            "similarity": "cosine",
                        }
                    },
                    {"$project": {
                        "text": 1,
                        "doc_id": 1,
                        "chunk_id": 1,
                        "meta": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }},
                ])
                results = list(self.collection.aggregate(pipeline))
                if results:
                    return results
            except Exception as e:
                logger.warning("⚠️ Vector search unavailable, fallback to cosine: %s", e)

        return LocalVectorStore(self.collection).query_knn(query_vec, k, filter_)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
class VectorStoreFactory:
    """Factory to create vector store backend dynamically."""

    _stores: Dict[str, Type[BaseVectorStore]] = {
        "mongo": MongoVectorStore,
        "local": LocalVectorStore,
    }

    @classmethod
    def register(cls, name: str, store_cls: Type[BaseVectorStore]) -> None:
        cls._stores[name.lower()] = store_cls
        logger.debug("Registered custom vector store: %s", name)

    @classmethod
    def create(cls, name: str = "mongo", **kwargs) -> BaseVectorStore:
        name = name.lower()
        if name not in cls._stores:
            raise ValueError(f"Unknown vector store type: {name}")
        store = cls._stores[name](**kwargs)
        logger.info("Vector store created: %s", name)
        return store
