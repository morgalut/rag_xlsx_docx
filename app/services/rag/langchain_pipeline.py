"""
LangChain RAG Pipeline (Unified Mongo + LLM)
===========================================

Combines:
 - Robust MongoDB Vector Store (Atlas + Local cosine fallback)
 - Dual-compatible LangChain RAG pipeline (LCEL / RetrievalQA / manual)
 - Hybrid LLM support (Ollama + OpenAI fallback)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
import numpy as np
import heapq
import logging
from pymongo import ASCENDING
from pymongo.collection import Collection
from langchain_core.documents import Document
from app.core.config import (
    COLLECTION,
    USE_VECTOR_SEARCH,
    TOP_K,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_HOST,
)
from app.db.mongo_client import MongoDBManager
from app.services.embeddings.embeddings import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

# Try to import MongoDBAtlasVectorSearch
try:
    from langchain_mongodb import MongoDBAtlasVectorSearch
    HAS_MONGODB_ATLAS = True
except ImportError:
    try:
        from langchain_community.vectorstores import MongoDBAtlasVectorSearch
        HAS_MONGODB_ATLAS = True
    except ImportError:
        HAS_MONGODB_ATLAS = False


# ---------------------------------------------------------------------
# VECTOR STORE IMPLEMENTATION
# ---------------------------------------------------------------------
class BaseVectorStore(ABC):
    """Abstract interface for any vector storage backend."""

    def __init__(self, collection: Any):
        self.collection = collection

    # --- Legacy interface for backward compatibility ---
    @abstractmethod
    def insert_chunks(
        self,
        doc_id: str,
        chunks: List[str],
        vectors: List[List[float]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    # --- Preferred interface (per-chunk metadata) ---
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

    # LangChain retriever adapter
    def as_retriever(self, embedding_fn, k: int = 5):
        def _retrieve(query: str):
            vec = embedding_fn([query])[0]
            hits = self.query_knn(vec, k=k)
            return [Document(page_content=h["text"], metadata=h) for h in hits]
        return _retrieve


class LocalVectorStore(BaseVectorStore):
    """Pure Python fallback store using cosine similarity (no Atlas required)."""

    def __init__(self, collection: Collection):
        super().__init__(collection)

    def insert_chunks(self, doc_id, chunks, vectors, meta=None):
        metas = [meta or {} for _ in chunks]
        return self.insert_segments(doc_id, chunks, vectors, metas)

    def insert_segments(self, doc_id, texts, vectors, metas=None):
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must match")
        metas = metas or [{} for _ in texts]
        docs = []
        for i, (t, v) in enumerate(zip(texts, vectors)):
            docs.append({"doc_id": doc_id, "chunk_id": i, "text": t, "vector": v, "meta": metas[i]})
        self.collection.insert_many(docs, ordered=False)
        logger.info("✅ Inserted %d local chunks for %s", len(docs), doc_id)
        return {"inserted": len(docs)}

    def query_knn(self, query_vec, k=5, filter_=None):
        q = np.array(query_vec, dtype=float)
        items = self.collection.find(filter_ or {}, {"text": 1, "vector": 1, "doc_id": 1, "chunk_id": 1, "meta": 1})
        heap = []
        for it in items:
            v = np.array(it.get("vector", []), dtype=float)
            if v.size != len(q):
                continue
            score = float(np.dot(q, v)) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)
            # ✅ use tuple (score, id(it)) to prevent dict comparison
            heapq.heappush(heap, (score, id(it), it))
            if len(heap) > max(k, 50):
                heapq.heappop(heap)
        results = sorted(heap, key=lambda x: x[0], reverse=True)[:k]
        return [{"text": it["text"], "doc_id": it["doc_id"], "chunk_id": it["chunk_id"], "score": score, "meta": it.get("meta", {})}
                for score, _, it in results]


class MongoVectorStore(BaseVectorStore):
    def __init__(self, coll_name: str = COLLECTION):
        mongo_manager = MongoDBManager()
        collection = mongo_manager.get_collection(coll_name)
        super().__init__(collection)
        self._ensure_schema()
        logger.info("✅ MongoVectorStore initialized (collection: %s)", coll_name)

    def _ensure_schema(self):
        try:
            self.collection.create_index([("doc_id", ASCENDING), ("chunk_id", ASCENDING)], unique=True)
            self.collection.create_index("vector")
            logger.debug("✅ Mongo indexes ensured")
        except Exception as e:
            logger.warning("⚠️ Index creation failed: %s", e)

    def insert_chunks(self, doc_id, chunks, vectors, meta=None):
        metas = [meta or {} for _ in chunks]
        return self.insert_segments(doc_id, chunks, vectors, metas)

    def insert_segments(self, doc_id, texts, vectors, metas=None):
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must match")
        metas = metas or [{} for _ in texts]
        last = self.collection.find_one({"doc_id": doc_id}, sort=[("chunk_id", -1)])
        start_idx = (last["chunk_id"] + 1) if last else 0
        docs = [{"doc_id": doc_id, "chunk_id": i + start_idx, "text": t, "vector": v, "meta": metas[i]} for i, (t, v) in enumerate(zip(texts, vectors))]
        self.collection.insert_many(docs, ordered=False)
        logger.info("✅ Inserted %d chunks for %s", len(docs), doc_id)
        return {"inserted": len(docs)}

    def query_knn(self, query_vec, k=5, filter_=None):
        mode = (USE_VECTOR_SEARCH or "auto").lower()
        if mode in ("true", "auto"):
            try:
                pipeline = []
                if filter_:
                    pipeline.append({"$match": filter_})
                pipeline += [
                    {"$vectorSearch": {"index": "vec_idx", "path": "vector", "queryVector": query_vec,
                                       "numCandidates": max(100, k * 25), "limit": k, "similarity": "cosine"}},
                    {"$project": {"text": 1, "doc_id": 1, "chunk_id": 1, "meta": 1,
                                  "score": {"$meta": "vectorSearchScore"}}}
                ]
                results = list(self.collection.aggregate(pipeline))
                if results:
                    return results
            except Exception as e:
                logger.warning("⚠️ Atlas vector search unavailable: %s", e)
        return LocalVectorStore(self.collection).query_knn(query_vec, k, filter_)

    def create_langchain_store(self, embedder):
        if HAS_MONGODB_ATLAS:
            try:
                store = MongoDBAtlasVectorSearch(
                    embedding=embedder, collection=self.collection,
                    index_name="vec_idx", text_key="text", embedding_key="vector"
                )
                logger.info("✅ MongoDBAtlasVectorSearch initialized")
                return store
            except Exception as e:
                logger.warning("⚠️ Atlas retriever failed (%s); fallback to LocalVectorStore", e)
        return LocalVectorStore(self.collection)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
class VectorStoreFactory:
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


# ---------------------------------------------------------------------
# RAG PIPELINE IMPLEMENTATION - FIXED FOR LANGCHAIN 1.0+
# ---------------------------------------------------------------------
# LangChain 1.0+ compatibility
HAS_RETRIEVAL_QA, HAS_LCEL_CHAIN = False, False
try:
    from langchain.chains import RetrievalQA
    HAS_RETRIEVAL_QA = True
except ImportError:
    logger.warning("RetrievalQA not available")

# For LangChain 1.0+ LCEL approach
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains import create_stuff_documents_chain
    HAS_LCEL_CHAIN = True
except ImportError:
    logger.warning("LCEL chains not available")

try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate, ChatPromptTemplate
    except ImportError:
        logger.warning("Prompt templates not available")


def _normalize_retriever(obj):
    """Ensure retriever works with .get_relevant_documents()."""
    if hasattr(obj, "get_relevant_documents"):
        return obj
    if hasattr(obj, "invoke"):
        class CompatRetriever:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_relevant_documents(self, query):
                return self.retriever.invoke(query)
        return CompatRetriever(obj)
    raise TypeError("Unsupported retriever type.")


class _OpenAIChatCompat:
    """Unified OpenAI Chat interface for both SDKs."""
    def __init__(self, model, api_key, temperature=0.2):
        try:
            from openai import OpenAI
            self._client, self._is_new = OpenAI(api_key=api_key), True
        except ImportError:
            import openai
            openai.api_key = api_key
            self._client, self._is_new = openai, False
        self._model, self._temperature = model, temperature

    def invoke(self, prompt: str):
        if self._is_new:
            r = self._client.chat.completions.create(model=self._model, messages=[{"role": "user", "content": prompt}], temperature=self._temperature)
            content = r.choices[0].message.content or ""
        else:
            r = self._client.ChatCompletion.create(model=self._model, messages=[{"role": "user", "content": prompt}], temperature=self._temperature)
            content = r["choices"][0]["message"]["content"]
        return type("Msg", (), {"content": content})


def _make_openai_llm(model, api_key, temperature=0.2):
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    except Exception as e:
        logger.warning("⚠️ ChatOpenAI unavailable (%s). Fallback to _OpenAIChatCompat.", e)
        return _OpenAIChatCompat(model, api_key, temperature)


def _make_ollama_llm(model, base_url, temperature=0.2):
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)
    except Exception as e:
        raise RuntimeError(f"Ollama not available: {e}")


def _make_llm(mode="hybrid"):
    mode = (mode or "hybrid").lower()
    if mode == "openai":
        return _make_openai_llm(OPENAI_MODEL, OPENAI_API_KEY)
    if mode == "ollama":
        return _make_ollama_llm(OLLAMA_MODEL, OLLAMA_HOST)
    try:
        return _make_ollama_llm(OLLAMA_MODEL, OLLAMA_HOST)
    except Exception as e:
        logger.warning("⚠️ Ollama unreachable, fallback to OpenAI: %s", e)
        return _make_openai_llm(OPENAI_MODEL, OPENAI_API_KEY)


class LangChainRAGPipeline:
    """LangChain-powered RAG pipeline."""
    def __init__(self, top_k: int = TOP_K, llm_mode: str = "hybrid"):
        self.top_k = top_k
        self.embedder = SentenceTransformerEmbedder()
        self.store = MongoVectorStore()
        
        # Create retriever with proper search kwargs
        langchain_store = self.store.create_langchain_store(self.embedder)
        self.retriever = langchain_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        self.llm = _make_llm(llm_mode)
        self.prompt_template = (
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\nAnswer:"
        )

    def _manual_run(self, question: str) -> Dict[str, Any]:
        """Manual RAG pipeline fallback."""
        try:
            # Use invoke for LangChain 1.0+
            if hasattr(self.retriever, 'invoke'):
                docs = self.retriever.invoke(question)
            else:
                docs = self.retriever.get_relevant_documents(question)
        except Exception as e:
            logger.warning("⚠️ Retriever failed: %s", e)
            docs = []
            
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            result = self.llm.invoke(prompt)
            text = getattr(result, "content", None) or str(result)
        except Exception as e:
            logger.error("❌ LLM invocation failed: %s", e)
            text = f"[Error generating answer: {e}]"
            
        return {
            "answer": text.strip(), 
            "sources": [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        }

    def run(self, question: str) -> Dict[str, Any]:
        """Run the RAG pipeline with automatic fallbacks."""
        try:
            # Try RetrievalQA first
            if HAS_RETRIEVAL_QA:
                try:
                    chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=self.retriever,
                        return_source_documents=True,
                    )
                    result = chain.invoke({"query": question})
                    return {
                        "answer": result.get("result", ""),
                        "sources": [{"content": d.page_content, "metadata": d.metadata} 
                                  for d in result.get("source_documents", [])]
                    }
                except Exception as e:
                    logger.warning("⚠️ RetrievalQA failed, trying LCEL: %s", e)

            # Try LCEL approach
            if HAS_LCEL_CHAIN:
                try:
                    prompt = ChatPromptTemplate.from_template(self.prompt_template)
                    combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
                    retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
                    result = retrieval_chain.invoke({"input": question})
                    
                    answer = result.get("answer", "")
                    sources = result.get("context", [])
                    
                    return {
                        "answer": answer.strip(),
                        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in sources]
                    }
                except Exception as e:
                    logger.warning("⚠️ LCEL chain failed, using manual: %s", e)

            # Fallback to manual
            logger.info("⚠️ Using manual RAG pipeline fallback")
            return self._manual_run(question)
            
        except Exception as e:
            logger.exception("❌ Pipeline failed: %s", e)
            return {
                "answer": f"[Pipeline error: {e}]", 
                "sources": []
            }