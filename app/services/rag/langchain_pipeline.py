"""
LangChain RAG Pipeline (OpenAI-only)
====================================
- Mongo/Local vector store retained.
- LangChain usage kept, but LLM helper uses only OpenAI.
- All Ollama/hybrid code removed.
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
)
from app.db.mongo_client import MongoDBManager
from app.services.embeddings.embeddings import SentenceTransformerEmbedder  # kept

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

# ---------------- Vector store (unchanged core) ----------------
class BaseVectorStore(ABC):
    def __init__(self, collection: Any):
        self.collection = collection

    @abstractmethod
    def insert_chunks(self, doc_id: str, chunks: List[str], vectors: List[List[float]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def insert_segments(self, doc_id: str, texts: List[str], vectors: List[List[float]], metas: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def query_knn(self, query_vec: List[float], k: int = 5, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ...

    def as_retriever(self, embedding_fn, k: int = 5):
        def _retrieve(query: str):
            vec = embedding_fn([query])[0]
            hits = self.query_knn(vec, k=k)
            return [Document(page_content=h["text"], metadata=h) for h in hits]
        return _retrieve

class LocalVectorStore(BaseVectorStore):
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
            if v.size != len(q) or v.size == 0:
                continue
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1e-9
            score = float(np.dot(q, v)) / denom
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
        docs = [{"doc_id": doc_id, "chunk_id": i + start_idx, "text": t, "vector": v, "meta": metas[i]}
                for i, (t, v) in enumerate(zip(texts, vectors))]
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

# ---------------- LangChain helpers (OpenAI only) ----------------
def _make_openai_llm(model: str, api_key: str, temperature: float = 0.2):
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    except Exception as e:
        logger.warning("⚠️ ChatOpenAI unavailable (%s). Falling back to native OpenAI.", e)
        class _Compat:
            def __init__(self, model, api_key, temperature):
                try:
                    from openai import OpenAI
                    self._client, self._new = OpenAI(api_key=api_key), True
                except Exception:
                    import openai
                    openai.api_key = api_key
                    self._client, self._new = openai, False
                self._model, self._temperature = model, temperature
            def invoke(self, prompt: str):
                if self._new:
                    r = self._client.chat.completions.create(
                        model=self._model, messages=[{"role":"user","content":prompt}],
                        temperature=self._temperature
                    )
                    content = r.choices[0].message.content or ""
                else:
                    r = self._client.ChatCompletion.create(
                        model=self._model, messages=[{"role":"user","content":prompt}],
                        temperature=self._temperature
                    )
                    content = r["choices"][0]["message"]["content"]
                return type("Msg", (), {"content": content})
        return _Compat(model, api_key, temperature)

# LangChain chain availability flags
HAS_RETRIEVAL_QA, HAS_LCEL_CHAIN = False, False
try:
    from langchain.chains import RetrievalQA
    HAS_RETRIEVAL_QA = True
except ImportError:
    logger.warning("RetrievalQA not available")
try:
    from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
    HAS_LCEL_CHAIN = True
except ImportError:
    logger.warning("LCEL chains not available")

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate  # type: ignore

class LangChainRAGPipeline:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.embedder = SentenceTransformerEmbedder()  # not Ollama
        self.store = MongoVectorStore()

        langchain_store = self.store.create_langchain_store(self.embedder)
        self.retriever = langchain_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        self.llm = _make_openai_llm(OPENAI_MODEL, OPENAI_API_KEY)
        self.prompt_template = (
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\nAnswer:"
        )

    def _manual_run(self, question: str) -> Dict[str, Any]:
        try:
            docs = (self.retriever.invoke(question)
                    if hasattr(self.retriever, 'invoke')
                    else self.retriever.get_relevant_documents(question))
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
        try:
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

            if HAS_LCEL_CHAIN:
                try:
                    prompt = ChatPromptTemplate.from_template(self.prompt_template)
                    from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
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

            logger.info("⚠️ Using manual RAG pipeline fallback")
            return self._manual_run(question)

        except Exception as e:
            logger.exception("❌ Pipeline failed: %s", e)
            return {"answer": f"[Pipeline error: {e}]", "sources": []}
