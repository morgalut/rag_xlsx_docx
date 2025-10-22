"""
Retriever Service (OOP)
=======================

Encapsulates logic for embedding a query and retrieving
most relevant text chunks from the Mongo vector store.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from app.services.embeddings.embeddings import SentenceTransformerEmbedder
from app.core.config import TOP_K
from app.services.load.vector_store_mongo import MongoVectorStore


class Retriever:
    """Retrieves semantically relevant text chunks."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.embedder = SentenceTransformerEmbedder()
        self.store = MongoVectorStore()

    def embed_query(self, query: str) -> List[float]:
        """Convert query to vector embedding."""
        return self.embedder.embed([query])[0]

    def retrieve(self, question: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Perform semantic retrieval from the vector store."""
        query_vec = self.embed_query(question)
        hits = self.store.query_knn(query_vec, k=self.top_k)
        context = [h["text"] for h in hits]
        return context, hits
