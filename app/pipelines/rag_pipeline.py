"""
RAG Pipeline (OOP)
==================

Defines a clean, modular Retrieval-Augmented Generation pipeline
that orchestrates:
  - Retriever (fetch relevant context)
  - Generator (produce final answer)

All components are dependency-injected to allow testing, replacement,
and extension (LangChainRetriever, HybridLLMGenerator, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from app.core.config import GENERATION_MODE, TOP_K
from app.services.rag.retriever import Retriever
from app.services.rag.generator import Generator


# ---------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Retrieve context and metadata hits."""
        raise NotImplementedError


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer based on question and context."""
        raise NotImplementedError


# ---------------------------------------------------------------------
# Concrete Pipeline
# ---------------------------------------------------------------------
class RAGPipeline:
    """Main pipeline that coordinates retrieval and generation."""

    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str) -> Dict[str, Any]:
        """Run the complete RAG flow."""
        context, hits = self.retriever.retrieve(question)
        answer = self.generator.generate(question, context)
        return {
            "question": question,
            "answer": answer,
            "context_snippet": context[:3],
            "chunks": hits,
        }


# ---------------------------------------------------------------------
# Factory for backward compatibility
# ---------------------------------------------------------------------
def create_default_pipeline() -> RAGPipeline:
    """
    Create the default RAG pipeline using standard Retriever + Generator.
    """
    retriever = Retriever(top_k=TOP_K)
    generator = Generator(mode=GENERATION_MODE)
    return RAGPipeline(retriever, generator)
