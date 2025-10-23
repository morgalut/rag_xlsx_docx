

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from app.core.config import GENERATION_MODE, TOP_K
from app.services.rag.retriever import Retriever
from app.services.rag.generator import Generator
from app.services.rag.question_analyzer import analyze_question



# Abstract base classes

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



# Concrete Pipeline

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str) -> Dict[str, Any]:
        analysis = analyze_question(question)
        diagnostics, context, hits = self.retriever.retrieve(question, analysis)
        answer = self.generator.generate(question, context, diagnostics["status"], hits)
        return {
            "question": question,
            "answer": answer,
            "context_snippet": context[:3],
            "chunks": hits,
            "diagnostics": diagnostics,
            "analysis": analysis,
        }


def create_default_pipeline() -> RAGPipeline:
    retriever = Retriever(top_k=TOP_K)
    generator = Generator(mode=GENERATION_MODE)
    return RAGPipeline(retriever, generator)