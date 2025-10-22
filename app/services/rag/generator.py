"""
Generator Service (OOP)
=======================

Responsible for turning retrieved context into a final answer
using either Ollama or OpenAI (or hybrid) models.
"""

from __future__ import annotations
from typing import List
from app.core.config import GENERATION_MODE, OLLAMA_MODEL, OPENAI_MODEL
from app.services.llm.llm_factory import LLMFactory

class PromptBuilder:
    """Builds prompts from question + context."""

    @staticmethod
    def build_prompt(question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks[:5])
        return (
            "You are a helpful assistant. Answer the question based on the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )


class Generator:
    """Uses a selected LLM backend to generate answers."""

    def __init__(self, mode: str = GENERATION_MODE):
        self.mode = mode.lower()
        self.llm = LLMFactory.create(self.mode)

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer using the configured LLM backend."""
        prompt = PromptBuilder.build_prompt(question, context_chunks)
        try:
            answer = self.llm.generate(prompt)
            if not answer:
                return "No answer generated."
            return answer.strip()
        except Exception as e:
            return f"[Generator] Generation failed: {e}"
