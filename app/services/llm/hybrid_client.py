"""
Hybrid LLM Client
=================
Composition-based hybrid client:
tries local Ollama first, then falls back to OpenAI.
"""

from typing import Any
from app.services.llm.base_client import BaseLLMClient
from app.services.llm.openai_client import OpenAIClient


class HybridLLMClient(BaseLLMClient):
    """Fallback strategy: Ollama → OpenAI."""

    def __init__(
        self,
        ollama_model: str,
        openai_key: str,
        openai_model: str,
        timeout: int = 60,
    ):
        super().__init__(f"{ollama_model}|{openai_model}")
        self.remote = OpenAIClient(openai_key, openai_model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Try Ollama first; fall back to OpenAI if failure."""
        response = self.local.generate(prompt, **kwargs)
        if not response:
            response = self.remote.generate(prompt, **kwargs)
        return response or "[HybridLLMClient] No response from either backend."
