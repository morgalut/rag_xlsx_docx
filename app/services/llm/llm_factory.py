"""
LLM Factory (OpenAI-only)
=========================
Removes hybrid and Ollama paths. Provides a single OpenAI client.
"""

from typing import Type, Dict, Any
from app.services.llm.base_client import BaseLLMClient
from app.services.llm.openai_client import OpenAIClient
from app.core.config import (
    GENERATION_MODE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

class LLMFactory:
    """Factory for creating LLM clients (OpenAI only)."""

    _registry: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAIClient,
    }

    @classmethod
    def register(cls, name: str, llm_cls: Type[BaseLLMClient]) -> None:
        cls._registry[name.lower()] = llm_cls

    @classmethod
    def create(cls, mode: str | None = None, **kwargs: Any) -> BaseLLMClient:
        """Instantiate an OpenAI client regardless of mode (defensive)."""
        name = (mode or GENERATION_MODE or "openai").lower()
        if name != "openai":
            # Force OpenAI to avoid accidental reintroduction of hybrid/ollama
            name = "openai"

        # Required args for OpenAIClient
        return OpenAIClient(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
