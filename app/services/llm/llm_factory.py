"""
LLM Factory
===========
Dynamic creator for any LLM client backend.
Uses the Factory pattern for easy registration and instantiation.
"""

from typing import Type, Dict, Any
from app.services.llm.base_client import BaseLLMClient
from app.services.llm.openai_client import OpenAIClient
from app.services.llm.hybrid_client import HybridLLMClient
from app.core.config import (
    GENERATION_MODE,
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)


class LLMFactory:
    """Factory for dynamically creating LLM clients."""

    _registry: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAIClient,
        "hybrid": HybridLLMClient,
    }

    @classmethod
    def register(cls, name: str, llm_cls: Type[BaseLLMClient]) -> None:
        """Register a new LLM client type."""
        cls._registry[name.lower()] = llm_cls

    @classmethod
    def create(cls, mode: str | None = None, **kwargs: Any) -> BaseLLMClient:
        """Instantiate a registered LLM client."""
        name = (mode or GENERATION_MODE).lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown LLM client: {name}")

        if name == "hybrid":
            return HybridLLMClient(
                ollama_host=OLLAMA_HOST,
                ollama_model=OLLAMA_MODEL,
                openai_key=OPENAI_API_KEY,
                openai_model=OPENAI_MODEL,
            )

        elif name == "openai":
            return OpenAIClient(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

        return cls._registry[name](**kwargs)
