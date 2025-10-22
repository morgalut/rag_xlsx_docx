"""
Base LLM Client
===============
Defines the abstract interface and shared behavior for any LLM client.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base class for language model clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from the given prompt."""
        raise NotImplementedError

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Allow direct function-style calls."""
        return self.generate(prompt, **kwargs)
