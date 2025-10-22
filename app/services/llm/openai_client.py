"""
OpenAI Client
=============
Implements the BaseLLMClient interface for OpenAI's ChatCompletion API.
"""

from typing import Any
import openai
from app.services.llm.base_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """LLM client for interacting with the OpenAI API."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        openai.api_key = api_key

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a chat response via the OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", 0.4),
                max_tokens=kwargs.get("max_tokens", 500),
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[OpenAIClient] Generation failed: {e}")
            return ""
