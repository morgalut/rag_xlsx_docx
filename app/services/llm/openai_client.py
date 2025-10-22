"""
OpenAI Client
=============
Thin wrapper around OpenAI Chat Completions.
Works with either the new `openai` SDK or the legacy one.
"""

from typing import Any, List, Dict, Optional
from app.services.llm.base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.2, timeout: int = 60):
        super().__init__(model_name)
        self._api_key = api_key
        self._temperature = float(temperature)
        self._timeout = int(timeout)

        try:
            from openai import OpenAI  # new SDK
            self._client = OpenAI(api_key=self._api_key)
            self._new = True
        except Exception:
            import openai  # legacy sdk
            openai.api_key = self._api_key
            self._client = openai
            self._new = False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._new:
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            return (r.choices[0].message.content or "").strip()
        else:
            r = self._client.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                request_timeout=self._timeout,
            )
            return (r["choices"][0]["message"]["content"] or "").strip()
