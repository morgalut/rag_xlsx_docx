from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import os
import torch

from sentence_transformers import SentenceTransformer
from app.core.config import EMBED_BACKEND, EMBED_MODEL_NAME, OPENAI_API_KEY

logger = logging.getLogger(__name__)



# Base Abstract Class

class BaseEmbedder(ABC):
    """Abstract base for all embedding backends, with safe tokenizer support."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "default"
        self.tokenizer = self._make_fallback_tokenizer()

    def _make_fallback_tokenizer(self):
        """Basic tokenizer used when a model-specific tokenizer is unavailable."""
        class FallbackTokenizer:
            def tokenize(self, text: str):
                return text.split()
        return FallbackTokenizer()

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """Safe interface for tokenization (always available)."""
        if not hasattr(self.tokenizer, "tokenize"):
            logger.warning(" Embedder has no tokenizer; using fallback.")
            self.tokenizer = self._make_fallback_tokenizer()
        return [self.tokenizer.tokenize(t) for t in texts]

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement 'embed()'")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)



# SentenceTransformer Implementation

class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper for SentenceTransformers with guaranteed tokenizer and safe defaults."""

    def __init__(self, model_name: Optional[str] = "all-MiniLM-L6-v2", device: Optional[str] = None):
        # Resolve model name robustly from .env or fallback
        resolved = model_name or EMBED_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
        if not isinstance(resolved, str) or not resolved.strip():
            raise ValueError("SentenceTransformerEmbedder: invalid model_name (empty).")
        resolved = resolved.strip()

        # Auto-detect device
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(model_name=resolved)

        try:
            self.model = SentenceTransformer(resolved, device=device)
            tok = getattr(self.model, "tokenizer", None)
            if hasattr(tok, "tokenize"):
                self.tokenizer = tok
                logger.info(f" Using SentenceTransformer tokenizer for {resolved} ({device})")
            else:
                logger.warning(f" No tokenizer in model; using fallback for {resolved}")
        except Exception as e:
            logger.error(f" Failed to load SentenceTransformer '{resolved}': {e}")
            raise

        logger.info(f" Loaded SentenceTransformer model: {resolved} on device={device}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Compute sentence embeddings safely."""
        if not texts:
            logger.warning(" Empty text list passed to embed().")
            return []

        try:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vectors.tolist()
        except Exception as e:
            logger.error(f"Embedding failed for {len(texts)} texts (model={self.model_name}): {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias for embedding document collections."""
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([text])[0]



# OpenAI Embedding Backend

class OpenAIEmbedder(BaseEmbedder):
    """Embedding backend using the OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(model_name=model_name)
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via the OpenAI API."""
        if not texts:
            logger.warning(" Empty input passed to OpenAIEmbedder.embed().")
            return []

        try:
            resp = self.client.embeddings.create(model=self.model_name, input=texts)
            embeddings = [d.embedding for d in resp.data]
            logger.info(f" Generated {len(embeddings)} OpenAI embeddings using {self.model_name}")
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise



# Embedder Factory

class EmbedderFactory:
    """Factory for dynamically selecting embedding backend."""

    _registry = {
        "sentence": SentenceTransformerEmbedder,
        "openai": OpenAIEmbedder,
    }

    @classmethod
    def register(cls, name: str, embedder_cls):
        cls._registry[name.lower()] = embedder_cls
        logger.debug(f"Registered embedder backend: {name}")

    @classmethod
    def create(cls) -> BaseEmbedder:
        """Auto-detect backend from .env or default to SentenceTransformer."""
        backend = os.getenv("EMBED_BACKEND", EMBED_BACKEND).lower()
        model_name = os.getenv("EMBED_MODEL_NAME", EMBED_MODEL_NAME)
        logger.info(f" Creating embedder: backend={backend}, model={model_name}")

        # Auto-detect by backend or model string
        if "sentence" in backend or "minilm" in backend:
            embedder_cls = cls._registry.get("sentence")
        elif "openai" in backend:
            embedder_cls = cls._registry.get("openai")
        else:
            # Fallback inference from model name
            if model_name and "sentence-transformers" in model_name.lower():
                embedder_cls = cls._registry.get("sentence")
            elif model_name and "text-embedding" in model_name.lower():
                embedder_cls = cls._registry.get("openai")
            else:
                raise ValueError(f"Unknown embedding backend: {backend}")

        embedder = embedder_cls(model_name=model_name)
        logger.info(f" EmbedderFactory created: {embedder_cls.__name__} ({model_name})")
        return embedder
