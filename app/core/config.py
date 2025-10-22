# rag_backend/app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

def _get_bool(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in ("1", "true", "yes", "y", "on")

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# -------------------------------
# MongoDB
# -------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "ragdb")
COLLECTION = os.getenv("MONGO_COLLECTION", "chunks")

# -------------------------------
# Embeddings
# -------------------------------
# Backward-compat: support old EMBED_MODEL while preferring new keys
# - EMBED_BACKEND: "sentence" or "openai"
# - EMBED_MODEL_NAME: e.g. "sentence-transformers/all-MiniLM-L6-v2" or "text-embedding-3-small"
_legacy_model = os.getenv("EMBED_MODEL", "").strip()

# Auto-infer backend if EMBED_BACKEND not set:
# - If model name contains "text-embedding" -> openai
# - Else default to "sentence"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "").strip().lower()
if not EMBED_BACKEND:
    if "text-embedding" in _legacy_model.lower():
        EMBED_BACKEND = "openai"
    else:
        EMBED_BACKEND = "sentence"

# Choose model name with precedence: EMBED_MODEL_NAME > EMBED_MODEL > sensible default
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    _legacy_model if _legacy_model else "sentence-transformers/all-MiniLM-L6-v2",
).strip()

# Dim is only used by some stores/checks; keep configurable
EMBED_DIM = _get_int("EMBED_DIM", 384)

# -------------------------------
# Retrieval / Search
# -------------------------------
TOP_K = _get_int("TOP_K", 5)

# Accepts "true", "false", or "auto" (string). Other code lower()s and interprets it.
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "auto").strip().lower()

# -------------------------------
# Generation & Providers
# -------------------------------
GENERATION_MODE = os.getenv("GENERATION_MODE", "hybrid").strip().lower()  # "ollama" | "openai" | "hybrid"

# Ollama (local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glassy").strip()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
