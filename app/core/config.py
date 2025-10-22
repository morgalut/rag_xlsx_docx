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
# Embeddings (kept as-is; not Ollama)
# -------------------------------
_legacy_model = os.getenv("EMBED_MODEL", "").strip()
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "").strip().lower() or (
    "openai" if "text-embedding" in _legacy_model.lower() else "sentence"
)
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    _legacy_model if _legacy_model else "sentence-transformers/all-MiniLM-L6-v2",
).strip()
EMBED_DIM = _get_int("EMBED_DIM", 384)

# -------------------------------
# Retrieval / Search
# -------------------------------
TOP_K = _get_int("TOP_K", 5)
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "auto").strip().lower()

# -------------------------------
# Generation (OpenAI only)
# -------------------------------
GENERATION_MODE = os.getenv("GENERATION_MODE", "openai").strip().lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Hard fail early if OpenAI is required but key is missing
if GENERATION_MODE == "openai" and not OPENAI_API_KEY:
    # Prefer raising at import time to avoid runtime 500s later.
    raise RuntimeError("OPENAI_API_KEY is required for GENERATION_MODE=openai")
