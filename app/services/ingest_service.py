

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback

from app.models.types import Segment
from app.services.embeddings.embeddings import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    EmbedderFactory,
)
from app.services.load.chunker import BaseChunker, ChunkerFactory
from app.services.load.vector_store_mongo import BaseVectorStore, VectorStoreFactory
from app.services.load.loaders import load_any


class IngestService:
    """Main orchestrator for document ingestion, embedding, and storage."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        chunker: Optional[BaseChunker] = None,
        vstore: Optional[BaseVectorStore] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embed_model: Optional[str] = None,
    ):
        # ------------------------------------------------------------------
        # Embedder initialization (safe, env-aware, and validated)
        # ------------------------------------------------------------------
        if embedder is not None:
            self.embedder = embedder
        elif embed_model:
            self.embedder = SentenceTransformerEmbedder(model_name=embed_model)
        else:
            self.embedder = EmbedderFactory.create()

        print(
            f"🧪 Using embedder={type(self.embedder).__name__}, "
            f"model={getattr(self.embedder, 'model_name', None)}"
        )

        # ------------------------------------------------------------------
        # Chunker and Vector Store
        # ------------------------------------------------------------------
        self.chunker = chunker or ChunkerFactory.create(
            "simple", max_chars=chunk_size, overlap=chunk_overlap
        )
        self.vstore = vstore or VectorStoreFactory.create("mongo")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, path: str) -> Path:
        """Ensure consistent absolute path resolution."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"❌ Path not found: {p}")
        return p

    def _iter_files(self, base_path: Path) -> List[Path]:
        """Collect supported files (.xlsx, .docx), case-insensitive."""
        if base_path.is_file():
            return [base_path]
        return [
            p
            for p in base_path.rglob("*")
            if p.suffix.lower() in (".xlsx", ".docx")
        ]

    def _normalize_segments(
        self, data: List[Any], file_meta: Dict[str, Any]
    ) -> List[Segment]:
        """Normalize raw loader output → List[Segment]."""
        if not data:
            return []
        if isinstance(data[0], Segment):
            return data
        return [
            Segment(text=str(t).strip(), meta=dict(file_meta))
            for t in data
            if str(t).strip()
        ]

    # ------------------------------------------------------------------
    # Main ingestion logic
    # ------------------------------------------------------------------
    def ingest(
        self, doc_id: str, path: str, base_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_meta = base_meta or {}
        base_path = self._resolve_path(path)

        files = self._iter_files(base_path)
        if not files:
            raise FileNotFoundError(f"❌ No supported files found in: {base_path}")

        total_segments = 0
        total_chunks = 0
        total_inserted = 0
        processed_files: List[str] = []

        print(f"🚀 Starting ingestion for doc_id='{doc_id}' from path='{base_path}'")

        for file_path in files:
            try:
                print(f"\n📄 Loading file: {file_path.name}")

                # --- Detect file type ---
                ext = file_path.suffix.lower().lstrip(".")
                file_doc_type = "xlsx" if ext == "xlsx" else "docx" if ext == "docx" else "unknown"

                # --- Load file content ---
                raw_data = load_any(str(file_path), base_meta=base_meta)
                if not raw_data:
                    print(f"⚠️ Loader returned empty list for {file_path.name}")
                    continue

                # --- Merge file metadata ---
                file_meta = {
                    **base_meta,
                    "file_name": file_path.name,
                    "file_ext": f".{file_doc_type}",
                    "doc_type": file_doc_type,
                    "source_path": str(file_path),
                }

                # --- Normalize ---
                segments = self._normalize_segments(raw_data, file_meta)
                if not segments:
                    print(f"⚠️ No segments found in {file_path.name}")
                    continue

                print(f"🧩 Loaded {len(segments)} segments from {file_path.name}")
                sample_text = segments[0].text[:120].replace("\n", " ")
                print(f"🔍 Sample segment: {sample_text}...")

                total_segments += len(segments)

                # --- Chunk ---
                chunks = self.chunker.chunk(segments)
                print(f"✂️  Created {len(chunks)} chunks from {file_path.name}")
                total_chunks += len(chunks)

                # --- Embed ---
                texts = [c.text for c in chunks]
                vectors = self.embedder.embed(texts)
                print(f"🧠 Generated embeddings for {len(vectors)} chunks")

                # --- Insert ---
                res = self.vstore.insert_segments(
                    doc_id=doc_id,
                    texts=texts,
                    vectors=vectors,
                    metas=[{
                        **c.meta,
                        "file_name": file_path.name,
                        "file_ext": f".{file_doc_type}",
                        "doc_type": file_doc_type,
                    } for c in chunks],
                )

                inserted = res.get("inserted", 0)
                print(f"✅ Inserted {inserted} chunks from {file_path.name} into MongoDB")

                total_inserted += inserted
                processed_files.append(file_path.name)

            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")

        print("\n📦 Ingestion Summary")
        print(f"  ├─ Segments loaded: {total_segments}")
        print(f"  ├─ Chunks generated: {total_chunks}")
        print(f"  └─ Chunks inserted: {total_inserted}")

        return {
            "ok": True,
            "doc_id": doc_id,
            "files_processed": processed_files,
            "segments_loaded": total_segments,
            "chunks_generated": total_chunks,
            "chunks_inserted": total_inserted,
            "auto_metadata": {
                "source_path": str(base_path),
                "file_type": "folder" if base_path.is_dir() else "file",
                "ingested_by": "system_auto",
                "ingest_mode": "batch",
            },
        }
