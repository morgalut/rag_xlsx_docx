
from fastapi import APIRouter, HTTPException
from pathlib import Path
from app.models.schemas import IngestRequest
from app.services.ingest_service import IngestService

router = APIRouter()


@router.post("/")
def ingest(req: IngestRequest):
    """
    Ingest one or multiple documents (Excel + Word) into the vector store.
    Metadata is automatically generated if not provided.
    """
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        #  Automatic metadata enrichment
        auto_meta = {
            "source_path": str(path),
            "file_type": "folder" if path.is_dir() else path.suffix.lower(),
            "ingested_by": "system_auto",
            "ingest_mode": "batch" if path.is_dir() else "single",
        }

        # Merge user-supplied metadata (if any)
        base_meta = {**auto_meta, **(req.metadata or {})}

        # Unified ingestion pipeline
        svc = IngestService()
        result = svc.ingest(
            doc_id=req.doc_id,
            path=req.path,
            base_meta=base_meta
        )

        return {
            "ok": True,
            "doc_id": req.doc_id,
            "mode": "folder" if path.is_dir() else "file",
            "files_processed": result.get("files_processed", []),
            "segments_loaded": result.get("segments", 0),
            "chunks_stored": result.get("inserted", 0),
            "auto_metadata": auto_meta,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
