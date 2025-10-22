from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest
from app.pipelines.rag_pipeline import create_default_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Query"])

@router.post("/", summary="Ask a question via RAG pipeline")
async def query_rag(req: QueryRequest):
    """
    Handles user RAG questions using the modular RAG pipeline.
    Combines Retriever + Generator to produce a contextual answer.
    """
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        logger.info(f"🧠 Received query: {req.query[:80]}...")

        # ✅ Create pipeline (Retriever + Generator)
        pipeline = create_default_pipeline()

        # ✅ Run question through the RAG pipeline
        result = pipeline.run(req.query)

        logger.info("✅ Query processed successfully.")
        return {
            "ok": True,
            "query": req.query,
            "response": result.get("answer", "No answer generated"),
            "context_preview": result.get("context_snippet", []),
            "retrieved_chunks": result.get("chunks", []),
        }

    except Exception as e:
        logger.exception("❌ Query failed.")
        raise HTTPException(status_code=500, detail=str(e))
