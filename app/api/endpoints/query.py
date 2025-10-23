

from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import QueryRequest
from app.pipelines.rag_pipeline import create_default_pipeline
from app.services.rag.question_analyzer import analyze_question

logger = logging.getLogger(__name__)
router = APIRouter(tags=["RAG Query"])


@router.post(
    "/",
    summary="Ask a question via RAG pipeline",
    response_description="Structured RAG answer with analysis and context",
)
async def query_rag(req: QueryRequest) -> JSONResponse:
    """
    Handle user RAG questions using the modular RAG pipeline.
    1. Analyze the question (domain, intent, keywords)
    2. Retrieve relevant context from the vector store
    3. Generate a final, contextualized answer using OpenAI

    Returns structured JSON ready for front-end consumption.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty.",
        )

    question = req.query.strip()
    logger.info(" Received RAG query: %s", question[:120])

    try:
        #   Analyze question semantics
        analysis = analyze_question(question)
        logger.debug(" Question analysis: %s", analysis)

        #   Build RAG pipeline (Retriever + Generator)
        pipeline = create_default_pipeline()

        #   Run query through RAG pipeline
        result = pipeline.run(question)

        #   Format structured response
        payload = {
            "ok": True,
            "query": question,
            "analysis": analysis,
            "response": {
                "answer": result.get("answer", "No answer generated."),
                "context_preview": result.get("context_snippet", []),
                "retrieved_chunks": result.get("chunks", []),
            },
        }

        logger.info(" RAG query processed successfully for domain=%s", analysis.get("domain"))
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

    except HTTPException:
        raise  # re-raise known HTTP errors unchanged
    except Exception as e:
        logger.exception(" RAG query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {e}",
        )
