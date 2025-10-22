from fastapi import APIRouter
from app.api.endpoints import  ingest, query

# Main API router (no FastAPI instance here)
api_router = APIRouter()

# Include endpoint routers with clear prefixes
api_router.include_router(ingest.router, prefix="/api/ingest", tags=["Ingest"])
api_router.include_router(query.router, prefix="/api/query", tags=["Query"])

# Optional root route for quick check
@api_router.get("/")
def root():
    return {"message": "RAG Backend is running 🚀"}
