# rag_backend/app/main.py
from fastapi import FastAPI
from app.api.router import api_router
from app.db.health import verify_mongo_connection
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain RAG API", version="1.0")

# ---------------------------------------------------------------------
#  On startup: verify MongoDB connection
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting RAG Backend application...")
    ok = verify_mongo_connection()
    if ok:
        print(" Database connection established successfully.")
        logger.info(" Database connection established successfully.")
    else:
        print(" MongoDB not reachable. The app will still start, but DB operations may fail.")
        logger.warning(" MongoDB not reachable. The app will still start, but DB operations may fail.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down RAG Backend application...")

# Include all routers
app.include_router(api_router)
