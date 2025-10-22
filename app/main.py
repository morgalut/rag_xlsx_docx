# rag_backend/app/main.py
from fastapi import FastAPI
from app.api.router import api_router
from app.db.health import verify_mongo_connection
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain RAG API", version="1.0")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting RAG Backend application...")
    ok = verify_mongo_connection()
    if ok:
        print(" Database connection established successfully.")
    else:
        print("⚠️ MongoDB not reachable. The app will still start, but DB operations may fail.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down RAG Backend application...")

app.include_router(api_router)
