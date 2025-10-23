

from fastapi import APIRouter
from app.db.mongo_client import MongoDBManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_mongo_connection() -> bool:
    """
    Perform a connection check to ensure MongoDB is available at startup.
    Returns True if the connection succeeds, False otherwise.
    """
    manager = MongoDBManager()
    try:
        client = manager.connect()
        client.admin.command("ping")
        logger.info(" MongoDB connection verified successfully on startup.")
        return True
    except Exception as e:
        logger.error(" MongoDB connection failed at startup: %s", e)
        return False


@router.get("/api/mongo/check", tags=["Database"])
def mongo_health_check():
    """API route to verify MongoDB connectivity."""
    try:
        manager = MongoDBManager()
        client = manager.connect()
        result = client.admin.command("ping")
        return {"status": "ok", "ping": result}
    except Exception as e:
        logger.error(" MongoDB health check failed: %s", e)
        return {"status": "error", "message": str(e)}
