"""
MongoDB Manager (OOP Singleton)
===============================

Encapsulates all MongoDB connection logic in an object-oriented, reusable class.
Ensures there is only one shared client throughout the app lifecycle.
"""

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from app.core.config import MONGO_URI, MONGO_DB, COLLECTION
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class MongoDBManager:
    """Singleton-like class for managing MongoDB client and database access."""

    _instance: Optional["MongoDBManager"] = None
    _client: Optional[MongoClient] = None

    def __new__(cls, *args, **kwargs):
        """Ensure a single shared instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        if not hasattr(self, "_initialized"):
            self._uri = uri
            self._db_name = db_name
            self._db = None
            self._initialized = True
            logger.debug("MongoDBManager initialized with URI=%s, DB=%s", uri, db_name)

    # ------------------------------------------------------------------
    # Core connection methods
    # ------------------------------------------------------------------
    def connect(self) -> MongoClient:
        """Connect to MongoDB if not already connected."""
        if MongoDBManager._client is None:
            try:
                MongoDBManager._client = MongoClient(self._uri, serverSelectionTimeoutMS=10000)
                MongoDBManager._client.admin.command("ping")
                logger.info("✅ Connected to MongoDB: %s", self._uri)
            except ServerSelectionTimeoutError as e:
                logger.error("❌ Could not connect to MongoDB at %s: %s", self._uri, e)
                raise
        return MongoDBManager._client

    def get_database(self):
        """Return a database reference (lazy-loaded)."""
        if self._db is None:
            client = self.connect()
            self._db = client[self._db_name]
            logger.debug("Using MongoDB database: %s", self._db_name)
        return self._db

    def get_collection(self, name: str = COLLECTION):
        """Return a MongoDB collection reference."""
        db = self.get_database()
        collection = db[name]
        logger.debug("Using MongoDB collection: %s", name)
        return collection

    # ------------------------------------------------------------------
    # Health / utility methods
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Check MongoDB connectivity."""
        try:
            client = self.connect()
            result = client.admin.command("ping")
            return bool(result.get("ok"))
        except Exception as e:
            logger.warning("MongoDB ping failed: %s", e)
            return False

    def close(self) -> None:
        """Close MongoDB connection."""
        if MongoDBManager._client:
            MongoDBManager._client.close()
            MongoDBManager._client = None
            logger.info("🔒 MongoDB connection closed.")

    # ------------------------------------------------------------------
    # Convenience factory method
    # ------------------------------------------------------------------
    @classmethod
    def get_collection_static(cls, name: str = COLLECTION):
        """Shortcut for modules that need quick collection access."""
        instance = cls()
        return instance.get_collection(name)


# ----------------------------------------------------------------------
# Export a shared singleton instance
# ----------------------------------------------------------------------
mongo_manager = MongoDBManager()
get_collection = mongo_manager.get_collection  # backward compatibility
