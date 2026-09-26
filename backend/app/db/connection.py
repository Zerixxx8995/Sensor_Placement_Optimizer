"""
db/connection.py
-----------------
MongoDB connection management via Motor (async MongoDB driver).

Provides a single shared AsyncIOMotorClient that is initialised on first
access and reused for the lifetime of the process.

Design notes:
- Motor is non-blocking: all I/O is awaited, never blocking the event loop.
- The client is module-level so it is shared across all requests.
- Call ``get_database()`` from anywhere in the app — it returns the Motor
  database handle.
- MONGODB_URL is read from the environment (set via docker-compose or .env).
"""

import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_DB_NAME = "sensor_optimizer"


def _get_client() -> AsyncIOMotorClient:
    """Return the shared Motor client, creating it on first call."""
    global _client
    if _client is None:
        url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(url, serverSelectionTimeoutMS=5000)
        logger.info("MongoDB client created (url=%s)", url)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the Motor database handle for ``sensor_optimizer``."""
    return _get_client()[_DB_NAME]


async def ping() -> bool:
    """Return True if MongoDB is reachable, False otherwise."""
    try:
        await get_database().command("ping")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB ping failed: %s", exc)
        return False
