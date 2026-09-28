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
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_DB_NAME = "sensor_optimizer"


def _get_client() -> AsyncIOMotorClient:
    """Return the shared Motor client, recreating it if the loop has changed or closed."""
    global _client
    url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _client is not None:
        client_loop = getattr(_client, "_attached_loop", None)
        if client_loop is None or client_loop.is_closed() or (current_loop is not None and client_loop is not current_loop):
            try:
                _client.close()
            except Exception:
                pass
            _client = None

    if _client is None:
        _client = AsyncIOMotorClient(url, serverSelectionTimeoutMS=5000)
        _client._attached_loop = current_loop
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
