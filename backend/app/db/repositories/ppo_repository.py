"""
db/repositories/ppo_repository.py
----------------------------------
Data-access layer for the ``ppo_training_runs`` MongoDB collection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pymongo import DESCENDING

from app.db.connection import get_database

logger = logging.getLogger(__name__)

_COLLECTION = "ppo_training_runs"


async def save_ppo_run(data: dict) -> str | None:
    """Save PPO training run document to MongoDB."""
    try:
        db = get_database()
        doc = {
            "created_at": datetime.now(timezone.utc),
            "episodes_trained": int(data.get("episodes_trained", 0)),
            "final_reward": float(data.get("final_reward", 0.0)),
            "best_coverage_achieved": float(data.get("best_coverage", 0.0)),
            "reward_history": [float(r) for r in data.get("reward_history", [])],
            "coverage_history": [float(c) for c in data.get("coverage_history", [])],
            "config": data.get("config", {"num_cameras": 20, "grid_size": {"L": 20, "W": 20}}),
        }
        res = await db[_COLLECTION].insert_one(doc)
        logger.info("Saved PPO training run id=%s", res.inserted_id)
        return str(res.inserted_id)
    except Exception as exc:
        logger.warning("Failed to save PPO training run to MongoDB: %s", exc)
        return None


async def get_latest_ppo_run() -> dict | None:
    """Get latest PPO training run document."""
    try:
        db = get_database()
        doc = await db[_COLLECTION].find_one(sort=[("created_at", DESCENDING)])
        if doc:
            doc["_id"] = str(doc["_id"])
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            return doc
        return None
    except Exception as exc:
        logger.warning("Failed to fetch PPO training run from MongoDB: %s", exc)
        return None
