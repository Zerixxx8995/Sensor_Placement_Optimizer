"""
jobs/ppo_training_job.py
-------------------------
Background job runner for PPO agent training.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

from app.core.ppo_trainer import PPOTrainer
from app.db.repositories.ppo_repository import save_ppo_run

logger = logging.getLogger(__name__)

# Global state for training status and progress broadcasting
_training_state = {
    "is_training": False,
    "current_episode": 0,
    "total_episodes": 0,
    "latest_reward": 0.0,
    "latest_coverage": 0.0,
    "best_coverage": 0.0,
    "listeners": [],
}


def get_ppo_training_state() -> dict:
    return {
        "is_training": _training_state["is_training"],
        "current_episode": _training_state["current_episode"],
        "total_episodes": _training_state["total_episodes"],
        "latest_reward": _training_state["latest_reward"],
        "latest_coverage": _training_state["latest_coverage"],
        "best_coverage": _training_state["best_coverage"],
    }


def add_listener(listener_queue: asyncio.Queue):
    _training_state["listeners"].append(listener_queue)


def remove_listener(listener_queue: asyncio.Queue):
    if listener_queue in _training_state["listeners"]:
        _training_state["listeners"].remove(listener_queue)


def _notify_listeners(data: dict):
    for q in list(_training_state["listeners"]):
        try:
            q.put_nowait(data)
        except Exception:
            pass


async def run_ppo_training_job(episodes: int = 50):
    """Background task to train PPO placement agent."""
    if _training_state["is_training"]:
        logger.warning("PPO training is already in progress.")
        return

    _training_state["is_training"] = True
    _training_state["total_episodes"] = episodes
    _training_state["current_episode"] = 0

    trainer = PPOTrainer()

    def on_progress(info: dict):
        _training_state["current_episode"] = info["episode"]
        _training_state["latest_reward"] = info["reward"]
        _training_state["latest_coverage"] = info["coverage"]
        _training_state["best_coverage"] = info["best_coverage"]
        _notify_listeners(info)

    try:
        # Run blocking trainer loop in executor so API remains non-blocking
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None, trainer.train_episodes, episodes, on_progress
        )

        # Save to MongoDB
        await save_ppo_run(res)
    except Exception as exc:
        logger.error("PPO training job failed: %s", exc)
    finally:
        _training_state["is_training"] = False
        _notify_listeners({"status": "completed"})
