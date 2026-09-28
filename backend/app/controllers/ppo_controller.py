"""
controllers/ppo_controller.py
------------------------------
Controller for PPO status, training execution, and training progress.
"""

from __future__ import annotations

import asyncio
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from app.core.ppo_trainer import PPOTrainer
from app.jobs.ppo_training_job import (
    run_ppo_training_job,
    get_ppo_training_state,
    add_listener,
    remove_listener,
)
from app.db.repositories.ppo_repository import get_latest_ppo_run


async def get_status() -> dict:
    """Return PPO agent status (trained state, episodes, best coverage)."""
    trainer = PPOTrainer()
    checkpoint_exists = trainer.checkpoint_path.exists()
    latest_db_run = await get_latest_ppo_run()

    active_state = get_ppo_training_state()

    return {
        "trained": checkpoint_exists,
        "episodes_trained": trainer.episodes_trained,
        "best_coverage": trainer.best_coverage,
        "is_training": active_state["is_training"],
        "current_training_episode": active_state["current_episode"],
        "latest_db_run": latest_db_run,
    }


async def trigger_training(background_tasks: BackgroundTasks, episodes: int = 50) -> dict:
    """Trigger background PPO training job."""
    background_tasks.add_task(run_ppo_training_job, episodes)
    return {"message": "PPO training started", "episodes": episodes}


async def stream_training_progress() -> StreamingResponse:
    """SSE stream of PPO training progress."""

    async def event_generator():
        queue = asyncio.Queue()
        add_listener(queue)
        try:
            while True:
                data = await queue.get()
                import json

                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") == "completed":
                    break
        finally:
            remove_listener(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
