"""
routers/ppo.py
--------------
Router endpoints for PPO agent status, training, and live progress stream.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query
from app.controllers import ppo_controller

router = APIRouter(prefix="/ppo", tags=["PPO Agent"])


@router.get("/status")
async def get_ppo_status():
    """Get PPO agent status."""
    return await ppo_controller.get_status()


@router.post("/train")
async def train_ppo_agent(
    background_tasks: BackgroundTasks,
    episodes: int = Query(50, ge=1, le=2000, description="Episodes to train"),
):
    """Trigger PPO background training."""
    return await ppo_controller.trigger_training(background_tasks, episodes)


@router.get("/training-progress")
async def get_ppo_training_progress():
    """Stream live PPO training progress over SSE."""
    return await ppo_controller.stream_training_progress()
