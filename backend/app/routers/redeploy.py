"""
routers/redeploy.py
-------------------
Router for DQN Adaptive Redeployment API endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, Query, Body
from app.controllers import redeploy_controller

router = APIRouter()


@router.post("/redeploy")
def redeploy_sensors(payload: dict = Body(...)):
    """
    Input: current deployment positions + dead_node_indices
    Output: redeployment actions, updated positions, before/after coverage metrics
    """
    return redeploy_controller.redeploy(payload)


@router.get("/redeploy/status")
def get_redeploy_status():
    """
    Get status: { agent_trained: bool, episodes_trained: int }
    """
    return redeploy_controller.get_status()


@router.post("/redeploy/train")
def train_redeploy_agent(
    background_tasks: BackgroundTasks,
    episodes: int = Query(default=500, ge=10, le=5000),
):
    """
    Trigger DQN agent training loop in background.
    """
    return redeploy_controller.trigger_train(background_tasks, episodes=episodes)
