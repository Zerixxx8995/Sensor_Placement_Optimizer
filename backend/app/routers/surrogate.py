"""
routers/surrogate.py
--------------------
Router for PyTorch surrogate model API endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, Query
from app.controllers import surrogate_controller

router = APIRouter()


@router.get("/surrogate/status")
async def get_surrogate_status():
    """
    Get surrogate model status: whether trained, available runs, runs needed, last trained timestamp.
    """
    return await surrogate_controller.get_status()


@router.post("/surrogate/train")
async def trigger_surrogate_train(
    background_tasks: BackgroundTasks,
    epochs: int = Query(default=50, ge=1, le=500),
):
    """
    Manually trigger surrogate model retraining from stored runs.
    """
    return await surrogate_controller.trigger_training(background_tasks, epochs=epochs)
