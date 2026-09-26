"""
controllers/surrogate_controller.py
-----------------------------------
Controller layer for surrogate model endpoints.
"""

from fastapi import HTTPException, BackgroundTasks
from app.services import surrogate_service
from app.jobs.surrogate_training_job import execute_surrogate_training


async def get_status() -> dict:
    """GET /api/v1/surrogate/status"""
    try:
        return await surrogate_service.get_surrogate_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch surrogate status: {str(e)}")


async def trigger_training(background_tasks: BackgroundTasks, epochs: int = 50) -> dict:
    """POST /api/v1/surrogate/train"""
    status = await surrogate_service.get_surrogate_status()
    if status["runs_available"] < surrogate_service.TRAINING_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient runs: {status['runs_available']} available, {surrogate_service.TRAINING_THRESHOLD} required.",
        )

    background_tasks.add_task(execute_surrogate_training, epochs)
    return {
        "message": "Surrogate model training started in background",
        "runs_used": status["runs_available"],
    }
