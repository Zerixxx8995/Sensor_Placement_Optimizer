"""
controllers/redeploy_controller.py
----------------------------------
Controller layer for DQN redeployment endpoints.
"""

from fastapi import HTTPException, BackgroundTasks
from app.services import redeployment_service
from app.jobs.dqn_training_job import execute_dqn_training


def redeploy(payload: dict) -> dict:
    """
    POST /api/v1/redeploy
    Payload expected:
    {
        "positions": [[x,y], ...],
        "dead_node_indices": [0, 3, 5],
        "area": {"width": 100, "height": 100},
        "sensing_radius": 15
    }
    """
    positions = payload.get("positions", [])
    dead_nodes = payload.get("dead_node_indices", [])
    area = payload.get("area", {})
    area_w = float(area.get("width", 100.0))
    area_h = float(area.get("height", 100.0))
    rs = float(payload.get("sensing_radius", 15.0))

    if not positions:
        raise HTTPException(status_code=400, detail="Missing sensor positions")

    try:
        return redeployment_service.redeploy_nodes(
            positions=positions,
            dead_node_indices=dead_nodes,
            area_w=area_w,
            area_h=area_h,
            sensing_radius=rs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redeployment failed: {str(e)}")


def get_status() -> dict:
    """GET /api/v1/redeploy/status"""
    return redeployment_service.get_dqn_status()


def trigger_train(background_tasks: BackgroundTasks, episodes: int = 500) -> dict:
    """POST /api/v1/redeploy/train"""
    background_tasks.add_task(execute_dqn_training, episodes)
    return {"message": "DQN training started in background", "target_episodes": episodes}
