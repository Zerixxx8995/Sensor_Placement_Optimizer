"""
services/optimization_service.py
----------------------------------
Business logic for submitting and managing optimization jobs.

One job = one PSO run. This layer:
  1. Converts the Pydantic config into the raw dict core/pso.py expects.
  2. Registers a new job in job_store.
  3. Enqueues the background task via FastAPI BackgroundTasks.
  4. Returns the job_id to the caller.

No HTTP knowledge, no algorithm code — orchestration only.
"""

import uuid

from fastapi import BackgroundTasks

from app.models.config import OptimizationConfig
from app.jobs import job_store
from app.jobs.optimization_job import run_optimization_job


def submit_optimization(
    config: OptimizationConfig, background_tasks: BackgroundTasks
) -> str:
    """
    Register a new optimization job and enqueue it for background execution.

    Args:
        config:           Validated OptimizationConfig from the controller.
        background_tasks: FastAPI BackgroundTasks injected by the router.

    Returns:
        job_id: UUID string the client uses to poll status/result.
    """
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)

    pso_config = _to_pso_config(config)
    background_tasks.add_task(run_optimization_job, job_id, pso_config)

    return job_id


def get_job_status(job_id: str) -> dict | None:
    """
    Return current job state dict or None if job_id is unknown.
    Consumers must handle None (404).
    """
    return job_store.get_job(job_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pso_config(config: OptimizationConfig) -> dict:
    """
    Translate OptimizationConfig (Pydantic model) into the raw dict
    that core/pso.run_pso() expects. This is the only place the
    translation lives — never scattered across layers.
    """
    return {
        "area": {
            "width": config.area.width,
            "height": config.area.height,
        },
        "num_nodes": config.num_nodes,
        "sensing_radius": config.sensing_radius,
        "comm_radius": config.comm_radius,
        "initial_energy": config.initial_energy,
        "weights": {
            "w1": config.weights.w1,
            "w2": config.weights.w2,
            "w3": config.weights.w3,
        },
        "pso_params": {
            "swarm_size": config.pso_params.swarm_size,
            "iterations": config.pso_params.iterations,
            "inertia": config.pso_params.inertia,
            "c1": config.pso_params.c1,
            "c2": config.pso_params.c2,
        },
        "use_gpu": config.use_gpu,
        "seed": config.seed,
        "restricted_areas": [ra.model_dump() for ra in config.restricted_areas],
        "non_critical_areas": [nca.model_dump() for nca in config.non_critical_areas],
        "intersections": [it.model_dump() for it in config.intersections],
        "cell_size": config.cell_size,
        "strategy": config.strategy,
        "method": config.method,
    }
