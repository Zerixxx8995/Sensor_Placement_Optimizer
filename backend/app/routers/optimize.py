"""
routers/optimize.py
---------------------
Layer 1 — URL routing ONLY.

Maps HTTP method + URL pattern to a controller function.
Contains zero logic, zero validation, zero data shaping.
"""

import queue
import json
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.models.config import OptimizationConfig
from app.controllers.optimize_controller import (
    handle_submit,
    handle_status,
    handle_result,
)
from app.services import optimization_service
from app.jobs import job_store

router = APIRouter()


@router.post("/optimize", status_code=200)
def post_optimize(config: OptimizationConfig, background_tasks: BackgroundTasks):
    """Submit a new PSO optimization job. Returns job_id immediately."""
    return handle_submit(config, background_tasks)


@router.get("/optimize/{job_id}/status", status_code=200)
def get_status(job_id: str):
    """Poll the lifecycle state of a submitted job."""
    return handle_status(job_id)


@router.get("/optimize/{job_id}/result", status_code=200)
def get_result(job_id: str):
    """Retrieve the full result of a completed job."""
    return handle_result(job_id)


@router.get("/optimize/{job_id}/stream", status_code=200)
def stream_job(job_id: str):
    """
    Stream real-time optimization progress via SSE.
    """
    job = optimization_service.get_job_status(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        )

    q = job_store.subscribe_job(job_id)

    def event_generator():
        try:
            # Yield connection established event
            yield f"data: {json.dumps({'event': 'connected'})}\n\n"

            while True:
                try:
                    event_data = q.get(timeout=1.0)
                except queue.Empty:
                    # Fallback check to ensure we don't stream forever if connection missed sentinel
                    current_job = job_store.get_job(job_id)
                    if current_job and current_job["status"] in ("complete", "failed"):
                        # Yield terminal event
                        if current_job["status"] == "complete":
                            yield f"data: {json.dumps({'event': 'complete', 'result': handle_result(job_id)})}\n\n"
                        else:
                            yield f"data: {json.dumps({'event': 'failed', 'error': current_job.get('error', 'unknown error')})}\n\n"
                        break
                    continue

                yield f"data: {json.dumps(event_data)}\n\n"

                if event_data.get("event") in ("complete", "failed"):
                    break
        finally:
            job_store.unsubscribe_job(job_id, q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
