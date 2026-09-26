"""
controllers/history_controller.py
-----------------------------------
Request parsing + response shaping for experiment history endpoints.

Each handler:
  1. Calls the relevant repository function.
  2. Shapes the response (adds HTTP-friendly defaults, raises HTTPException).
  3. Does NOT contain business logic or database queries — delegates to
     run_repository directly (history is simple CRUD, no service needed).
"""

import logging

from fastapi import HTTPException

from app.db.repositories import run_repository

logger = logging.getLogger(__name__)


async def handle_list_runs(page: int, page_size: int) -> dict:
    """GET /history — paginated list of past runs."""
    try:
        return await run_repository.list_runs(page=page, page_size=page_size)
    except Exception as exc:
        logger.error("Failed to list runs: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


async def handle_get_run(job_id: str) -> dict:
    """GET /history/{job_id} — full result for a single run."""
    try:
        doc = await run_repository.get_run(job_id)
    except Exception as exc:
        logger.error("Failed to get run %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found in history")
    return doc


async def handle_delete_run(job_id: str) -> dict:
    """DELETE /history/{job_id} — remove a run."""
    try:
        deleted = await run_repository.delete_run(job_id)
    except Exception as exc:
        logger.error("Failed to delete run %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found in history")
    return {"deleted": True, "job_id": job_id}


async def handle_get_stats() -> dict:
    """GET /history/stats — aggregate statistics."""
    try:
        return await run_repository.get_stats()
    except Exception as exc:
        logger.error("Failed to get stats: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
