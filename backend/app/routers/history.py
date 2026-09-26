"""
routers/history.py
-------------------
URL → controller binding for experiment history endpoints.

Routes:
  GET  /api/v1/history               → list past runs (paginated)
  GET  /api/v1/history/stats         → aggregate statistics
  GET  /api/v1/history/{job_id}      → full result for a single run
  DELETE /api/v1/history/{job_id}    → delete a run

No logic here — pure routing only.
"""

from fastapi import APIRouter, Query

from app.controllers.history_controller import (
    handle_list_runs,
    handle_get_run,
    handle_delete_run,
    handle_get_stats,
)

router = APIRouter()


@router.get("/history", summary="List past optimization runs (paginated)")
async def list_runs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    return await handle_list_runs(page=page, page_size=page_size)


@router.get("/history/stats", summary="Aggregate statistics across all runs")
async def get_stats():
    return await handle_get_stats()


@router.get("/history/{job_id}", summary="Full result for a past run")
async def get_run(job_id: str):
    return await handle_get_run(job_id=job_id)


@router.delete("/history/{job_id}", summary="Delete a past run")
async def delete_run(job_id: str):
    return await handle_delete_run(job_id=job_id)
