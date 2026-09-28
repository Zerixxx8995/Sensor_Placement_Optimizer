"""
routers/fault.py
-----------------
Layer 1 — URL routing ONLY for the fault-inject endpoint.
Zero logic, zero validation, zero data shaping.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.controllers.fault_controller import handle_fault_inject

router = APIRouter()


class FaultInjectRequest(BaseModel):
    """Request body for POST /fault-inject."""

    job_id: str = Field(..., description="UUID of a completed optimization job.")
    dropout_percent: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Percentage of nodes to randomly disable (0, 100].",
    )
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility.")


@router.post("/fault-inject", status_code=200)
async def post_fault_inject(body: FaultInjectRequest):
    """
    Simulate random node failures on a completed optimization result.
    Randomly disables dropout_percent of nodes, then recomputes coverage
    and connectivity on the degraded deployment.
    """
    return await handle_fault_inject(body.job_id, body.dropout_percent, seed=body.seed)
