"""
controllers/fault_controller.py
---------------------------------
Request / response handling for POST /api/v1/fault-inject.

Responsibilities:
  - Accept the raw request body.
  - Run input validation via the validator layer.
  - Call fault_service.run_fault_injection().
  - Raise HTTPException for validation failures or missing jobs.
  - Return the shaped response dict.

No business logic, no algorithm code.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.validators.fault_validator import validate_fault_input
from app.services import fault_service


async def handle_fault_inject(job_id: str, dropout_percent: float, seed: int | None = None) -> dict:
    """
    POST /fault-inject — validate inputs → run injection → return result.
    Raises 422 for invalid inputs, 404 if the job doesn't exist or isn't done.
    """
    errors = validate_fault_input(job_id, dropout_percent)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    try:
        result = await fault_service.run_fault_injection(job_id, dropout_percent, seed=seed)
    except ValueError as exc:
        msg = str(exc)
        # Distinguish "not found" from "not complete" to return appropriate codes
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=422, detail=msg) from exc

    return result
