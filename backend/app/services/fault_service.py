"""
services/fault_service.py
--------------------------
Business logic for POST /api/v1/fault-inject.

Takes a completed optimization result, randomly disables dropout_percent of
sensor nodes, and returns before/after coverage metrics plus the degraded
coverage map.

No HTTP knowledge — pure orchestration and math only.
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from app.core.sensing_model import coverage_map as compute_coverage_map
from app.core.fitness import _connectivity_ratio
from app.jobs import job_store


from app.db.repositories import run_repository


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_fault_injection(job_id: str, dropout_percent: float, seed: int | None = None) -> dict:
    """
    Simulate random node failures on a completed optimization result.

    Args:
        job_id:          ID of a completed optimization job.
        dropout_percent: Percentage of nodes to disable (must be in (0, 100]).
        seed:            Optional RNG seed for reproducibility.

    Returns:
        {
          "job_id":                  str,
          "original_coverage_ratio": float,
          "degraded_coverage_ratio": float,
          "nodes_failed":            int,
          "total_nodes":             int,
          "dropout_percent":         float,
          "coverage_map":            list[list[float]],
          "connectivity_ratio":      float,
        }

    Raises:
        ValueError: If the job does not exist, is not complete, or has no positions.
    """
    job = job_store.get_job(job_id)
    raw = None

    if job is not None:
        if job["status"] != "complete":
            raise ValueError(
                f"Job '{job_id}' is not complete (status={job['status']}). "
                "Fault injection requires a completed optimization result."
            )
        raw = job.get("result")
    else:
        # Fallback to MongoDB repository if job is not in memory job_store
        db_doc = await run_repository.get_run(job_id)
        if db_doc is not None:
            raw = db_doc.get("result")

    config = job.get("config", {}) if job else (db_doc.get("config", {}) if db_doc else {})

    if not raw:
        raise ValueError(f"Job '{job_id}' not found.")

    return _compute_fault_injection(job_id, raw, dropout_percent, seed, config=config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_fault_injection(
    job_id: str,
    raw: dict,
    dropout_percent: float,
    seed: int | None,
    config: dict | None = None,
) -> dict:
    """Core fault injection computation — works on raw result dicts."""
    config = config or {}

    # --- Unpack positions from result (comes in as list[list[float]]) ---
    positions = np.array(raw["best_positions"], dtype=np.float64)   # (N, 2)
    total_nodes = len(positions)

    cov_map_stored = np.array(raw.get("coverage_map", [[]]), dtype=np.float64)
    rows, cols = cov_map_stored.shape if cov_map_stored.ndim == 2 else (1, 1)

    area = config.get("area", {})
    if area.get("width") and area.get("height"):
        area_W = float(area["width"])
        area_H = float(area["height"])
    else:
        xs, ys = positions[:, 0], positions[:, 1]
        area_W = float(max(xs.max() * 1.05, 1.0))
        area_H = float(max(ys.max() * 1.05, 1.0))

    if config.get("sensing_radius"):
        Rs = float(config["sensing_radius"])
    elif cols > 1:
        Rs = (area_W / cols) * 2.0
    else:
        Rs = 2.0

    if config.get("cell_size"):
        cell_size = float(config["cell_size"])
    elif cols > 1:
        cell_size = area_W / cols
    else:
        cell_size = 1.0

    if config.get("comm_radius"):
        Rc = float(config["comm_radius"])
    else:
        Rc = Rs * 2.0

    # ---  Better approach: use stored coverage_ratio to skip recalculation ---
    original_coverage_ratio = float(raw.get("coverage_ratio", 0.0))

    # --- Determine which nodes fail ---
    nodes_failed = max(1, math.ceil(total_nodes * dropout_percent / 100.0))
    nodes_failed = min(nodes_failed, total_nodes)

    rng = np.random.default_rng(seed)
    failed_indices = rng.choice(total_nodes, size=nodes_failed, replace=False)
    surviving_mask = np.ones(total_nodes, dtype=bool)
    surviving_mask[failed_indices] = False
    surviving_positions = positions[surviving_mask]

    # --- Recompute coverage on surviving nodes ---
    if len(surviving_positions) == 0:
        degraded_cov_map = np.zeros((rows, cols), dtype=np.float64)
        degraded_coverage_ratio = 0.0
        degraded_connectivity = 0.0
    else:
        degraded_cov_map = compute_coverage_map(
            surviving_positions, area_W, area_H, Rs,
            cell_size=cell_size,
        )
        degraded_coverage_ratio = float(np.mean(degraded_cov_map))

        degraded_connectivity = _connectivity_ratio(
            surviving_positions, Rc, sink=(0.0, 0.0)
        )

    return {
        "job_id": str(uuid.uuid4()),
        "original_coverage_ratio": original_coverage_ratio,
        "degraded_coverage_ratio": degraded_coverage_ratio,
        "nodes_failed": nodes_failed,
        "total_nodes": total_nodes,
        "dropout_percent": dropout_percent,
        "coverage_map": degraded_cov_map.tolist(),
        "connectivity_ratio": degraded_connectivity,
        "failed_indices": failed_indices.tolist(),
    }
