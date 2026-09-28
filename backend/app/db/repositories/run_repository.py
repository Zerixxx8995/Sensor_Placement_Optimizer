"""
db/repositories/run_repository.py
-----------------------------------
Data-access layer for the ``optimization_runs`` MongoDB collection.

Provides:
  save_run()   – persist a completed PSO run document
  get_run()    – fetch a single run by job_id
  list_runs()  – paginated list of run summaries
  delete_run() – remove a run by job_id
  get_stats()  – aggregate statistics across all runs

All public functions are ``async`` and accept/return plain Python dicts
or None — no Motor types leak out of this module.

Collection schema (matches PSO_additions.md):
  {
    _id: ObjectId,
    job_id: str,
    created_at: datetime (UTC),
    config: { ... },
    result: {
      coverage_ratio, connectivity_ratio, avg_energy,
      compute_time_seconds, iterations_run, gpu_used,
      best_positions, fitness_history, coverage_map,
      strategy
    },
    surrogate_used: bool,
    surrogate_switch_iteration: int | None,
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
import numpy as np

from bson import ObjectId
from pymongo import DESCENDING

from app.db.connection import get_database


logger = logging.getLogger(__name__)

_COLLECTION = "optimization_runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(doc: dict) -> dict:
    """Convert ObjectId and datetime to JSON-safe types."""
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _clean_for_mongo(obj: Any) -> Any:
    """Recursively convert NumPy arrays and scalars to plain Python types for BSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _clean_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_mongo(v) for v in obj]
    return obj


async def save_run(job_id: str, config: dict, result: dict,
                   surrogate_used: bool = False,
                   surrogate_switch_iteration: int | None = None) -> str:
    """
    Persist a completed optimization run to MongoDB.

    Returns the inserted document's string ``_id``.
    Raises on any database error (caller should catch).
    """
    db = get_database()
    cleaned_result = _clean_for_mongo(result)
    cleaned_config = _clean_for_mongo(config)

    doc = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc),
        "config": cleaned_config,
        "result": {
            "coverage_ratio": cleaned_result.get("coverage_ratio"),
            "connectivity_ratio": cleaned_result.get("connectivity_ratio"),
            "avg_energy": cleaned_result.get("avg_energy"),
            "compute_time_seconds": cleaned_result.get("compute_time_seconds"),
            "iterations_run": cleaned_result.get("iterations_run"),
            "gpu_used": cleaned_result.get("gpu_used", False),
            "best_positions": cleaned_result.get("best_positions", []),
            "fitness_history": cleaned_result.get("fitness_history", []),
            "coverage_map": cleaned_result.get("coverage_map", []),
            "strategy": cleaned_result.get("strategy", cleaned_config.get("strategy", "pso")),
        },
        "surrogate_used": surrogate_used,
        "surrogate_switch_iteration": surrogate_switch_iteration,
    }
    inserted = await db[_COLLECTION].insert_one(doc)
    logger.debug("Saved run %s → _id=%s", job_id, inserted.inserted_id)
    return str(inserted.inserted_id)



async def get_run(job_id: str) -> dict | None:
    """
    Return the full document for ``job_id``, or None if not found.
    """
    db = get_database()
    doc = await db[_COLLECTION].find_one({"job_id": job_id})
    return _serialize(doc) if doc else None


async def list_runs(page: int = 1, page_size: int = 20) -> dict:
    """
    Return a paginated summary list.

    Each item contains: job_id, created_at, strategy, num_nodes,
    coverage_ratio, compute_time_seconds, gpu_used.
    """
    db = get_database()
    skip = (page - 1) * page_size
    projection = {
        "job_id": 1,
        "created_at": 1,
        "result.coverage_ratio": 1,
        "result.compute_time_seconds": 1,
        "result.gpu_used": 1,
        "result.strategy": 1,
        "config.num_nodes": 1,
    }
    cursor = (
        db[_COLLECTION]
        .find({}, projection)
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(page_size)
    )
    docs = await cursor.to_list(length=page_size)
    total = await db[_COLLECTION].count_documents({})

    runs = []
    for doc in docs:
        runs.append({
            "job_id": doc.get("job_id"),
            "created_at": doc.get("created_at").isoformat()
                          if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at")),
            "strategy": doc.get("result", {}).get("strategy"),
            "num_nodes": doc.get("config", {}).get("num_nodes"),
            "coverage_ratio": doc.get("result", {}).get("coverage_ratio"),
            "compute_time_seconds": doc.get("result", {}).get("compute_time_seconds"),
            "gpu_used": doc.get("result", {}).get("gpu_used", False),
        })

    return {
        "runs": runs,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


async def delete_run(job_id: str) -> bool:
    """
    Delete the document for ``job_id``.
    Returns True if a document was deleted, False if none found.
    """
    db = get_database()
    result = await db[_COLLECTION].delete_one({"job_id": job_id})
    return result.deleted_count > 0


async def get_stats() -> dict:
    """
    Aggregate statistics across all stored runs.

    Returns:
      total_runs, best_coverage_ever, avg_compute_time,
      runs_per_strategy (dict), most_used_strategy.
    """
    db = get_database()
    total = await db[_COLLECTION].count_documents({})

    if total == 0:
        return {
            "total_runs": 0,
            "best_coverage_ever": None,
            "avg_compute_time": None,
            "runs_per_strategy": {},
            "most_used_strategy": None,
        }

    pipeline = [
        {
            "$group": {
                "_id": None,
                "best_coverage": {"$max": "$result.coverage_ratio"},
                "avg_compute_time": {"$avg": "$result.compute_time_seconds"},
            }
        }
    ]
    agg = await db[_COLLECTION].aggregate(pipeline).to_list(1)
    best_coverage = agg[0]["best_coverage"] if agg else None
    avg_time = agg[0]["avg_compute_time"] if agg else None

    # Per-strategy breakdown
    strategy_pipeline = [
        {"$group": {"_id": "$result.strategy", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    strategy_docs = await db[_COLLECTION].aggregate(strategy_pipeline).to_list(20)
    runs_per_strategy = {d["_id"]: d["count"] for d in strategy_docs if d["_id"]}
    most_used = strategy_docs[0]["_id"] if strategy_docs else None

    return {
        "total_runs": total,
        "best_coverage_ever": best_coverage,
        "avg_compute_time": avg_time,
        "runs_per_strategy": runs_per_strategy,
        "most_used_strategy": most_used,
    }


async def count_runs() -> int:
    """Return total number of stored runs (used by surrogate training trigger)."""
    db = get_database()
    return await db[_COLLECTION].count_documents({})


async def get_training_samples(limit: int = 500) -> list[dict]:
    """
    Return the most recent ``limit`` runs as lightweight dicts for
    surrogate model training: { config, fitness_history, coverage_ratio }.
    """
    db = get_database()
    projection = {
        "config": 1,
        "result.fitness_history": 1,
        "result.coverage_ratio": 1,
        "result.best_positions": 1,
    }
    cursor = (
        db[_COLLECTION]
        .find({}, projection)
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return [
        {
            "config": doc.get("config", {}),
            "fitness_history": doc.get("result", {}).get("fitness_history", []),
            "coverage_ratio": doc.get("result", {}).get("coverage_ratio"),
            "best_positions": doc.get("result", {}).get("best_positions", []),
        }
        for doc in docs
    ]
