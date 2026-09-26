"""
tests/test_history.py
----------------------
Unit and integration tests for the MongoDB experiment history feature.

Tests:
  1. save_run() writes correct fields
  2. GET /history returns paginated list
  3. GET /history/{job_id} returns full result shape
  4. DELETE /history/{job_id} removes document
  5. MongoDB failure does NOT fail the optimization job
  6. GET /history/stats returns aggregate structure

Uses mongomock-motor for in-process MongoDB mocking (no real MongoDB needed).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "area": {"width": 100, "height": 100},
    "num_nodes": 10,
    "sensing_radius": 15.0,
    "comm_radius": 30.0,
    "initial_energy": 1.0,
    "weights": {"w1": 0.5, "w2": 0.25, "w3": 0.25},
    "pso_params": {"swarm_size": 20, "iterations": 50, "inertia": 0.7, "c1": 1.5, "c2": 1.5},
    "use_gpu": False,
    "strategy": "pso",
}

SAMPLE_RESULT = {
    "coverage_ratio": 0.85,
    "connectivity_ratio": 0.92,
    "avg_energy": 0.78,
    "compute_time_seconds": 3.5,
    "iterations_run": 50,
    "gpu_used": False,
    "best_positions": [[10.0, 20.0], [50.0, 60.0]],
    "fitness_history": [0.9, 0.85, 0.82],
    "coverage_map": [[0.5, 0.8], [0.9, 1.0]],
    "strategy": "pso",
}


# ---------------------------------------------------------------------------
# Helper: build an in-memory async mock collection
# ---------------------------------------------------------------------------

def _make_mock_collection(docs: list[dict] | None = None):
    """Return a mock that behaves like a Motor collection for our tests."""
    stored = list(docs or [])

    async def insert_one(doc):
        from bson import ObjectId
        doc = dict(doc)
        doc["_id"] = ObjectId()
        stored.append(doc)
        m = MagicMock()
        m.inserted_id = doc["_id"]
        return m

    async def find_one(query):
        for doc in stored:
            if doc.get("job_id") == query.get("job_id"):
                return doc
        return None

    async def delete_one(query):
        idx = next((i for i, d in enumerate(stored) if d.get("job_id") == query.get("job_id")), None)
        m = MagicMock()
        if idx is not None:
            stored.pop(idx)
            m.deleted_count = 1
        else:
            m.deleted_count = 0
        return m

    async def count_documents(query):
        return len(stored)

    col = MagicMock()
    col.insert_one = insert_one
    col.find_one = find_one
    col.delete_one = delete_one
    col.count_documents = count_documents

    # find() with chained .sort().skip().limit()
    async def to_list(length=None):
        return stored[:length] if length else stored

    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = to_list
    col.find.return_value = cursor

    # aggregate()
    async def agg_to_list(n):
        return [{"best_coverage": 0.85, "avg_compute_time": 3.5}]

    agg_cursor = MagicMock()
    agg_cursor.to_list = agg_to_list
    col.aggregate.return_value = agg_cursor

    return col, stored


# ---------------------------------------------------------------------------
# Test: save_run writes correct fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_run_writes_correct_fields():
    col, stored = _make_mock_collection()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=col)

    with patch("app.db.repositories.run_repository.get_database", return_value=mock_db):
        from app.db.repositories import run_repository
        doc_id = await run_repository.save_run(
            job_id="test-job-001",
            config=SAMPLE_CONFIG,
            result=SAMPLE_RESULT,
        )

    assert doc_id is not None
    assert len(stored) == 1
    saved = stored[0]
    assert saved["job_id"] == "test-job-001"
    assert saved["config"]["num_nodes"] == 10
    assert saved["result"]["coverage_ratio"] == 0.85
    assert saved["result"]["strategy"] == "pso"
    assert saved["surrogate_used"] is False
    assert saved["surrogate_switch_iteration"] is None
    assert "created_at" in saved


# ---------------------------------------------------------------------------
# Test: get_run returns correct document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_run_returns_document():
    from datetime import datetime, timezone
    from bson import ObjectId

    pre_stored = [{
        "_id": ObjectId(),
        "job_id": "test-job-001",
        "created_at": datetime.now(timezone.utc),
        "config": SAMPLE_CONFIG,
        "result": SAMPLE_RESULT,
        "surrogate_used": False,
        "surrogate_switch_iteration": None,
    }]
    col, _ = _make_mock_collection(pre_stored)
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=col)

    with patch("app.db.repositories.run_repository.get_database", return_value=mock_db):
        from app.db.repositories import run_repository
        doc = await run_repository.get_run("test-job-001")

    assert doc is not None
    assert doc["job_id"] == "test-job-001"
    assert "result" in doc
    assert doc["result"]["coverage_ratio"] == 0.85


# ---------------------------------------------------------------------------
# Test: GET /history returns paginated list (API layer)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_list_runs_returns_paginated():
    mock_list = AsyncMock(return_value={
        "runs": [{"job_id": "abc", "strategy": "pso", "num_nodes": 10, "coverage_ratio": 0.85,
                  "created_at": "2024-01-01T00:00:00", "compute_time_seconds": 3.5, "gpu_used": False}],
        "total": 1, "page": 1, "page_size": 20, "total_pages": 1
    })
    with patch("app.controllers.history_controller.run_repository.list_runs", mock_list):
        from app.controllers.history_controller import handle_list_runs
        result = await handle_list_runs(page=1, page_size=20)

    assert "runs" in result
    assert len(result["runs"]) == 1
    assert result["runs"][0]["job_id"] == "abc"
    assert result["total"] == 1


# ---------------------------------------------------------------------------
# Test: GET /history/{job_id} returns correct shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_get_run_returns_full_shape():
    from datetime import datetime, timezone

    full_doc = {
        "_id": "some-oid",
        "job_id": "test-job-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": SAMPLE_CONFIG,
        "result": SAMPLE_RESULT,
        "surrogate_used": False,
        "surrogate_switch_iteration": None,
    }
    mock_get = AsyncMock(return_value=full_doc)
    with patch("app.controllers.history_controller.run_repository.get_run", mock_get):
        from app.controllers.history_controller import handle_get_run
        result = await handle_get_run("test-job-001")

    assert result["job_id"] == "test-job-001"
    assert "config" in result
    assert "result" in result
    assert result["result"]["best_positions"] == [[10.0, 20.0], [50.0, 60.0]]


# ---------------------------------------------------------------------------
# Test: DELETE /history/{job_id} removes document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_delete_run_removes_document():
    mock_delete = AsyncMock(return_value=True)
    with patch("app.controllers.history_controller.run_repository.delete_run", mock_delete):
        from app.controllers.history_controller import handle_delete_run
        result = await handle_delete_run("test-job-001")

    assert result["deleted"] is True
    assert result["job_id"] == "test-job-001"
    mock_delete.assert_awaited_once_with("test-job-001")


@pytest.mark.asyncio
async def test_api_delete_run_not_found_raises_404():
    from fastapi import HTTPException
    mock_delete = AsyncMock(return_value=False)
    with patch("app.controllers.history_controller.run_repository.delete_run", mock_delete):
        from app.controllers.history_controller import handle_delete_run
        with pytest.raises(HTTPException) as exc_info:
            await handle_delete_run("nonexistent-job")
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test: MongoDB failure does NOT fail the optimization job
# ---------------------------------------------------------------------------

def test_mongodb_failure_does_not_fail_optimization_job():
    """
    The run_optimization_job should complete successfully even if MongoDB
    throws an exception during save_run.
    """
    import importlib
    from unittest.mock import patch, MagicMock

    # Build a minimal config
    config = {
        "strategy": "random",
        "area": {"width": 50, "height": 50},
        "num_nodes": 5,
        "sensing_radius": 10.0,
        "comm_radius": 20.0,
        "initial_energy": 1.0,
        "weights": {"w1": 0.5, "w2": 0.25, "w3": 0.25},
        "pso_params": {"swarm_size": 10, "iterations": 10, "inertia": 0.7, "c1": 1.5, "c2": 1.5},
        "use_gpu": False,
        "seed": 0,
        "restricted_areas": [],
        "non_critical_areas": [],
        "cell_size": 1.0,
    }

    from app.jobs import job_store

    job_store.create_job("fail-mongo-job")

    # Patch asyncio.run in the optimization_job module to raise an exception
    with patch("app.jobs.optimization_job.asyncio.run", side_effect=Exception("Mongo is down")):
        from app.jobs.optimization_job import run_optimization_job
        run_optimization_job("fail-mongo-job", config)

    state = job_store.get_job("fail-mongo-job")
    # Job must be "complete", not "failed"
    assert state["status"] == "complete", (
        f"Expected status=complete, got {state['status']}"
    )


# ---------------------------------------------------------------------------
# Test: GET /history/stats returns aggregate structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_get_stats_returns_aggregate():
    expected_stats = {
        "total_runs": 5,
        "best_coverage_ever": 0.93,
        "avg_compute_time": 4.2,
        "runs_per_strategy": {"pso": 3, "random": 2},
        "most_used_strategy": "pso",
    }
    mock_stats = AsyncMock(return_value=expected_stats)
    with patch("app.controllers.history_controller.run_repository.get_stats", mock_stats):
        from app.controllers.history_controller import handle_get_stats
        result = await handle_get_stats()

    assert result["total_runs"] == 5
    assert result["best_coverage_ever"] == 0.93
    assert "runs_per_strategy" in result
    assert result["most_used_strategy"] == "pso"
