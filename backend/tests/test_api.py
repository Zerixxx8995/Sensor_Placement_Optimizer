"""
tests/test_api.py
-----------------
Integration tests for the optimize API endpoints.

Uses FastAPI's TestClient (synchronous). When using TestClient, FastAPI
BackgroundTasks execute synchronously before the test can send the next
request — so after POST /optimize, the job will already be complete by the
time we poll /status.

Covers:
  - GET /api/v1/health → 200
  - POST /api/v1/optimize (valid)  → 200, returns job_id
  - POST /api/v1/optimize (bad weights)  → 422
  - POST /api/v1/optimize (Rs >= Rc)     → 422
  - POST /api/v1/optimize (missing field) → 422 (Pydantic)
  - GET /api/v1/optimize/{job_id}/status → 200 with valid status value
  - GET /api/v1/optimize/unknown-id/status → 404
  - GET /api/v1/optimize/{job_id}/result  → 200 with correct shape
  - GET /api/v1/optimize/unknown-id/result → 404
  - Result payload has all required keys
  - best_positions shape = [[x,y], ...] with num_nodes entries
  - fitness_history is non-empty
  - coverage_ratio and connectivity_ratio in [0, 1]
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.jobs import job_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_job_store():
    """Wipe the in-memory store before every test to prevent state leakage."""
    job_store.clear_all()
    yield
    job_store.clear_all()


@pytest.fixture(scope="module")
def client():
    """Shared TestClient for the module. BackgroundTasks run synchronously."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Minimal fast config (small field, few nodes, few iterations)
# ---------------------------------------------------------------------------

FAST_CONFIG = {
    "area": {"width": 50.0, "height": 50.0},
    "num_nodes": 5,
    "sensing_radius": 8.0,
    "comm_radius": 16.0,
    "initial_energy": 1.0,
    "weights": {"w1": 0.5, "w2": 0.25, "w3": 0.25},
    "pso_params": {
        "swarm_size": 5,
        "iterations": 5,
        "inertia": 0.7,
        "c1": 1.5,
        "c2": 1.5,
    },
    "use_gpu": False,
    "use_vdcoa": False,
    "seed": 42,
    "restricted_areas": [],
    "non_critical_areas": [],
    "strategy": "pso",
    "cell_size": 5.0,
}


def post_optimize(client, config=None):
    """Helper: POST /optimize and return the Response object."""
    return client.post("/api/v1/optimize", json=config or FAST_CONFIG)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/v1/optimize — submission
# ---------------------------------------------------------------------------

class TestPostOptimize:

    def test_valid_config_returns_200(self, client):
        resp = post_optimize(client)
        assert resp.status_code == 200

    def test_valid_config_returns_job_id(self, client):
        resp = post_optimize(client)
        body = resp.json()
        assert "job_id" in body
        assert isinstance(body["job_id"], str)
        assert len(body["job_id"]) > 0

    def test_valid_config_returns_pending_status(self, client):
        """Status in the POST response is 'pending' (before background runs)."""
        resp = post_optimize(client)
        body = resp.json()
        assert "status" in body
        # The background task may or may not have completed; status field must exist
        assert body["status"] in ("pending", "running", "complete")

    def test_bad_weights_returns_422(self, client):
        bad = {**FAST_CONFIG, "weights": {"w1": 0.0, "w2": 0.0, "w3": 0.0}}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_rs_equals_rc_returns_422(self, client):
        bad = {**FAST_CONFIG, "sensing_radius": 16.0, "comm_radius": 16.0}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_rs_greater_than_rc_returns_422(self, client):
        bad = {**FAST_CONFIG, "sensing_radius": 30.0, "comm_radius": 10.0}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        """Pydantic should reject a payload missing 'area'."""
        bad = {k: v for k, v in FAST_CONFIG.items() if k != "area"}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_unknown_strategy_returns_422(self, client):
        bad = {**FAST_CONFIG, "strategy": "genetic_algorithm"}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_negative_num_nodes_returns_422(self, client):
        bad = {**FAST_CONFIG, "num_nodes": 0}
        resp = post_optimize(client, bad)
        assert resp.status_code == 422

    def test_two_jobs_get_different_ids(self, client):
        id1 = post_optimize(client).json()["job_id"]
        id2 = post_optimize(client).json()["job_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# GET /api/v1/optimize/{job_id}/status
# ---------------------------------------------------------------------------

class TestGetStatus:

    def test_known_job_returns_200(self, client):
        job_id = post_optimize(client).json()["job_id"]
        resp = client.get(f"/api/v1/optimize/{job_id}/status")
        assert resp.status_code == 200

    def test_known_job_has_status_field(self, client):
        job_id = post_optimize(client).json()["job_id"]
        body = client.get(f"/api/v1/optimize/{job_id}/status").json()
        assert "status" in body

    def test_known_job_status_is_valid_value(self, client):
        job_id = post_optimize(client).json()["job_id"]
        status = client.get(f"/api/v1/optimize/{job_id}/status").json()["status"]
        assert status in ("pending", "running", "complete", "failed")

    def test_known_job_returns_job_id_field(self, client):
        job_id = post_optimize(client).json()["job_id"]
        body = client.get(f"/api/v1/optimize/{job_id}/status").json()
        assert body["job_id"] == job_id

    def test_unknown_job_id_returns_404(self, client):
        resp = client.get("/api/v1/optimize/does-not-exist/status")
        assert resp.status_code == 404

    def test_background_task_completes(self, client):
        """
        With TestClient, BackgroundTasks run synchronously.
        After POST the job should be complete when we poll status.
        """
        job_id = post_optimize(client).json()["job_id"]
        status = client.get(f"/api/v1/optimize/{job_id}/status").json()["status"]
        assert status == "complete"


# ---------------------------------------------------------------------------
# GET /api/v1/optimize/{job_id}/result
# ---------------------------------------------------------------------------

class TestGetResult:

    def _submit_and_get_result(self, client):
        job_id = post_optimize(client).json()["job_id"]
        resp = client.get(f"/api/v1/optimize/{job_id}/result")
        return job_id, resp

    def test_completed_job_returns_200(self, client):
        _, resp = self._submit_and_get_result(client)
        assert resp.status_code == 200

    def test_unknown_job_returns_404(self, client):
        resp = client.get("/api/v1/optimize/no-such-job/result")
        assert resp.status_code == 404

    def test_result_has_required_keys(self, client):
        _, resp = self._submit_and_get_result(client)
        body = resp.json()
        required = [
            "job_id", "status", "best_positions", "coverage_ratio",
            "connectivity_ratio", "avg_energy", "fitness_history",
            "coverage_map", "compute_time_seconds", "iterations_run", "gpu_used",
        ]
        for key in required:
            assert key in body, f"Missing key in result: {key}"

    def test_result_status_is_complete(self, client):
        _, resp = self._submit_and_get_result(client)
        assert resp.json()["status"] == "complete"

    def test_best_positions_shape(self, client):
        """best_positions must be a list of [x, y] pairs = num_nodes entries."""
        _, resp = self._submit_and_get_result(client)
        positions = resp.json()["best_positions"]
        assert isinstance(positions, list)
        assert len(positions) == FAST_CONFIG["num_nodes"]
        for p in positions:
            assert len(p) == 2

    def test_coverage_ratio_in_range(self, client):
        _, resp = self._submit_and_get_result(client)
        assert 0.0 <= resp.json()["coverage_ratio"] <= 1.0

    def test_connectivity_ratio_in_range(self, client):
        _, resp = self._submit_and_get_result(client)
        assert 0.0 <= resp.json()["connectivity_ratio"] <= 1.0

    def test_fitness_history_non_empty(self, client):
        _, resp = self._submit_and_get_result(client)
        history = resp.json()["fitness_history"]
        assert isinstance(history, list)
        assert len(history) > 0

    def test_fitness_history_length(self, client):
        """Length = iterations + 1 (initial + one per iter)."""
        _, resp = self._submit_and_get_result(client)
        history = resp.json()["fitness_history"]
        assert len(history) == FAST_CONFIG["pso_params"]["iterations"] + 1

    def test_gpu_used_is_false(self, client):
        _, resp = self._submit_and_get_result(client)
        assert resp.json()["gpu_used"] is False

    def test_coverage_map_is_2d_list(self, client):
        _, resp = self._submit_and_get_result(client)
        cov_map = resp.json()["coverage_map"]
        assert isinstance(cov_map, list)
        assert isinstance(cov_map[0], list)

    def test_result_job_id_matches(self, client):
        job_id, resp = self._submit_and_get_result(client)
        assert resp.json()["job_id"] == job_id

    def test_compute_time_positive(self, client):
        _, resp = self._submit_and_get_result(client)
        assert resp.json()["compute_time_seconds"] > 0.0

    def test_iterations_run_matches_config(self, client):
        _, resp = self._submit_and_get_result(client)
        assert resp.json()["iterations_run"] == FAST_CONFIG["pso_params"]["iterations"]


# ---------------------------------------------------------------------------
# GET /api/v1/optimize/{job_id}/stream
# ---------------------------------------------------------------------------

class TestGetStream:

    def test_stream_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/v1/optimize/nonexistent-uuid/stream")
        assert resp.status_code == 404

    def test_stream_returns_event_stream(self, client):
        import json
        resp_submit = client.post("/api/v1/optimize", json=FAST_CONFIG)
        job_id = resp_submit.json()["job_id"]

        with client.stream("GET", f"/api/v1/optimize/{job_id}/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            events = []
            for line in response.iter_lines():
                if line.strip().startswith("data: "):
                    data_str = line.strip()[6:]
                    event_data = json.loads(data_str)
                    events.append(event_data)
                    if event_data.get("event") in ("complete", "failed") or len(events) >= 5:
                        break
            
            assert len(events) > 0
            assert events[0]["event"] in ("connected", "iteration", "complete")


# ---------------------------------------------------------------------------
# POST /api/v1/compare
# ---------------------------------------------------------------------------

EXPECTED_STRATEGIES = {"random", "grid", "pso", "pso_vdcoa", "rl"}
STRATEGY_METRIC_KEYS = {
    "strategy", "coverage_ratio", "connectivity_ratio",
    "avg_energy", "compute_time_seconds",
}


class TestPostCompare:

    def _post_compare(self, client, config=None):
        return client.post("/api/v1/compare", json=config or FAST_CONFIG)

    # --- Successful requests ---

    def test_valid_config_returns_200(self, client):
        resp = self._post_compare(client)
        assert resp.status_code == 200

    def test_response_has_job_id(self, client):
        body = self._post_compare(client).json()
        assert "job_id" in body
        assert isinstance(body["job_id"], str)
        assert len(body["job_id"]) > 0

    def test_response_status_is_complete(self, client):
        body = self._post_compare(client).json()
        assert body["status"] == "complete"

    def test_response_has_results_list(self, client):
        body = self._post_compare(client).json()
        assert "results" in body
        assert isinstance(body["results"], list)

    def test_results_has_five_entries(self, client):
        results = self._post_compare(client).json()["results"]
        assert len(results) == 5, f"Expected 5 strategy results, got {len(results)}"

    def test_all_five_strategies_present(self, client):
        results = self._post_compare(client).json()["results"]
        returned = {r["strategy"] for r in results}
        assert returned == EXPECTED_STRATEGIES, (
            f"Missing strategies: {EXPECTED_STRATEGIES - returned}"
        )

    def test_each_result_has_metric_keys(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            missing = STRATEGY_METRIC_KEYS - set(r.keys())
            assert not missing, f"Strategy '{r.get('strategy')}' missing keys: {missing}"

    # --- Metric value ranges ---

    def test_coverage_ratios_in_range(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            assert 0.0 <= r["coverage_ratio"] <= 1.0, (
                f"{r['strategy']}: coverage_ratio={r['coverage_ratio']}"
            )

    def test_connectivity_ratios_in_range(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            assert 0.0 <= r["connectivity_ratio"] <= 1.0, (
                f"{r['strategy']}: connectivity_ratio={r['connectivity_ratio']}"
            )

    def test_avg_energies_in_range(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            assert 0.0 <= r["avg_energy"] <= 1.0, (
                f"{r['strategy']}: avg_energy={r['avg_energy']}"
            )

    def test_compute_times_non_negative(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            assert r["compute_time_seconds"] >= 0.0

    def test_strategy_values_are_strings(self, client):
        results = self._post_compare(client).json()["results"]
        for r in results:
            assert isinstance(r["strategy"], str)

    # --- Invalid requests ---

    def test_bad_weights_returns_422(self, client):
        bad = {**FAST_CONFIG, "weights": {"w1": 0.0, "w2": 0.0, "w3": 0.0}}
        assert self._post_compare(client, bad).status_code == 422

    def test_rs_equals_rc_returns_422(self, client):
        bad = {**FAST_CONFIG, "sensing_radius": 16.0, "comm_radius": 16.0}
        assert self._post_compare(client, bad).status_code == 422

    def test_missing_area_returns_422(self, client):
        bad = {k: v for k, v in FAST_CONFIG.items() if k != "area"}
        assert self._post_compare(client, bad).status_code == 422

    def test_different_calls_return_different_job_ids(self, client):
        id1 = self._post_compare(client).json()["job_id"]
        id2 = self._post_compare(client).json()["job_id"]
        assert id1 != id2
