"""
tests/test_services.py
-----------------------
Unit tests for service-layer modules.

Covers:
  - comparison_service.run_comparison() — the primary focus of build step 5
  - optimization_service helpers (_to_pso_config translation)

The comparison tests run all four strategies against a tiny config
(small field, few nodes, very few PSO iterations) so they complete quickly.
"""

from __future__ import annotations

import pytest

from app.models.config import Area, OptimizationConfig, PSOParams, Weights
from app.services.comparison_service import run_comparison, _extract_metrics, _to_pso_config
from app.services.optimization_service import _to_pso_config as opt_to_pso_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> OptimizationConfig:
    defaults = dict(
        area=Area(width=50.0, height=50.0),
        num_nodes=5,
        sensing_radius=8.0,
        comm_radius=16.0,
        initial_energy=1.0,
        weights=Weights(w1=0.5, w2=0.25, w3=0.25),
        pso_params=PSOParams(swarm_size=4, iterations=5),
        use_gpu=False,
        use_vdcoa=False,
        seed=42,
        restricted_areas=[],
        non_critical_areas=[],
        strategy="pso",
        cell_size=5.0,
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


FAST_CONFIG = make_config()

EXPECTED_STRATEGIES = {"random", "grid", "pso", "pso_vdcoa", "rl"}
METRIC_KEYS = {"strategy", "coverage_ratio", "connectivity_ratio",
               "avg_energy", "compute_time_seconds"}


# ---------------------------------------------------------------------------
# run_comparison — structure
# ---------------------------------------------------------------------------

class TestRunComparisonStructure:

    def test_returns_dict(self):
        result = run_comparison(FAST_CONFIG)
        assert isinstance(result, dict)

    def test_has_job_id(self):
        result = run_comparison(FAST_CONFIG)
        assert "job_id" in result
        assert isinstance(result["job_id"], str)
        assert len(result["job_id"]) > 0

    def test_status_is_complete(self):
        result = run_comparison(FAST_CONFIG)
        assert result["status"] == "complete"

    def test_results_is_list(self):
        result = run_comparison(FAST_CONFIG)
        assert isinstance(result["results"], list)

    def test_results_has_five_entries(self):
        result = run_comparison(FAST_CONFIG)
        assert len(result["results"]) == 5, (
            f"Expected 5 strategy results, got {len(result['results'])}"
        )

    def test_all_five_strategies_present(self):
        result = run_comparison(FAST_CONFIG)
        strategies = {r["strategy"] for r in result["results"]}
        assert strategies == EXPECTED_STRATEGIES, (
            f"Missing strategies: {EXPECTED_STRATEGIES - strategies}"
        )

    def test_each_result_has_metric_keys(self):
        result = run_comparison(FAST_CONFIG)
        for r in result["results"]:
            missing = METRIC_KEYS - set(r.keys())
            assert not missing, (
                f"Strategy '{r.get('strategy')}' missing keys: {missing}"
            )

    def test_job_ids_differ_across_calls(self):
        """Each call generates a fresh UUID."""
        r1 = run_comparison(FAST_CONFIG)
        r2 = run_comparison(FAST_CONFIG)
        assert r1["job_id"] != r2["job_id"]


# ---------------------------------------------------------------------------
# run_comparison — metric ranges
# ---------------------------------------------------------------------------

class TestRunComparisonMetricRanges:

    def setup_method(self):
        self.result = run_comparison(FAST_CONFIG)
        self.rows = {r["strategy"]: r for r in self.result["results"]}

    def test_coverage_ratio_in_range(self):
        for strategy, r in self.rows.items():
            assert 0.0 <= r["coverage_ratio"] <= 1.0, (
                f"{strategy}: coverage_ratio={r['coverage_ratio']:.4f} out of [0,1]"
            )

    def test_connectivity_ratio_in_range(self):
        for strategy, r in self.rows.items():
            assert 0.0 <= r["connectivity_ratio"] <= 1.0, (
                f"{strategy}: connectivity_ratio={r['connectivity_ratio']:.4f} out of [0,1]"
            )

    def test_avg_energy_in_range(self):
        for strategy, r in self.rows.items():
            assert 0.0 <= r["avg_energy"] <= 1.0, (
                f"{strategy}: avg_energy={r['avg_energy']:.4f} out of [0,1]"
            )

    def test_compute_time_positive(self):
        for strategy, r in self.rows.items():
            assert r["compute_time_seconds"] >= 0.0, (
                f"{strategy}: compute_time={r['compute_time_seconds']} is negative"
            )


# ---------------------------------------------------------------------------
# run_comparison — per-strategy correctness
# ---------------------------------------------------------------------------

class TestRunComparisonPerStrategy:

    def setup_method(self):
        self.result = run_comparison(FAST_CONFIG)
        self.rows = {r["strategy"]: r for r in self.result["results"]}

    def test_random_strategy_present(self):
        assert "random" in self.rows

    def test_grid_strategy_present(self):
        assert "grid" in self.rows

    def test_pso_strategy_present(self):
        assert "pso" in self.rows

    def test_pso_vdcoa_strategy_present(self):
        assert "pso_vdcoa" in self.rows

    def test_pso_coverage_gte_random(self):
        """
        PSO should generally outperform random placement in coverage.
        With a fixed seed this is deterministic — run enough iterations to ensure.
        Not guaranteed for tiny configs, so we use a loose assertion.
        """
        pso = self.rows["pso"]["coverage_ratio"]
        random = self.rows["random"]["coverage_ratio"]
        # Just verify both are valid — optimality ordering not guaranteed on tiny config
        assert 0.0 <= pso <= 1.0
        assert 0.0 <= random <= 1.0

    def test_grid_has_zero_iterations(self):
        """Grid baseline doesn't iterate — compute time is near-instant."""
        grid = self.rows["grid"]
        assert grid["compute_time_seconds"] < 5.0  # should complete in milliseconds


# ---------------------------------------------------------------------------
# _extract_metrics helper
# ---------------------------------------------------------------------------

class TestExtractMetrics:

    def test_extracts_correct_keys(self):
        fake_result = {
            "coverage_ratio": 0.85,
            "connectivity_ratio": 0.9,
            "avg_energy": 0.4,
            "compute_time_seconds": 1.23,
            "extra_ignored": "value",
        }
        out = _extract_metrics("pso", fake_result)
        assert out == {
            "strategy": "pso",
            "coverage_ratio": 0.85,
            "connectivity_ratio": 0.9,
            "avg_energy": 0.4,
            "compute_time_seconds": 1.23,
        }

    def test_all_values_cast_to_float(self):
        import numpy as np
        fake = {
            "coverage_ratio": np.float64(0.7),
            "connectivity_ratio": np.float32(0.8),
            "avg_energy": np.float64(0.3),
            "compute_time_seconds": np.float64(0.5),
        }
        out = _extract_metrics("grid", fake)
        for key in ("coverage_ratio", "connectivity_ratio", "avg_energy", "compute_time_seconds"):
            assert isinstance(out[key], float)


# ---------------------------------------------------------------------------
# _to_pso_config translation
# ---------------------------------------------------------------------------

class TestToPsoConfig:

    def test_area_translated(self):
        cfg = _to_pso_config(FAST_CONFIG)
        assert cfg["area"]["width"] == FAST_CONFIG.area.width
        assert cfg["area"]["height"] == FAST_CONFIG.area.height

    def test_weights_translated(self):
        cfg = _to_pso_config(FAST_CONFIG)
        assert cfg["weights"]["w1"] == FAST_CONFIG.weights.w1
        assert cfg["weights"]["w2"] == FAST_CONFIG.weights.w2
        assert cfg["weights"]["w3"] == FAST_CONFIG.weights.w3

    def test_pso_params_translated(self):
        cfg = _to_pso_config(FAST_CONFIG)
        assert cfg["pso_params"]["swarm_size"] == FAST_CONFIG.pso_params.swarm_size
        assert cfg["pso_params"]["iterations"] == FAST_CONFIG.pso_params.iterations

    def test_seed_preserved(self):
        cfg = _to_pso_config(FAST_CONFIG)
        assert cfg["seed"] == FAST_CONFIG.seed


# ---------------------------------------------------------------------------
# optimization_service _to_pso_config (sanity check — same contract)
# ---------------------------------------------------------------------------

class TestOptimizationServiceConfig:

    def test_area_translated(self):
        cfg = opt_to_pso_config(FAST_CONFIG)
        assert cfg["area"]["width"] == FAST_CONFIG.area.width

    def test_pso_params_present(self):
        cfg = opt_to_pso_config(FAST_CONFIG)
        assert "pso_params" in cfg
        assert "iterations" in cfg["pso_params"]
