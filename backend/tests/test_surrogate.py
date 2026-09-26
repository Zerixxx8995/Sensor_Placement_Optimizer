"""
tests/test_surrogate.py
------------------------
Unit and integration tests for PyTorch surrogate model (Feature 1).
"""

import time
import pytest
import numpy as np
import torch
from unittest.mock import AsyncMock, patch

from app.core.surrogate_model import (
    FitnessSurrogateMLP,
    encode_features,
    predict_surrogate_batch,
    train_surrogate,
)
from app.core.fitness import compute_fitness
from app.core.pso import run_pso
from app.services import surrogate_service


def test_mlp_output_shape():
    """assert MLP output shape is (1,) scalar or (batch_size,) for tensor inputs."""
    model = FitnessSurrogateMLP()
    model.eval()

    single_input = torch.zeros((1, 207), dtype=torch.float32)
    output = model(single_input)
    assert output.shape == (1,)

    batch_input = torch.zeros((30, 207), dtype=torch.float32)
    batch_output = model(batch_input)
    assert batch_output.shape == (30,)


def test_predict_faster_than_analytical():
    """assert predict() is faster than analytical fitness by measurable margin."""
    model = FitnessSurrogateMLP()
    model.eval()

    N = 25
    P = 30
    positions = np.random.uniform(0, 100, size=(P, N, 2))
    cfg = {
        "area": {"width": 100, "height": 100},
        "num_nodes": N,
        "sensing_radius": 10,
        "comm_radius": 20,
        "weights": {"w1": 0.5, "w2": 0.3, "w3": 0.2},
    }

    fitness_cfg = {
        "area_W": 100.0,
        "area_H": 100.0,
        "Rs": 10.0,
        "Rc": 20.0,
        "w1": 0.5,
        "w2": 0.3,
        "w3": 0.2,
    }

    # Time analytical computation for 30 particles
    t0 = time.perf_counter()
    for i in range(P):
        _ = compute_fitness(positions[i], fitness_cfg, iteration=0, max_iterations=100)
    t_analytical = time.perf_counter() - t0

    # Time surrogate batch prediction for 30 particles
    t0 = time.perf_counter()
    _ = predict_surrogate_batch(model, positions, cfg)
    t_surrogate = time.perf_counter() - t0

    assert t_surrogate < t_analytical


@pytest.mark.asyncio
async def test_training_threshold():
    """assert training triggers correctly at 50 run threshold."""
    with patch("app.db.repositories.run_repository.count_runs", new_callable=AsyncMock) as mock_count:
        mock_count.return_value = 40
        status = await surrogate_service.get_surrogate_status()
        assert status["runs_needed"] == 10
        assert status["runs_available"] == 40

        mock_count.return_value = 55
        status_ready = await surrogate_service.get_surrogate_status()
        assert status_ready["runs_needed"] == 0
        assert status_ready["runs_available"] == 55


def test_pso_switching_logic():
    """assert switching logic engages at correct iteration threshold (0.7 * G)."""
    model = FitnessSurrogateMLP()
    model.eval()

    cfg = {
        "area": {"width": 100, "height": 100},
        "num_nodes": 10,
        "sensing_radius": 15,
        "comm_radius": 30,
        "weights": {"w1": 0.5, "w2": 0.3, "w3": 0.2},
        "pso_params": {"swarm_size": 10, "iterations": 20},
        "seed": 42,
    }

    res = run_pso(cfg, surrogate_model=model)
    assert res["surrogate_used"] is True
    assert res["surrogate_switch_iteration"] == 14  # 0.7 * 20 = 14
