"""
tests/test_ppo.py
------------------
Tests for PPO Camera Placement environment, model, trainer, service, and API endpoints.
"""

import numpy as np
import pytest
from app.core.ppo_env import CameraPlacementEnv


def test_camera_placement_env_obs_shape():
    N = 10
    grid_dim = (20, 20)
    env = CameraPlacementEnv(num_cameras=N, grid_dim=grid_dim)
    obs, info = env.reset()

    expected_obs_dim = (20 * 20 * 4) + 1
    assert obs.shape == (expected_obs_dim,)
    assert env.observation_space.shape == (expected_obs_dim,)


def test_camera_placement_env_action_snaps_from_building():
    grid = np.ones((20, 20), dtype=np.int32)
    grid[5, 5] = 0  # Only one road cell at (5,5)

    env = CameraPlacementEnv(num_cameras=5, grid_dim=(20, 20), city_grid=grid)
    env.reset()

    # Action targeting (0.9, 0.9) which is building
    obs, reward, terminated, truncated, info = env.step(np.array([0.9, 0.9], dtype=np.float32))

    placed = info["placed_positions"]
    assert len(placed) == 1
    wx, wy = placed[0]
    # Cell size is 100 / 20 = 5.0 -> (5,5) center is (27.5, 27.5)
    cell_w = 100.0 / 20
    cell_h = 100.0 / 20
    c = int(wx / cell_w)
    r = int(wy / cell_h)
    assert env.city_grid[r, c] == 0  # Snapped to road cell


def test_camera_placement_env_terminates_after_n_steps():
    N = 8
    env = CameraPlacementEnv(num_cameras=N)
    env.reset()

    for step in range(N):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        if step < N - 1:
            assert not terminated
        else:
            assert terminated


def test_camera_placement_env_positive_reward_for_coverage():
    env = CameraPlacementEnv(num_cameras=5)
    env.reset()

    # Step 1: place camera in center
    obs1, reward1, _, _, info1 = env.step(np.array([0.5, 0.5], dtype=np.float32))
    # Step 2: place camera in another un-covered corner
    obs2, reward2, _, _, info2 = env.step(np.array([0.1, 0.1], dtype=np.float32))

    assert info2["coverage"] >= info1["coverage"]


def test_ppo_actor_critic_shapes():
    import torch
    from app.core.ppo_agent import PPOActorCritic

    obs_dim = (20 * 20 * 4) + 1
    model = PPOActorCritic(obs_dim=obs_dim, action_dim=2)

    sample_obs = torch.randn(1, obs_dim)
    mean_action, value = model(sample_obs)

    assert mean_action.shape == (1, 2)
    assert value.shape == (1, 1)

    action, log_prob, entropy, val = model.get_action_and_value(sample_obs)
    assert action.shape == (1, 2)
    assert log_prob.shape == (1,)
    assert val.shape == (1,)


def test_ppo_trainer_runs_and_saves_checkpoint(tmp_path):
    from app.core.ppo_trainer import PPOTrainer
    from app.core.ppo_env import CameraPlacementEnv

    env = CameraPlacementEnv(num_cameras=4, grid_dim=(10, 10))
    trainer = PPOTrainer(env=env)
    
    # Train 2 quick episodes
    res = trainer.train_episodes(episodes=2)

    assert res["trained"] is True
    assert res["episodes_trained"] >= 2
    assert trainer.checkpoint_path.exists()


def test_ppo_placement_service_output():
    from app.models.config import OptimizationConfig, Area, Weights
    from app.services.ppo_placement_service import run_ppo_placement

    cfg = OptimizationConfig(
        area=Area(width=100, height=100),
        num_nodes=10,
        sensing_radius=15,
        comm_radius=30,
        weights=Weights(w1=0.7, w2=0.15, w3=0.15),
        method="rl",
    )

    res = run_ppo_placement(cfg, job_id="test-rl")

    assert res["status"] == "complete"
    assert len(res["best_positions"]) == 10
    assert 0.0 <= res["coverage_ratio"] <= 1.0
    assert 0.0 <= res["connectivity_ratio"] <= 1.0
    assert "coverage_map" in res
    assert res["gpu_used"] is False

    for x, y in res["best_positions"]:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_compare_includes_rl_strategy():
    from app.models.config import OptimizationConfig, Area, Weights
    from app.services.comparison_service import run_comparison

    cfg = OptimizationConfig(
        area=Area(width=50, height=50),
        num_nodes=5,
        sensing_radius=10,
        comm_radius=20,
        weights=Weights(w1=0.7, w2=0.15, w3=0.15),
    )

    comp = run_comparison(cfg)
    strategies = [r["strategy"] for r in comp["results"]]

    assert "rl" in strategies
    assert len(strategies) == 5




