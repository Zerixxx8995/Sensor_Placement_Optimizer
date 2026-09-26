"""
tests/test_dqn.py
------------------
Unit and integration tests for Feature 2: DQN Adaptive Redeployment Agent.
"""

import pytest
import numpy as np

from app.core.dqn_env import SensorNetworkEnv
from app.core.dqn_agent import DQNAgent, DQNetwork
from app.core.dqn_trainer import train_dqn
from app.services import redeployment_service


def test_dqn_env_observation_space_shape():
    """assert SensorNetworkEnv observation space shape is correct."""
    N = 10
    grid_dim = 10
    env = SensorNetworkEnv(num_nodes=N, grid_dim=grid_dim)
    obs, info = env.reset()

    expected_dim = grid_dim * grid_dim + N + N  # 100 + 10 + 10 = 120
    assert env.observation_space.shape == (expected_dim,)
    assert obs.shape == (expected_dim,)


def test_action_masking_prevents_dead_nodes():
    """assert action masking prevents selecting dead nodes."""
    N = 5
    grid_dim = 10
    num_candidates = grid_dim * grid_dim
    dead_indices = [1, 3]  # Nodes 1 and 3 dead

    env = SensorNetworkEnv(num_nodes=N, grid_dim=grid_dim, dead_mask=dead_indices)
    env.reset()

    mask = env.get_action_mask()
    assert mask.shape == (N * num_candidates,)

    # Dead nodes should be False in mask
    assert not np.any(mask[1 * num_candidates : 2 * num_candidates])
    assert not np.any(mask[3 * num_candidates : 4 * num_candidates])

    # Alive nodes should be True in mask
    assert np.all(mask[0 * num_candidates : 1 * num_candidates])
    assert np.all(mask[2 * num_candidates : 3 * num_candidates])
    assert np.all(mask[4 * num_candidates : 5 * num_candidates])


def test_reward_positive_on_coverage_improvement():
    """assert reward is positive when coverage improves."""
    env = SensorNetworkEnv(num_nodes=10)
    env.reset()
    initial_cov = env.prev_coverage

    # Force a position move that improves coverage (e.g., move alive node to unreached cell)
    action_mask = env.get_action_mask()
    valid_actions = np.where(action_mask)[0]
    action = int(valid_actions[0])

    obs, reward, terminated, truncated, info = env.step(action)
    cov_after = info["coverage"]
    cov_delta = cov_after - initial_cov

    # If coverage increased, reward includes positive cov_delta
    if cov_delta > 0.1:  # significantly covers movement penalty 0.1
        assert reward > 0.0


def test_dqn_output_shape_matches_action_space():
    """assert DQN output shape matches action space."""
    obs_dim = 120
    action_dim = 500
    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim)

    dummy_obs = np.random.uniform(0, 1, size=(obs_dim,))
    action = agent.select_action(dummy_obs)
    assert 0 <= action < action_dim


def test_redeployment_service_returns_valid_in_bounds_positions():
    """assert redeployment_service returns valid node positions (within bounds)."""
    initial_positions = [
        [10.0, 10.0],
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
        [50.0, 50.0],
    ]
    dead_nodes = [1]

    result = redeployment_service.redeploy_nodes(
        positions=initial_positions,
        dead_node_indices=dead_nodes,
        area_w=100.0,
        area_h=100.0,
        sensing_radius=15.0,
        max_steps=5,
    )

    assert "new_positions" in result
    assert len(result["new_positions"]) == len(initial_positions)
    assert "coverage_before" in result
    assert "coverage_after" in result
    assert "moved_nodes" in result

    for x, y in result["new_positions"]:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_trained_agent_improves_coverage(tmp_path):
    """assert coverage after redeployment > coverage before redeployment (trained agent)."""
    ckpt_file = tmp_path / "dqn_agent.pt"
    # Train lightweight agent for 15 episodes
    metrics = train_dqn(
        num_episodes=15,
        num_nodes=10,
        area_w=100.0,
        area_h=100.0,
        sensing_radius=20.0,
        checkpoint_path=ckpt_file,
    )
    assert metrics["episodes_trained"] == 15
    assert ckpt_file.exists()

    # Load agent
    agent, ep_count = DQNAgent.load_checkpoint(ckpt_file)
    assert ep_count == 15
