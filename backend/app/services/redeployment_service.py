"""
services/redeployment_service.py
---------------------------------
Service layer for DQN adaptive node redeployment.
- Manages DQN checkpoint lifecycle
- Runs inference given current network state and failure list
- Orchestrates background training
"""

import logging
from pathlib import Path
import asyncio
import numpy as np

from app.core.dqn_agent import DQNAgent
from app.core.dqn_env import SensorNetworkEnv
from app.core.dqn_trainer import train_dqn
from app.core.sensing_model import coverage_map as compute_coverage_map

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "dqn_agent.pt"


def is_agent_trained() -> bool:
    return CHECKPOINT_PATH.exists()


def get_dqn_status() -> dict:
    """Return status: { agent_trained: bool, episodes_trained: int }."""
    if not is_agent_trained():
        return {"agent_trained": False, "episodes_trained": 0}

    try:
        _, episodes = DQNAgent.load_checkpoint(CHECKPOINT_PATH)
        return {"agent_trained": True, "episodes_trained": episodes}
    except Exception as e:
        logger.warning("Failed to load DQN status: %s", e)
        return {"agent_trained": False, "episodes_trained": 0}


def redeploy_nodes(
    positions: list[list[float]],
    dead_node_indices: list[int],
    area_w: float = 100.0,
    area_h: float = 100.0,
    sensing_radius: float = 15.0,
    max_steps: int = 15,
) -> dict:
    """
    Given current positions and dead node indices, use DQN agent (or heuristic)
    to reposition surviving nodes to recover coverage.

    Returns:
    {
        "new_positions": [[x,y], ...],
        "coverage_before": float,
        "coverage_after": float,
        "moved_nodes": [int, ...],
        "agent_used": bool
    }
    """
    pos_arr = np.array(positions, dtype=np.float32)
    N = len(pos_arr)

    env = SensorNetworkEnv(
        num_nodes=N,
        area_w=area_w,
        area_h=area_h,
        sensing_radius=sensing_radius,
        max_steps=max_steps,
        initial_positions=pos_arr,
        dead_mask=dead_node_indices,
    )

    obs, info = env.reset()
    cov_before = info["coverage"]

    agent_used = False
    agent = None
    if is_agent_trained():
        try:
            agent, _ = DQNAgent.load_checkpoint(CHECKPOINT_PATH)
            agent_used = True
        except Exception as e:
            logger.warning("Failed to load trained DQN agent: %s", e)

    moved_nodes = set()

    for _ in range(max_steps):
        action_mask = env.get_action_mask()
        if agent is not None and agent.obs_dim == obs.shape[0] and agent.action_dim == env.action_space.n:
            action = agent.select_action(obs, action_mask=action_mask, epsilon=0.0)
        else:
            # Fallback heuristic: pick random valid action
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                break
            action = int(np.random.choice(valid_actions))

        obs, reward, terminated, truncated, step_info = env.step(action)
        if step_info.get("moved_node") is not None:
            moved_nodes.add(step_info["moved_node"])

        if terminated or truncated:
            break

    cov_after = env._compute_coverage()

    return {
        "new_positions": env.positions.tolist(),
        "coverage_before": float(cov_before),
        "coverage_after": float(cov_after),
        "moved_nodes": list(moved_nodes),
        "agent_used": agent_used,
    }


async def run_dqn_training_pipeline(episodes: int = 500) -> dict:
    """Async wrapper to train DQN agent in background executor."""
    loop = asyncio.get_running_loop()

    def _train():
        return train_dqn(
            num_episodes=episodes,
            checkpoint_path=CHECKPOINT_PATH,
        )

    metrics = await loop.run_in_executor(None, _train)
    logger.info("DQN training completed: %s", metrics)
    return metrics
