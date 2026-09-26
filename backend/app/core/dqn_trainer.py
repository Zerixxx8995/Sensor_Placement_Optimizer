"""
core/dqn_trainer.py
-------------------
Training loop manager for DQN adaptive redeployment agent.
"""

import logging
from pathlib import Path
import numpy as np

from app.core.dqn_env import SensorNetworkEnv
from app.core.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


def train_dqn(
    num_episodes: int = 500,
    num_nodes: int = 20,
    area_w: float = 100.0,
    area_h: float = 100.0,
    sensing_radius: float = 15.0,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """
    Run full training loop for DQN agent on simulated sensor failures.
    """
    env = SensorNetworkEnv(
        num_nodes=num_nodes,
        area_w=area_w,
        area_h=area_h,
        sensing_radius=sensing_radius,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim, lr=1e-3)

    eps_start = 1.0
    eps_end = 0.05
    target_update_freq = 100
    batch_size = 64

    total_steps = 0
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        # Linear epsilon decay over num_episodes
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (episode / num_episodes))

        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(obs, action_mask=action_mask, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            total_steps += 1

            # Train step
            agent.train_step(batch_size=batch_size)

            # Target network update
            if total_steps % target_update_freq == 0:
                agent.update_target_network()

        episode_rewards.append(episode_reward)

        if episode % 50 == 0 or episode == num_episodes:
            avg_rew = float(np.mean(episode_rewards[-50:]))
            logger.info("Episode %d/%d - Epsilon: %.3f - Avg Reward: %.2f", episode, num_episodes, epsilon, avg_rew)

    metrics = {
        "episodes_trained": num_episodes,
        "final_epsilon": float(epsilon),
        "total_steps": total_steps,
        "avg_reward_last_50": float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0,
    }

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path, episodes_trained=num_episodes)
        logger.info("Saved DQN agent checkpoint to %s", checkpoint_path)

    return metrics
