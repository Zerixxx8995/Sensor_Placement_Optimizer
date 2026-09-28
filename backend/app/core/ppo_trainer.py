"""
core/ppo_trainer.py
-------------------
Full PPO training loop with RolloutBuffer and GAE (Generalized Advantage Estimation).
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.core.ppo_agent import PPOActorCritic
from app.core.ppo_env import CameraPlacementEnv

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


class RolloutBuffer:
    """Storage buffer for transitions collected during PPO rollouts."""

    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE) returns and advantages."""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_terminal * last_gae

        returns = advantages + values[:n]
        return returns, advantages


class PPOTrainer:
    """PPO Training manager."""

    def __init__(
        self,
        env: CameraPlacementEnv | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
    ):
        self.env = env or CameraPlacementEnv()
        self.obs_dim = self.env.obs_dim
        self.model = PPOActorCritic(obs_dim=self.obs_dim, action_dim=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.buffer = RolloutBuffer()
        self.reward_history: list[float] = []
        self.coverage_history: list[float] = []
        self.best_coverage: float = 0.0
        self.episodes_trained: int = 0

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = CHECKPOINTS_DIR / "ppo_model.pt"

        if self.checkpoint_path.exists():
            try:
                self.load_checkpoint()
            except Exception:
                pass

    def save_checkpoint(self):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "episodes_trained": self.episodes_trained,
                "best_coverage": self.best_coverage,
                "reward_history": self.reward_history,
                "coverage_history": self.coverage_history,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self):
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.episodes_trained = ckpt.get("episodes_trained", 0)
            self.best_coverage = ckpt.get("best_coverage", 0.0)
            self.reward_history = ckpt.get("reward_history", [])
            self.coverage_history = ckpt.get("coverage_history", [])

    def update_ppo(self, returns: np.ndarray, advantages: np.ndarray):
        """PPO mini-batch gradient updates."""
        obs_tensor = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32)
        act_tensor = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        dataset_size = len(self.buffer.observations)

        for _ in range(self.n_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                mb_obs = obs_tensor[mb_indices]
                mb_act = act_tensor[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]

                _, new_log_probs, entropy, values = self.model.get_action_and_value(
                    mb_obs, mb_act
                )

                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                total_loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def train_episodes(
        self, episodes: int = 50, progress_callback=None
    ) -> dict:
        """Run training loop for specified number of episodes."""
        self.model.train()

        for ep in range(episodes):
            # Randomize camera count per episode so policy generalizes to any camera budget
            self.env.num_cameras = int(np.random.choice([20, 30, 50, 80, 100]))
            obs, _ = self.env.reset()
            ep_reward = 0.0
            self.buffer.clear()

            while True:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, _, val = self.model.get_action_and_value(obs_t)

                act_np = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(act_np)

                self.buffer.observations.append(obs)
                self.buffer.actions.append(act_np)
                self.buffer.rewards.append(reward)
                self.buffer.values.append(val.item())
                self.buffer.log_probs.append(log_prob.item())
                self.buffer.dones.append(terminated or truncated)

                obs = next_obs
                ep_reward += reward

                if terminated or truncated:
                    break

            # GAE computation
            last_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, last_val = self.model(last_obs_t)
            returns, advantages = self.buffer.compute_gae(last_val.item(), self.gamma, self.gae_lambda)

            # PPO update
            self.update_ppo(returns, advantages)

            final_cov = info.get("coverage", 0.0)
            self.episodes_trained += 1
            self.reward_history.append(ep_reward)
            self.coverage_history.append(final_cov)

            if final_cov > self.best_coverage:
                self.best_coverage = final_cov

            progress_data = {
                "episode": self.episodes_trained,
                "reward": float(ep_reward),
                "coverage": float(final_cov),
                "best_coverage": float(self.best_coverage),
            }

            if progress_callback:
                progress_callback(progress_data)

        self.save_checkpoint()
        return {
            "trained": True,
            "episodes_trained": self.episodes_trained,
            "final_reward": float(self.reward_history[-1]) if self.reward_history else 0.0,
            "best_coverage": float(self.best_coverage),
            "reward_history": self.reward_history,
            "coverage_history": self.coverage_history,
        }
