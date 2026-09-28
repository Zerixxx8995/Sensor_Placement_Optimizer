"""
core/ppo_agent.py
-----------------
PyTorch PPO Actor-Critic Neural Network Model.

Shared backbone -> Actor Head (action mean + log_std) & Critic Head (state value).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class PPOActorCritic(nn.Module):
    """Actor-Critic network for continuous action space (x, y) camera placement."""

    def __init__(self, obs_dim: int, action_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Outputs in [-1, 1], remapped to [0, 1]
        )

        # Log std parameter for Gaussian policy
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _get_action_dist(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.backbone(obs)
        tanh_mean = self.actor_head(features)
        # Remap Tanh [-1, 1] -> [0, 1]
        action_mean = 0.5 * (tanh_mean + 1.0)

        log_std = torch.clamp(self.actor_log_std, -2.0, 0.5)
        std = torch.exp(log_std)

        dist = Normal(action_mean, std)
        return dist, features

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, features = self._get_action_dist(obs)
        value = self.critic_head(features)
        return dist.mean, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, features = self._get_action_dist(obs)
        value = self.critic_head(features)

        if action is None:
            raw_action = dist.sample()
            action = torch.clamp(raw_action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return action, log_prob, entropy, value.squeeze(-1)
