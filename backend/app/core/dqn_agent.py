"""
core/dqn_agent.py
-----------------
Deep Q-Network (DQN) agent with Experience Replay, Target Network, and Epsilon-Greedy exploration.
"""

import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, buffer_capacity: int = 10000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy_net = DQNetwork(obs_dim, action_dim)
        self.target_net = DQNetwork(obs_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray | None = None, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection with action masking."""
        if random.random() < epsilon:
            if action_mask is not None and np.any(action_mask):
                valid_actions = np.where(action_mask)[0]
                return int(random.choice(valid_actions))
            return random.randint(0, self.action_dim - 1)

        self.policy_net.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(obs_tensor).squeeze(0).numpy()

        if action_mask is not None and np.any(action_mask):
            masked_q = q_values.copy()
            masked_q[~action_mask] = -1e9
            return int(np.argmax(masked_q))

        return int(np.argmax(q_values))

    def train_step(self, batch_size: int = 64, gamma: float = 0.99) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None

        self.policy_net.train()
        obs, action, reward, next_obs, done = self.replay_buffer.sample(batch_size)

        obs_t = torch.tensor(obs, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward_t = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
        done_t = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_eval = self.policy_net(obs_t).gather(1, action_t)

        with torch.no_grad():
            q_next = self.target_net(next_obs_t).max(1, keepdim=True)[0]
            target_q = reward_t + (1.0 - done_t) * gamma * q_next

        loss = self.criterion(q_eval, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: str | Path, episodes_trained: int = 0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "episodes_trained": episodes_trained,
        }, str(path))

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> tuple["DQNAgent", int]:
        path = Path(path)
        checkpoint = torch.load(str(path), weights_only=False)
        agent = cls(obs_dim=checkpoint["obs_dim"], action_dim=checkpoint["action_dim"])
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent, checkpoint.get("episodes_trained", 0)
