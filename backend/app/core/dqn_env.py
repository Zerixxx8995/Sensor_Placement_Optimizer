"""
core/dqn_env.py
---------------
Custom Gymnasium environment for adaptive sensor network redeployment.
Simulates node failures and models node movements to recover network coverage.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from app.core.sensing_model import coverage_map as compute_coverage_map


class SensorNetworkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_nodes: int = 20,
        area_w: float = 100.0,
        area_h: float = 100.0,
        sensing_radius: float = 15.0,
        grid_dim: int = 10,  # 10x10 grid for coverage observation & candidate moves
        max_steps: int = 50,
        initial_positions: np.ndarray | None = None,
        dead_mask: np.ndarray | None = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.area_w = area_w
        self.area_h = area_h
        self.sensing_radius = sensing_radius
        self.grid_dim = grid_dim
        self.num_candidates = grid_dim * grid_dim
        self.max_steps = max_steps

        self.cell_w = area_w / grid_dim
        self.cell_h = area_h / grid_dim

        # Action space: Discrete(num_nodes * num_candidates)
        self.action_space = spaces.Discrete(num_nodes * self.num_candidates)

        # Observation space:
        # - flattened coverage map: (grid_dim * grid_dim,)
        # - node energy levels: (num_nodes,)
        # - node alive mask (1=alive, 0=dead): (num_nodes,)
        obs_dim = self.num_candidates + self.num_nodes + self.num_nodes
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._initial_positions_param = initial_positions
        self._initial_dead_mask_param = dead_mask

        self.positions = None
        self.energies = None
        self.alive_mask = None
        self.current_step_count = 0
        self.prev_coverage = 0.0

    def get_action_mask(self) -> np.ndarray:

        """Return boolean mask of shape (action_space.n,) where True = valid (alive node)."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        for n in range(self.num_nodes):
            if self.alive_mask[n] == 1:
                mask[n * self.num_candidates : (n + 1) * self.num_candidates] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        alive_positions = self.positions[self.alive_mask == 1]
        if len(alive_positions) > 0:
            cov_grid = compute_coverage_map(
                alive_positions,
                self.area_w,
                self.area_h,
                self.sensing_radius,
                cell_size=self.cell_w,
            )
            # Ensure grid matches grid_dim x grid_dim
            if cov_grid.shape != (self.grid_dim, self.grid_dim):
                from scipy.ndimage import zoom
                # fallback simple resize if shape slightly differs
                cov_flat = np.resize(cov_grid, (self.grid_dim, self.grid_dim)).flatten()
            else:
                cov_flat = cov_grid.flatten()
        else:
            cov_flat = np.zeros(self.num_candidates, dtype=np.float32)

        cov_flat = np.clip(cov_flat, 0.0, 1.0).astype(np.float32)
        energies = np.clip(self.energies, 0.0, 1.0).astype(np.float32)
        alive = self.alive_mask.astype(np.float32)

        return np.concatenate([cov_flat, energies, alive])

    def _compute_coverage(self) -> float:
        alive_positions = self.positions[self.alive_mask == 1]
        if len(alive_positions) == 0:
            return 0.0
        cov_grid = compute_coverage_map(
            alive_positions,
            self.area_w,
            self.area_h,
            self.sensing_radius,
            cell_size=self.cell_w,
        )
        return float(np.mean(cov_grid))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        if self._initial_positions_param is not None:
            self.positions = np.array(self._initial_positions_param, dtype=np.float32).copy()
            self.num_nodes = len(self.positions)
        else:
            self.positions = rng.uniform(
                low=[0.0, 0.0], high=[self.area_w, self.area_h], size=(self.num_nodes, 2)
            ).astype(np.float32)

        self.energies = np.ones(self.num_nodes, dtype=np.float32)

        if self._initial_dead_mask_param is not None:
            self.alive_mask = np.ones(self.num_nodes, dtype=np.int32)
            for idx in self._initial_dead_mask_param:
                if 0 <= idx < self.num_nodes:
                    self.alive_mask[idx] = 0
        else:
            # Kill K% of nodes (K ~ Uniform(10, 30))
            k_pct = rng.uniform(0.10, 0.30)
            k_dead = max(1, int(self.num_nodes * k_pct))
            dead_indices = rng.choice(self.num_nodes, size=k_dead, replace=False)
            self.alive_mask = np.ones(self.num_nodes, dtype=np.int32)
            self.alive_mask[dead_indices] = 0

        # Update action & observation space if num_nodes changed
        self.action_space = spaces.Discrete(self.num_nodes * self.num_candidates)
        obs_dim = self.num_candidates + self.num_nodes + self.num_nodes
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.current_step_count = 0
        self.prev_coverage = self._compute_coverage()

        return self._get_obs(), {"coverage": self.prev_coverage}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step_count += 1

        node_idx = action // self.num_candidates
        grid_idx = action % self.num_candidates

        reward = 0.0
        # If action is on a dead node, penalize heavily
        if node_idx >= self.num_nodes or self.alive_mask[node_idx] == 0:
            reward = -2.0
            curr_cov = self.prev_coverage
        else:
            # Move node_idx to target grid cell center
            col = grid_idx % self.grid_dim
            row = grid_idx // self.grid_dim
            target_x = (col + 0.5) * self.cell_w
            target_y = (row + 0.5) * self.cell_h

            # Deduct energy for movement
            old_pos = self.positions[node_idx].copy()
            move_dist = np.sqrt(np.sum((np.array([target_x, target_y]) - old_pos) ** 2))
            max_dist = np.sqrt(self.area_w**2 + self.area_h**2)
            self.energies[node_idx] = max(0.0, self.energies[node_idx] - 0.05 * (move_dist / max_dist))

            self.positions[node_idx] = [target_x, target_y]

            curr_cov = self._compute_coverage()
            cov_delta = curr_cov - self.prev_coverage

            reward += cov_delta
            reward -= 0.1  # movement cost

            if curr_cov >= 0.90:
                reward += 5.0
            elif curr_cov < 0.70:
                reward -= 5.0

            self.prev_coverage = curr_cov

        terminated = bool(curr_cov >= 0.95)
        truncated = bool(self.current_step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "coverage": curr_cov,
            "moved_node": node_idx,
            "step": self.current_step_count,
        }

        return obs, float(reward), terminated, truncated, info
