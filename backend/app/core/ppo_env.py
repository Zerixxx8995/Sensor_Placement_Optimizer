"""
core/ppo_env.py
----------------
Gymnasium environment for PPO camera placement: CameraPlacementEnv.

Agent places N cameras sequentially on a city grid to maximize road coverage,
network connectivity, and minimize deployment cost.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from app.core.city_templates import get_city_template


class CameraPlacementEnv(gym.Env):
    """
    Episode: place N cameras one at a time on the city grid.
    Agent places one camera per step until all N are placed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_cameras: int = 20,
        grid_dim: tuple[int, int] = (20, 20),
        area_dim: tuple[float, float] = (100.0, 100.0),
        sensing_radius: float = 15.0,
        comm_radius: float = 30.0,
        city_grid: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.L, self.W = grid_dim
        self.area_W, self.area_H = area_dim
        self.sensing_radius = sensing_radius
        self.comm_radius = comm_radius
        self.initial_city_grid = city_grid

        # Fixed 20x20 obs grid spatial dimension for model compatibility across any grid resolution
        self.target_grid = (20, 20)
        self.obs_dim = (20 * 20 * 4) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action: normalized (x, y) coordinates in [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._seed = seed
        self.reset(seed=seed)

    def _get_valid_road_cells(self) -> list[tuple[int, int]]:
        """Return list of (r, c) cell indices that are not building cells."""
        valid = []
        for r in range(self.L):
            for c in range(self.W):
                if self.city_grid[r, c] != 1:  # Not building cell
                    valid.append((r, c))
        if not valid:  # Fallback if map was all buildings
            valid = [(self.L // 2, self.W // 2)]
        return valid

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        if self.initial_city_grid is not None:
            self.city_grid = self.initial_city_grid.copy()
        else:
            self.city_grid = get_city_template(seed=self._seed, L=self.L, W=self.W)

        self.valid_road_cells = self._get_valid_road_cells()

        # Control center location: center road cell or (0,0)
        self.control_center_pos = (self.area_W / 2.0, self.area_H / 2.0)

        self.placed_cameras: list[tuple[float, float]] = []
        self.step_count = 0
        self.current_coverage_map = np.zeros((self.L, self.W), dtype=np.float32)
        self.current_connectivity_map = np.zeros((self.L, self.W), dtype=np.float32)
        self.camera_grid_mask = np.zeros((self.L, self.W), dtype=np.float32)

        self.prev_coverage = 0.0
        self.prev_connected_ratio = 0.0

        obs = self._get_observation()
        info = {}
        return obs, info

    def _snap_to_valid_road(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        """Convert normalized (x,y) to world coords, snapping buildings to nearest road cell."""
        wx = np.clip(x_norm * self.area_W, 0.0, self.area_W)
        wy = np.clip(y_norm * self.area_H, 0.0, self.area_H)

        cell_w = self.area_W / self.W
        cell_h = self.area_H / self.L

        c = int(np.clip(wx / cell_w, 0, self.W - 1))
        r = int(np.clip(wy / cell_h, 0, self.L - 1))

        # If it's a building cell (1), snap to nearest valid road cell
        if self.city_grid[r, c] == 1:
            best_dist = float("inf")
            best_r, best_c = r, c
            for vr, vc in self.valid_road_cells:
                dist = (vr - r) ** 2 + (vc - c) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_r, best_c = vr, vc
            r, c = best_r, best_c
            wx = (c + 0.5) * cell_w
            wy = (r + 0.5) * cell_h

        return wx, wy

    def _compute_coverage(self) -> float:
        """Compute coverage over valid road cells (intersections have double weight)."""
        cell_w = self.area_W / self.W
        cell_h = self.area_H / self.L

        total_weight = 0.0
        covered_weight = 0.0

        for r in range(self.L):
            for c in range(self.W):
                cell_type = self.city_grid[r, c]
                if cell_type == 1 or cell_type == 2:
                    # Building (1) or Low-priority (2) -> not counted in coverage denominator
                    self.current_coverage_map[r, c] = 0.0
                    continue

                cell_x = (c + 0.5) * cell_w
                cell_y = (r + 0.5) * cell_h
                w = 2.0 if cell_type == 3 else 1.0  # Intersection cells double weight

                # Probability cell is covered by at least 1 camera
                prob_uncovered = 1.0
                for cx, cy in self.placed_cameras:
                    dist = np.sqrt((cx - cell_x) ** 2 + (cy - cell_y) ** 2)
                    if dist <= self.sensing_radius:
                        prob = 1.0 - (dist / self.sensing_radius) ** 2
                        prob_uncovered *= (1.0 - prob)

                cov = 1.0 - prob_uncovered
                self.current_coverage_map[r, c] = cov
                total_weight += w
                covered_weight += cov * w

        return covered_weight / max(1.0, total_weight)

    def _eval_temp_coverage(self, temp_cameras: list[tuple[float, float]]) -> float:
        """Evaluate coverage ratio for a candidate set of camera positions."""
        cell_w = self.area_W / self.W
        cell_h = self.area_H / self.L

        total_weight = 0.0
        covered_weight = 0.0

        for r in range(self.L):
            for c in range(self.W):
                cell_type = self.city_grid[r, c]
                if cell_type == 1 or cell_type == 2:
                    continue

                cell_x = (c + 0.5) * cell_w
                cell_y = (r + 0.5) * cell_h
                w = 2.0 if cell_type == 3 else 1.0

                prob_uncovered = 1.0
                for cx, cy in temp_cameras:
                    dist = np.sqrt((cx - cell_x) ** 2 + (cy - cell_y) ** 2)
                    if dist <= self.sensing_radius:
                        prob = 1.0 - (dist / self.sensing_radius) ** 2
                        prob_uncovered *= (1.0 - prob)

                cov = 1.0 - prob_uncovered
                total_weight += w
                covered_weight += cov * w

        return covered_weight / max(1.0, total_weight)

    def _compute_connectivity(self) -> float:
        """Compute fraction of placed cameras connected to control center."""
        if not self.placed_cameras:
            return 1.0

        n = len(self.placed_cameras)
        # Build adjacency graph
        adj = np.zeros((n + 1, n + 1), dtype=bool)

        # Node n is the control center
        cc_x, cc_y = self.control_center_pos

        for i, (x1, y1) in enumerate(self.placed_cameras):
            # Check connection to CC
            if np.hypot(x1 - cc_x, y1 - cc_y) <= self.comm_radius:
                adj[i, n] = adj[n, i] = True
            for j in range(i + 1, n):
                x2, y2 = self.placed_cameras[j]
                if np.hypot(x1 - x2, y1 - y2) <= self.comm_radius:
                    adj[i, j] = adj[j, i] = True

        # BFS from control center (node n)
        visited = np.zeros(n + 1, dtype=bool)
        queue = [n]
        visited[n] = True

        while queue:
            curr = queue.pop(0)
            for neighbor in range(n + 1):
                if adj[curr, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # Update connectivity map for obs
        self.current_connectivity_map.fill(0.0)
        cell_w = self.area_W / self.W
        cell_h = self.area_H / self.L
        for i in range(n):
            if visited[i]:
                cx, cy = self.placed_cameras[i]
                c = int(np.clip(cx / cell_w, 0, self.W - 1))
                r = int(np.clip(cy / cell_h, 0, self.L - 1))
                self.current_connectivity_map[r, c] = 1.0

        connected_cameras = np.sum(visited[:n])
        return float(connected_cameras / n)

    def _get_observation(self) -> np.ndarray:
        # Channel 0: normalized cell types (0 to 1)
        ch0 = (self.city_grid / 3.0).astype(np.float32)
        # Channel 1: current coverage map
        ch1 = self.current_coverage_map.astype(np.float32)
        # Channel 2: current connectivity map
        ch2 = self.current_connectivity_map.astype(np.float32)
        # Channel 3: cameras placed mask
        ch3 = self.camera_grid_mask.astype(np.float32)

        stack = np.stack([ch0, ch1, ch2, ch3], axis=-1)  # (L, W, 4)
        if (self.L, self.W) != (20, 20):
            import torch
            import torch.nn.functional as F

            t = torch.tensor(stack, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            resized = F.interpolate(t, size=(20, 20), mode="bilinear", align_corners=False)
            stack = resized.squeeze(0).permute(1, 2, 0).numpy()

        grid_obs = stack.flatten()
        step_ratio = np.array([self.step_count / float(self.num_cameras)], dtype=np.float32)

        return np.concatenate([grid_obs, step_ratio])

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        x_norm, y_norm = float(action[0]), float(action[1])

        # Map to valid world coordinate & snap if on building
        wx, wy = self._snap_to_valid_road(x_norm, y_norm)
        self.placed_cameras.append((wx, wy))

        # Update camera grid mask
        cell_w = self.area_W / self.W
        cell_h = self.area_H / self.L
        c = int(np.clip(wx / cell_w, 0, self.W - 1))
        r = int(np.clip(wy / cell_h, 0, self.L - 1))
        self.camera_grid_mask[r, c] = 1.0

        self.step_count += 1

        # Re-evaluate metrics
        cur_coverage = self._compute_coverage()
        cur_connectivity = self._compute_connectivity()

        cov_delta = cur_coverage - self.prev_coverage
        conn_delta = cur_connectivity - self.prev_connected_ratio

        # Cost: 1.5 at intersections (cell type 3), 1.0 elsewhere
        cost_delta = 1.5 if self.city_grid[r, c] == 3 else 1.0

        # Reward formulation
        reward = (cov_delta * 10.0) + (conn_delta * 5.0) - (cost_delta * 0.1)

        self.prev_coverage = cur_coverage
        self.prev_connected_ratio = cur_connectivity

        terminated = self.step_count >= self.num_cameras
        truncated = False

        if terminated:
            # End-of-episode terminal rewards
            if cur_coverage > 0.90:
                reward += 20.0
            elif cur_coverage > 0.80:
                reward += 10.0
            elif cur_coverage < 0.60:
                reward -= 10.0

            if cur_connectivity > 0.95:
                reward += 5.0

        obs = self._get_observation()
        info = {
            "coverage": cur_coverage,
            "connectivity": cur_connectivity,
            "placed_positions": list(self.placed_cameras),
        }

        return obs, float(reward), terminated, truncated, info
