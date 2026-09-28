"""
services/ppo_placement_service.py
---------------------------------
Inference service that uses trained PPO agent to place N cameras on a city grid.
Returns exact OptimizationResult shape.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from app.core.ppo_agent import PPOActorCritic
from app.core.ppo_env import CameraPlacementEnv
from app.core.ppo_trainer import CHECKPOINTS_DIR
from app.models.config import OptimizationConfig


def run_ppo_placement(config: OptimizationConfig, job_id: str = "ppo-run") -> dict:
    """Run PPO camera placement policy for given config."""
    t0 = time.time()

    area_W = float(config.area.width)
    area_H = float(config.area.height)
    num_cameras = config.num_nodes
    cell_size = config.cell_size
    sensing_radius = config.sensing_radius
    comm_radius = config.comm_radius

    cols = max(1, int(area_W / cell_size))
    rows = max(1, int(area_H / cell_size))

    # Build custom city grid from config painted cells / restricted / non-critical / intersections
    city_grid = np.zeros((rows, cols), dtype=np.int32)

    for ra in config.restricted_areas:
        c1 = int(np.clip(ra.x1 / cell_size, 0, cols - 1))
        r1 = int(np.clip(ra.y1 / cell_size, 0, rows - 1))
        c2 = int(np.clip(ra.x2 / cell_size, 0, cols))
        r2 = int(np.clip(ra.y2 / cell_size, 0, rows))
        city_grid[r1:r2, c1:c2] = 1  # Building cell

    for nca in config.non_critical_areas:
        c1 = int(np.clip(nca.x1 / cell_size, 0, cols - 1))
        r1 = int(np.clip(nca.y1 / cell_size, 0, rows - 1))
        c2 = int(np.clip(nca.x2 / cell_size, 0, cols))
        r2 = int(np.clip(nca.y2 / cell_size, 0, rows))
        city_grid[r1:r2, c1:c2] = 2  # Low-priority cell

    if hasattr(config, "intersections") and config.intersections:
        for it in config.intersections:
            c1 = int(np.clip(it.x1 / cell_size, 0, cols - 1))
            r1 = int(np.clip(it.y1 / cell_size, 0, rows - 1))
            c2 = int(np.clip(it.x2 / cell_size, 0, cols))
            r2 = int(np.clip(it.y2 / cell_size, 0, rows))
            city_grid[r1:r2, c1:c2] = 3  # Intersection cell

    env = CameraPlacementEnv(
        num_cameras=num_cameras,
        grid_dim=(rows, cols),
        area_dim=(area_W, area_H),
        sensing_radius=sensing_radius,
        comm_radius=comm_radius,
        city_grid=city_grid,
    )

    obs, _ = env.reset()

    # Load trained model if checkpoint exists
    model = PPOActorCritic(obs_dim=env.obs_dim, action_dim=2)
    ckpt_path = CHECKPOINTS_DIR / "ppo_model.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
        except Exception:
            pass
    model.eval()

    rewards_history = []

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model._get_action_dist(obs_t)
            # Generate policy action candidates
            candidates = [dist.mean.squeeze(0).numpy()]
            for _ in range(20):
                c_act = dist.sample().squeeze(0).numpy()
                c_act = np.clip(c_act, 0.0, 1.0)
                candidates.append(c_act)

        # Pick policy candidate that maximizes incremental coverage
        best_candidate = candidates[0]
        best_score = -float("inf")

        for c_act in candidates:
            wx, wy = env._snap_to_valid_road(float(c_act[0]), float(c_act[1]))
            temp_cams = env.placed_cameras + [(wx, wy)]
            cov_score = env._eval_temp_coverage(temp_cams)

            dist_penalty = 0.0
            if env.placed_cameras:
                min_dist = min(
                    np.hypot(wx - cx, wy - cy) for cx, cy in env.placed_cameras
                )
                if min_dist < 0.4 * env.sensing_radius:
                    dist_penalty = 0.15

            score = cov_score - dist_penalty

            if score > best_score:
                best_score = score
                best_candidate = c_act

        obs, reward, terminated, truncated, info = env.step(best_candidate)
        rewards_history.append(float(reward))

        if terminated or truncated:
            break

    t1 = time.time()

    best_positions = info.get("placed_positions", [])
    coverage_ratio = float(info.get("coverage", 0.0))
    connectivity_ratio = float(info.get("connectivity", 0.0))
    coverage_map = env.current_coverage_map.tolist()

    return {
        "job_id": job_id,
        "status": "complete",
        "best_positions": [[float(p[0]), float(p[1])] for p in best_positions],
        "coverage_ratio": coverage_ratio,
        "connectivity_ratio": connectivity_ratio,
        "avg_energy": 1.0,
        "fitness_history": rewards_history,
        "coverage_map": coverage_map,
        "compute_time_seconds": float(t1 - t0),
        "iterations_run": num_cameras,
        "gpu_used": False,
        "surrogate_used": False,
        "strategy": "rl",
    }
