"""
core/pso.py
-----------
Standard Particle Swarm Optimization (PSO) engine — CPU only, pure NumPy.

Implements Spatial Position Encoding (SPE): each particle IS one sensor node.
Population size = number of sensor nodes to deploy.

Velocity update:
    V[i] = ω·V[i] + c1·r1·(Pbest[i] - X[i]) + c2·r2·(Gbest - X[i])

Position update:
    X[i] = X[i] + V[i]

Returns a result dict with best_positions, fitness_history, coverage_map, and
runtime metadata. This is the only entry point other layers should call.

No I/O, no HTTP, no business logic — pure algorithm only.
"""

import time
import numpy as np

from .fitness import compute_fitness
from .sensing_model import coverage_map as compute_coverage_map


def _build_fitness_config(config: dict) -> dict:
    """
    Extract and normalise all keys required by compute_fitness from the
    top-level PSO config dict.
    """
    area = config["area"]
    area_W = float(area["width"])
    area_H = float(area["height"])
    weights = config["weights"]
    pso_params = config.get("pso_params", {})

    default_sink = (area_W / 2.0, area_H / 2.0)
    sink = tuple(config["sink"]) if config.get("sink") is not None else default_sink

    return {
        "area_W": area_W,
        "area_H": area_H,
        "Rs": float(config["sensing_radius"]),
        "Rc": float(config["comm_radius"]),
        "lam": float(config.get("lam", 0.5)),
        "cell_size": float(config.get("cell_size", 1.0)),
        "w1": float(weights["w1"]),
        "w2": float(weights["w2"]),
        "w3": float(weights["w3"]),
        "sink": sink,
        "restricted_mask": config.get("restricted_mask", None),
    }


def _build_restricted_mask(
    restricted_areas: list[dict],
    area_W: float,
    area_H: float,
    cell_size: float,
) -> np.ndarray | None:
    """
    Convert a list of restricted-area rectangles into a boolean grid mask.

    Each rect: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    Returns a (rows, cols) bool array where True = restricted.
    Returns None if restricted_areas is empty.
    """
    if not restricted_areas:
        return None

    cols = int(np.ceil(area_W / cell_size))
    rows = int(np.ceil(area_H / cell_size))
    mask = np.zeros((rows, cols), dtype=bool)

    for ra in restricted_areas:
        x1, y1 = float(ra["x1"]), float(ra["y1"])
        x2, y2 = float(ra["x2"]), float(ra["y2"])
        # Convert world coords to cell indices
        c1 = max(0, int(x1 / cell_size))
        r1 = max(0, int(y1 / cell_size))
        c2 = min(cols, int(np.ceil(x2 / cell_size)))
        r2 = min(rows, int(np.ceil(y2 / cell_size)))
        mask[r1:r2, c1:c2] = True

    return mask


def _match_nodes(X: np.ndarray, Target: np.ndarray) -> np.ndarray:
    """
    Match each node in X (N, 2) to the nearest unassigned node in Target (N, 2)
    using greedy spatial distance matching to prevent node permutation crossing.
    """
    N = X.shape[0]
    matched = np.zeros_like(X)
    available_mask = np.ones(N, dtype=bool)

    dists = np.sum((X[:, np.newaxis, :] - Target[np.newaxis, :, :]) ** 2, axis=2)

    for i in range(N):
        dists_i = dists[i].copy()
        dists_i[~available_mask] = np.inf
        best_j = int(np.argmin(dists_i))
        matched[i] = Target[best_j]
        available_mask[best_j] = False

    return matched


def run_pso(config: dict, on_iteration=None, surrogate_model=None) -> dict:
    """
    Run the PSO optimization and return the best sensor deployment found.
    """
    # --- Unpack config ---
    area = config["area"]
    area_W = float(area["width"])
    area_H = float(area["height"])
    N = int(config["num_nodes"])          # particles = sensor nodes
    Rs = float(config["sensing_radius"])
    Rc = float(config["comm_radius"])
    cell_size = float(config.get("cell_size", 1.0))
    seed = config.get("seed", None)

    pso_params = config.get("pso_params", {})
    P = int(pso_params.get("swarm_size", 30))   # swarm size (independent restarts)
    G = int(pso_params.get("iterations", 500))
    omega = float(pso_params.get("inertia", 0.7))
    c1 = float(pso_params.get("c1", 1.5))
    c2 = float(pso_params.get("c2", 1.5))

    restricted_areas = config.get("restricted_areas", [])
    restricted_mask = _build_restricted_mask(restricted_areas, area_W, area_H, cell_size)

    # Build fitness config once (shared across all evaluations)
    fitness_cfg = _build_fitness_config(config)
    fitness_cfg["restricted_mask"] = restricted_mask

    surrogate_used = surrogate_model is not None
    switch_iter = int(0.7 * G) if surrogate_used else None

    # --- Seed ---
    rng = np.random.default_rng(seed)

    # --- Initialise swarm ---
    positions = np.zeros((P, N, 2), dtype=np.float64)
    grid_cols = int(np.ceil(np.sqrt(N * (area_W / area_H))))
    grid_rows = int(np.ceil(N / max(1, grid_cols)))
    grid_xs = np.linspace(area_W * 0.05, area_W * 0.95, max(1, grid_cols))
    grid_ys = np.linspace(area_H * 0.05, area_H * 0.95, max(1, grid_rows))
    gx, gy = np.meshgrid(grid_xs, grid_ys)
    base_grid = np.column_stack([gx.ravel(), gy.ravel()])[:N]

    if len(base_grid) < N:
        extra = rng.uniform([0.0, 0.0], [area_W, area_H], size=(N - len(base_grid), 2))
        base_grid = np.vstack([base_grid, extra])

    for i in range(P):
        if i < P // 2:
            jitter = rng.normal(0.0, max(area_W, area_H) * 0.05, size=(N, 2))
            positions[i] = np.clip(base_grid + jitter, [0.0, 0.0], [area_W, area_H])
        else:
            positions[i] = rng.uniform(low=[0.0, 0.0], high=[area_W, area_H], size=(N, 2))

    v_max = np.array([area_W, area_H]) * 0.1
    velocities = rng.uniform(low=-v_max, high=v_max, size=(P, N, 2))

    # --- Evaluate initial fitness ---
    if surrogate_used:
        from app.core.surrogate_model import predict_surrogate_batch
        fitnesses = predict_surrogate_batch(surrogate_model, positions, config)
    else:
        fitnesses = np.array([
            compute_fitness(positions[i], fitness_cfg, iteration=0, max_iterations=G)
            for i in range(P)
        ])

    # Personal bests
    pbest_pos = positions.copy()
    pbest_fit = fitnesses.copy()

    # Global best
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    fitness_history = [gbest_fit]

    # --- Main PSO loop ---
    t_start = time.perf_counter()

    for g in range(1, G + 1):
        omega_g = omega * (1.0 - 0.5 * (g / G))

        r1 = rng.uniform(0.0, 1.0, size=(P, N, 2))
        r2 = rng.uniform(0.0, 1.0, size=(P, N, 2))

        # Spatial nearest-neighbor matching for cognitive & social targets
        pbest_matched = np.array([_match_nodes(positions[i], pbest_pos[i]) for i in range(P)])
        gbest_matched = np.array([_match_nodes(positions[i], gbest_pos) for i in range(P)])

        # Velocity update
        velocities = (
            omega_g * velocities
            + c1 * r1 * (pbest_matched - positions)
            + c2 * r2 * (gbest_matched - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update & boundary enforcement
        positions = positions + velocities

        # Clamp positions strictly to deployment field bounds
        oob_low_x = positions[:, :, 0] < 0.0
        oob_high_x = positions[:, :, 0] > area_W
        oob_low_y = positions[:, :, 1] < 0.0
        oob_high_y = positions[:, :, 1] > area_H

        positions[:, :, 0] = np.clip(positions[:, :, 0], 0.0, area_W)
        positions[:, :, 1] = np.clip(positions[:, :, 1], 0.0, area_H)

        # Reflect velocities for nodes that hit boundaries
        velocities[:, :, 0][oob_low_x | oob_high_x] *= -0.5
        velocities[:, :, 1][oob_low_y | oob_high_y] *= -0.5

        # Evaluate fitness for all particles
        if surrogate_used and g <= switch_iter:
            from app.core.surrogate_model import predict_surrogate_batch
            fitnesses = predict_surrogate_batch(surrogate_model, positions, config)
        else:
            fitnesses = np.array([
                compute_fitness(positions[i], fitness_cfg, iteration=g, max_iterations=G)
                for i in range(P)
            ])

        # Update personal bests
        improved = fitnesses < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitnesses[improved]

        # Update global best
        current_best_idx = int(np.argmin(pbest_fit))
        if pbest_fit[current_best_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[current_best_idx])
            gbest_pos = pbest_pos[current_best_idx].copy()

        fitness_history.append(gbest_fit)

        if on_iteration is not None:
            clamped_positions = np.clip(positions, [0.0, 0.0], [area_W, area_H])
            clamped_gbest_pos = np.clip(gbest_pos, [0.0, 0.0], [area_W, area_H])
            on_iteration(g, clamped_positions, clamped_gbest_pos, gbest_fit)
            time.sleep(max(0.002, min(0.01, 3.0 / G)))

    compute_time = time.perf_counter() - t_start

    # --- Final metrics on best positions (clamped to field) ---
    best_clamped = np.clip(gbest_pos, [0.0, 0.0], [area_W, area_H])

    final_cov_map = compute_coverage_map(
        best_clamped, area_W, area_H, Rs,
        lam=fitness_cfg.get("lam", 0.5),
        cell_size=cell_size,
        restricted_mask=restricted_mask,
    )

    if restricted_mask is not None:
        valid = ~restricted_mask
        coverage_ratio = float(np.mean(final_cov_map[valid])) if valid.any() else 0.0
    else:
        coverage_ratio = float(np.mean(final_cov_map))

    from .fitness import _connectivity_ratio, _energy_cost
    connectivity_ratio = _connectivity_ratio(
        best_clamped, Rc, sink=fitness_cfg.get("sink", (0.0, 0.0))
    )
    avg_energy = _energy_cost(best_clamped, area_W, area_H)

    return {
        "best_positions": best_clamped,
        "fitness_history": fitness_history,
        "coverage_map": final_cov_map,
        "coverage_ratio": coverage_ratio,
        "connectivity_ratio": connectivity_ratio,
        "avg_energy": avg_energy,
        "compute_time_seconds": compute_time,
        "iterations_run": G,
        "gpu_used": False,
        "surrogate_used": surrogate_used,
        "surrogate_switch_iteration": switch_iter,
    }

