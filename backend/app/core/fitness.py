"""
core/fitness.py
---------------
Multi-objective fitness function for sensor placement.

Fitness formula:
    F(X) = w1·(1 - Coverage) + w2·EnergyAvg + w3·(1 - ConnectivityRatio)

Where:
  - Coverage:          Mean detection probability across all non-restricted cells.
  - EnergyAvg:         Average normalized energy proxy (distance to sink / max_dist).
  - ConnectivityRatio: Fraction of nodes that can reach the base sink within Rc
                       (directly or through a multi-hop connected component).
  - Adaptive penalty:  OOB / overlap penalty coefficient α decays 0.5 → 0
                       over the first half of total iterations.

No I/O, no HTTP, no business logic — pure math only.
"""

import numpy as np
from .sensing_model import coverage_map


def _connectivity_ratio(
    positions: np.ndarray,
    Rc: float,
    sink: tuple[float, float] | None = None,
) -> float:
    """
    Compute the fraction of nodes that are in the same connected component as
    the base sink using a simple BFS/union-find over the communication graph.

    Two nodes are connected if their Euclidean distance <= Rc.
    The sink (base station) is treated as a virtual node at `sink` position.
    If sink is None, defaults to (0, 0).

    Args:
        positions: Array of shape (N, 2).
        Rc:        Communication radius (meters).
        sink:      (x, y) position of the base sink. Default (0, 0).

    Returns:
        Connectivity ratio in [0.0, 1.0].
    """
    if len(positions) == 0:
        return 0.0

    sink_pos = np.array(sink if sink is not None else (0.0, 0.0))
    N = len(positions)

    # Augment positions with sink as node index N
    all_pos = np.vstack([positions, sink_pos[np.newaxis, :]])  # (N+1, 2)

    # BFS from sink (index N)
    visited = np.zeros(N + 1, dtype=bool)
    queue = [N]
    visited[N] = True

    while queue:
        curr = queue.pop(0)
        curr_pos = all_pos[curr]
        dists = np.sqrt(np.sum((all_pos - curr_pos) ** 2, axis=1))
        neighbors = np.where((dists <= Rc) & (~visited))[0]
        for nb in neighbors:
            visited[nb] = True
            queue.append(int(nb))

    # Count how many sensor nodes (not the sink) are reachable
    connected_sensors = int(np.sum(visited[:N]))
    return connected_sensors / N


def _energy_cost(
    positions: np.ndarray,
    area_W: float,
    area_H: float,
) -> float:
    """
    Proxy for average energy consumption: normalized mean distance from each
    node to the area centroid (center of the field). Lower distance = less
    transmission cost assumed.

    Returns average value in [0.0, 1.0].
    """
    if len(positions) == 0:
        return 1.0
    centroid = np.array([area_W / 2.0, area_H / 2.0])
    max_dist = np.sqrt((area_W / 2.0) ** 2 + (area_H / 2.0) ** 2)
    if max_dist == 0:
        return 0.0
    dists = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
    return float(np.mean(dists) / max_dist)


def _adaptive_penalty_alpha(iteration: int, max_iterations: int) -> float:
    """
    Adaptive penalty coefficient that decays from 0.5 → 0 over the first
    half of total iterations, then stays at 0.

    Args:
        iteration:      Current iteration (0-indexed).
        max_iterations: Total number of iterations.

    Returns:
        Penalty coefficient α in [0.0, 0.5].
    """
    half = max_iterations / 2.0
    if iteration >= half:
        return 0.0
    return 0.5 * (1.0 - iteration / half)


def _oob_penalty(
    positions: np.ndarray,
    area_W: float,
    area_H: float,
) -> float:
    """
    Out-of-bounds penalty: mean normalized distance that each out-of-bounds
    node exceeds the field boundary.

    Returns 0.0 if all nodes are in bounds.
    """
    if len(positions) == 0:
        return 0.0

    xs, ys = positions[:, 0], positions[:, 1]
    max_dim = max(area_W, area_H)

    x_viol = np.maximum(0.0, -xs) + np.maximum(0.0, xs - area_W)
    y_viol = np.maximum(0.0, -ys) + np.maximum(0.0, ys - area_H)
    total_viol = np.mean((x_viol + y_viol) / max_dim)
    return float(total_viol)


def compute_fitness(
    positions: np.ndarray,
    config: dict,
    iteration: int = 0,
    max_iterations: int = 1,
) -> float:
    """
    Compute the multi-objective fitness for a given sensor deployment.

    A lower fitness value is better (minimization problem).

    Args:
        positions:       Array of shape (N, 2) — sensor (x, y) positions.
        config:          Configuration dict with keys:
                           area_W, area_H  — field dimensions
                           Rs              — sensing radius
                           Rc              — comm radius
                           lam             — sensing falloff coefficient (default 0.5)
                           cell_size       — grid resolution (default 1.0)
                           w1, w2, w3      — objective weights (must sum to 1)
                           sink            — (x, y) base sink position (default (0,0))
                           restricted_mask — optional 2D bool np.ndarray
        iteration:       Current PSO iteration (for adaptive penalty).
        max_iterations:  Total PSO iterations (for adaptive penalty).

    Returns:
        Scalar fitness value (lower = better).
    """
    area_W = float(config["area_W"])
    area_H = float(config["area_H"])
    Rs = float(config["Rs"])
    Rc = float(config["Rc"])
    lam = float(config.get("lam", 0.5))
    cell_size = float(config.get("cell_size", 1.0))
    w1 = float(config["w1"])
    w2 = float(config["w2"])
    w3 = float(config["w3"])
    sink = config.get("sink") if config.get("sink") is not None else (area_W / 2.0, area_H / 2.0)
    restricted_mask = config.get("restricted_mask", None)

    # Clamp positions for fitness evaluation (penalty handles OOB separately)
    clamped = np.clip(positions, [0, 0], [area_W, area_H])

    # Coverage component
    cov = coverage_map(clamped, area_W, area_H, Rs, lam, cell_size, restricted_mask)
    if restricted_mask is not None:
        valid_cells = ~restricted_mask
        coverage_ratio = float(np.mean(cov[valid_cells])) if valid_cells.any() else 0.0
    else:
        coverage_ratio = float(np.mean(cov))

    # Energy component
    energy_avg = _energy_cost(clamped, area_W, area_H)

    # Connectivity component
    connectivity_ratio = _connectivity_ratio(clamped, Rc, sink)

    # Base fitness
    fitness = (
        w1 * (1.0 - coverage_ratio)
        + w2 * energy_avg
        + w3 * (1.0 - connectivity_ratio)
    )

    # Adaptive OOB penalty
    alpha = _adaptive_penalty_alpha(iteration, max_iterations)
    if alpha > 0:
        penalty = _oob_penalty(positions, area_W, area_H)
        fitness += alpha * penalty

    return float(fitness)
