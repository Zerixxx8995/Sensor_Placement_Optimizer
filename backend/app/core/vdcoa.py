"""
core/vdcoa.py
-------------
Variable-Dimension Chaotic Optimization Algorithm (VDCOA) refinement layer.

VDCOA improves a PSO solution by applying chaotic perturbations to the
best-found positions.  The chaos sequence (logistic map) generates
pseudo-random search directions that avoid local optima better than
uniform random restarts.

Reference:
  Liu, H. et al., "A Variable-Dimension Chaotic Optimization Algorithm
  Based on PSO" (paraphrased implementation for WSN placement).

Algorithm outline
-----------------
1. Start from PSO global-best positions (X_best).
2. For each chaos iteration:
   a. Generate a chaotic variable z ∈ (0, 1) via the logistic map:
          z_{n+1} = μ · z_n · (1 − z_n),   μ = 4.0
   b. Map z to a perturbation in [−Δ, +Δ] where Δ shrinks with iteration.
   c. Apply the perturbation to one *variable dimension* at a time
      (randomly chosen node index and x/y coordinate).
   d. Clamp the trial position to field bounds.
   e. If trial fitness < current best fitness → accept.
3. Return the refined positions + merged metadata.

The dimension-selection step keeps the search targeted (one sensor moves
at a time), avoiding the high-dimensional random walk that plagues naive
restarts.

No I/O, no HTTP, no business logic — pure refinement algorithm only.
"""

from __future__ import annotations

import time

import numpy as np

from .fitness import compute_fitness, _connectivity_ratio, _energy_cost
from .sensing_model import coverage_map as compute_coverage_map


# ---------------------------------------------------------------------------
# Logistic chaos map
# ---------------------------------------------------------------------------

def _logistic_map(z: float, mu: float = 4.0) -> float:
    """One step of the logistic chaos map: z' = μ·z·(1−z)."""
    return mu * z * (1.0 - z)


def _init_chaos_seed(rng: np.random.Generator) -> float:
    """
    Draw a seed for the chaos map from (0, 1) excluding {0.25, 0.50, 0.75}
    (those are fixed points / period-2 cycles of the logistic map).
    """
    z = rng.uniform(0.01, 0.99)
    # nudge away from known bad seeds
    while abs(z - 0.25) < 0.01 or abs(z - 0.50) < 0.01 or abs(z - 0.75) < 0.01:
        z = rng.uniform(0.01, 0.99)
    return float(z)


# ---------------------------------------------------------------------------
# Build fitness config helper (mirrors pso.py logic)
# ---------------------------------------------------------------------------

def _build_fitness_config(config: dict) -> dict:
    area = config["area"]
    weights = config["weights"]
    return {
        "area_W": float(area["width"]),
        "area_H": float(area["height"]),
        "Rs":     float(config["sensing_radius"]),
        "Rc":     float(config["comm_radius"]),
        "lam":    float(config.get("lam", 0.5)),
        "cell_size": float(config.get("cell_size", 1.0)),
        "w1": float(weights["w1"]),
        "w2": float(weights["w2"]),
        "w3": float(weights["w3"]),
        "sink": tuple(config.get("sink", (0.0, 0.0))),
        "restricted_mask": config.get("restricted_mask", None),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_vdcoa_refinement(
    pso_result: dict,
    config: dict,
    *,
    chaos_iterations: int = 200,
    delta_init: float = 0.15,
    delta_decay: float = 0.995,
    on_iteration=None,
) -> dict:
    """
    Refine a PSO result using VDCOA chaotic perturbation.

    Args:
        pso_result:        Output dict from ``core/pso.run_pso()``.
        config:            Original top-level optimization config dict.
        chaos_iterations:  Number of chaotic search steps (default 200).
        delta_init:        Initial max perturbation as a fraction of field size.
        delta_decay:       Per-iteration multiplicative decay of delta.
        on_iteration:      Optional callback(iter, positions, gbest, fitness).

    Returns:
        Result dict with the same keys as ``run_pso()``, enriched with
        ``vdcoa_used: True`` and updated runtime stats.
    """
    area = config["area"]
    area_W = float(area["width"])
    area_H = float(area["height"])
    Rs = float(config["sensing_radius"])
    Rc = float(config["comm_radius"])
    cell_size = float(config.get("cell_size", 1.0))
    seed = config.get("seed", None)

    restricted_mask = config.get("restricted_mask", None)
    fitness_cfg = _build_fitness_config(config)
    fitness_cfg["restricted_mask"] = restricted_mask

    rng = np.random.default_rng(seed)

    # Starting point — clone the PSO best
    best_pos = pso_result["best_positions"].copy()
    best_fit = float(
        compute_fitness(best_pos, fitness_cfg, iteration=chaos_iterations, max_iterations=chaos_iterations)
    )

    fitness_history = list(pso_result["fitness_history"])  # carry over PSO history

    # Chaos state
    z = _init_chaos_seed(rng)
    delta_W = area_W * delta_init
    delta_H = area_H * delta_init

    N = len(best_pos)

    t_start = time.perf_counter()

    for i in range(chaos_iterations):
        # Generate chaos value
        z = _logistic_map(z)

        # Choose one node and one dimension to perturb
        node_idx = int(rng.integers(0, N))
        dim = int(rng.integers(0, 2))  # 0 = x, 1 = y

        # Map chaos value to signed perturbation
        perturbation = (2.0 * z - 1.0) * (delta_W if dim == 0 else delta_H)

        trial = best_pos.copy()
        trial[node_idx, dim] = np.clip(
            trial[node_idx, dim] + perturbation,
            0.0,
            area_W if dim == 0 else area_H,
        )

        trial_fit = compute_fitness(trial, fitness_cfg, iteration=i, max_iterations=chaos_iterations)

        if trial_fit < best_fit:
            best_fit = trial_fit
            best_pos = trial

        fitness_history.append(best_fit)

        # Decay search radius
        delta_W *= delta_decay
        delta_H *= delta_decay

        if on_iteration is not None:
            on_iteration(i, best_pos[np.newaxis, :, :], best_pos, best_fit)
            time.sleep(max(0.002, min(0.01, 1.0 / max(1, chaos_iterations))))

    compute_time = pso_result["compute_time_seconds"] + (time.perf_counter() - t_start)

    # Recompute final metrics
    final_cov_map = compute_coverage_map(
        best_pos, area_W, area_H, Rs,
        lam=fitness_cfg.get("lam", 0.5),
        cell_size=cell_size,
        restricted_mask=restricted_mask,
    )

    if restricted_mask is not None:
        valid = ~restricted_mask
        coverage_ratio = float(np.mean(final_cov_map[valid])) if valid.any() else 0.0
    else:
        coverage_ratio = float(np.mean(final_cov_map))

    connectivity_ratio = float(
        _connectivity_ratio(best_pos, Rc, sink=fitness_cfg.get("sink", (0.0, 0.0)))
    )
    avg_energy = float(_energy_cost(best_pos, area_W, area_H))

    return {
        "best_positions":      best_pos,
        "fitness_history":     fitness_history,
        "coverage_map":        final_cov_map,
        "coverage_ratio":      coverage_ratio,
        "connectivity_ratio":  connectivity_ratio,
        "avg_energy":          avg_energy,
        "compute_time_seconds": compute_time,
        "iterations_run":      pso_result["iterations_run"] + chaos_iterations,
        "gpu_used":            pso_result.get("gpu_used", False),
        "vdcoa_used":          True,
    }
