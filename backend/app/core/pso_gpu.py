"""
core/pso_gpu.py
---------------
GPU-accelerated PSO fitness evaluation via Numba CUDA.

Architecture
------------
- Detection: queries numba + cuda.is_available() at import time.
- Graceful fallback: if no CUDA device is found (or Numba not installed),
  run_pso_gpu() transparently delegates to the CPU run_pso() and sets
  gpu_used=False in the result.
- Kernel design: ONE CUDA thread = ONE particle (complete N-node deployment).
  Each thread iterates over every grid cell to compute coverage probability
  and energy cost. This parallelises across the swarm (P particles run
  simultaneously).
- Grid-stride loop: threads wrap around so P > gridDim×blockDim is handled
  correctly.
- Self-adaptive kernel config: BlockDim = min(P, 512),
  GridDim = ceil(P / BlockDim).
- Position/velocity updates and connectivity BFS stay on CPU (single copy,
  no transfer overhead). Only the expensive fitness kernel runs on GPU.
- Double precision (float64) throughout to match CPU results within 1e-4.

No I/O, no HTTP, no business logic — pure algorithm.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np

from .pso import run_pso, _build_restricted_mask
from .fitness import _connectivity_ratio, _energy_cost
from .sensing_model import coverage_map as compute_coverage_map


# ---------------------------------------------------------------------------
# CUDA / Numba detection (at import time, never raises)
# ---------------------------------------------------------------------------

_NUMBA_AVAILABLE: bool = False
_HAS_CUDA: bool = False

try:
    import numba  # noqa: F401 — check install only
    from numba import cuda as _numba_cuda  # type: ignore[import]
    _NUMBA_AVAILABLE = True
    try:
        _HAS_CUDA = _numba_cuda.is_available()
    except Exception:
        _HAS_CUDA = False
except ImportError:
    pass


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is detected and Numba is installed."""
    return _HAS_CUDA


# ---------------------------------------------------------------------------
# CUDA kernel (defined lazily — avoids cuda.jit at import on non-CUDA hosts)
# ---------------------------------------------------------------------------

_KERNEL_CACHE: object | None = None


def _get_coverage_energy_kernel():
    """
    Build and cache the CUDA kernel on first call.
    Must only be called when _HAS_CUDA is True.
    """
    global _KERNEL_CACHE
    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE

    from numba import cuda  # type: ignore[import]

    @cuda.jit
    def _coverage_energy_kernel(
        positions,      # (P, N, 2) float64 — sensor positions per particle
        area_W,         # scalar float64
        area_H,         # scalar float64
        Rs,             # scalar float64 — reliable sensing radius
        lam,            # scalar float64 — falloff coefficient
        cell_size,      # scalar float64 — grid resolution
        coverage_out,   # (P,) float64 — output: per-particle coverage ratio
        energy_out,     # (P,) float64 — output: per-particle energy cost
    ):
        """
        One CUDA thread computes fitness components for one particle.

        Coverage: mean detection probability across all grid cells using
            P_cell = 1 - prod_n(1 - P_n(d_n))
        Energy: mean normalised Euclidean distance to field centroid.

        Grid-stride loop handles P > gridDim × blockDim safely.
        """
        stride = cuda.gridsize(1)
        p_idx = cuda.grid(1)
        P = positions.shape[0]
        N = positions.shape[1]

        while p_idx < P:
            cols = int(math.ceil(area_W / cell_size))
            rows = int(math.ceil(area_H / cell_size))
            total_cells = rows * cols

            # --- Coverage ---
            coverage_sum = 0.0
            for r in range(rows):
                for c in range(cols):
                    cx = (c + 0.5) * cell_size
                    cy = (r + 0.5) * cell_size

                    prob_miss = 1.0
                    for n in range(N):
                        sx = positions[p_idx, n, 0]
                        sy = positions[p_idx, n, 1]
                        dx = cx - sx
                        dy = cy - sy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist <= Rs:
                            p_detect = 1.0
                        else:
                            p_detect = math.exp(-lam * (dist - Rs))

                        prob_miss *= (1.0 - p_detect)

                    coverage_sum += (1.0 - prob_miss)

            coverage_out[p_idx] = coverage_sum / total_cells if total_cells > 0 else 0.0

            # --- Energy (normalised distance to centroid) ---
            cx0 = area_W * 0.5
            cy0 = area_H * 0.5
            max_dist_sq = cx0 * cx0 + cy0 * cy0
            max_dist = math.sqrt(max_dist_sq) if max_dist_sq > 0.0 else 1.0

            energy_sum = 0.0
            for n in range(N):
                sx = positions[p_idx, n, 0]
                sy = positions[p_idx, n, 1]
                dx = sx - cx0
                dy = sy - cy0
                energy_sum += math.sqrt(dx * dx + dy * dy)

            energy_out[p_idx] = (energy_sum / N) / max_dist if N > 0 else 0.0

            p_idx += stride  # grid-stride advance

    _KERNEL_CACHE = _coverage_energy_kernel
    return _KERNEL_CACHE


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pso_gpu(config: dict, on_iteration=None) -> dict:
    """
    Run PSO with GPU-accelerated fitness evaluation.

    If no CUDA device is available, silently falls back to the CPU
    implementation (run_pso) and sets gpu_used=False in the result.

    Args / Returns: identical contract to core/pso.run_pso().
    """
    if not _HAS_CUDA:
        warnings.warn(
            "No CUDA-capable GPU detected (or Numba not installed). "
            "Falling back to CPU PSO — results are identical but slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = run_pso(config, on_iteration=on_iteration)
        result["gpu_used"] = False
        return result

    return _run_gpu_impl(config, on_iteration=on_iteration)


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _run_gpu_impl(config: dict, on_iteration=None) -> dict:
    """
    GPU PSO loop.

    Position/velocity updates run on CPU (cheap, sequential).
    Fitness evaluation (coverage + energy) runs on GPU (expensive, parallel).
    Connectivity BFS runs on CPU (not GPU-friendly for small P).
    """
    from numba import cuda  # type: ignore[import]

    # --- Unpack config ---
    area = config["area"]
    area_W = float(area["width"])
    area_H = float(area["height"])
    N = int(config["num_nodes"])
    Rs = float(config["sensing_radius"])
    Rc = float(config["comm_radius"])
    lam = float(config.get("lam", 0.5))
    cell_size = float(config.get("cell_size", 1.0))
    seed = config.get("seed", None)
    default_sink = (area_W / 2.0, area_H / 2.0)
    sink = tuple(config["sink"]) if config.get("sink") is not None else default_sink

    pso_params = config.get("pso_params", {})
    P = int(pso_params.get("swarm_size", 30))
    G = int(pso_params.get("iterations", 500))
    omega = float(pso_params.get("inertia", 0.7))
    c1 = float(pso_params.get("c1", 1.5))
    c2 = float(pso_params.get("c2", 1.5))

    weights = config.get("weights", {"w1": 0.5, "w2": 0.25, "w3": 0.25})
    w1 = float(weights.get("w1", 0.5))
    w2 = float(weights.get("w2", 0.25))
    w3 = float(weights.get("w3", 0.25))

    restricted_areas = config.get("restricted_areas", [])
    restricted_mask = _build_restricted_mask(restricted_areas, area_W, area_H, cell_size)

    # --- RNG (same seed as CPU → deterministic comparison) ---
    rng = np.random.default_rng(seed)

    # --- Swarm initialisation ---
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
    velocities = rng.uniform(low=-v_max, high=v_max, size=(P, N, 2)).astype(np.float64)

    # --- Self-adaptive kernel config ---
    block_dim = min(P, 512)
    grid_dim = math.ceil(P / block_dim)
    kernel = _get_coverage_energy_kernel()

    # Pre-allocate GPU output arrays (reused every iteration)
    d_coverage = cuda.device_array(P, dtype=np.float64)
    d_energy = cuda.device_array(P, dtype=np.float64)

    def _eval_fitness(pos: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for all P particles.
        pos: (P, N, 2) numpy array — positions this iteration.
        Returns: (P,) fitness array.
        """
        clamped = np.clip(pos, [0.0, 0.0], [area_W, area_H])
        d_pos = cuda.to_device(clamped)

        kernel[grid_dim, block_dim](
            d_pos,
            np.float64(area_W), np.float64(area_H),
            np.float64(Rs), np.float64(lam), np.float64(cell_size),
            d_coverage, d_energy,
        )
        cuda.synchronize()

        cov_ratios = d_coverage.copy_to_host()
        energy_vals = d_energy.copy_to_host()

        # Connectivity BFS on CPU (one call per particle)
        fitnesses = np.zeros(P, dtype=np.float64)
        for i in range(P):
            conn = _connectivity_ratio(clamped[i], Rc, sink)
            fitnesses[i] = (
                w1 * (1.0 - cov_ratios[i])
                + w2 * energy_vals[i]
                + w3 * (1.0 - conn)
            )
        return fitnesses

    # --- Initial evaluation ---
    fitnesses = _eval_fitness(positions)
    pbest_pos = positions.copy()
    pbest_fit = fitnesses.copy()

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])
    fitness_history = [gbest_fit]

    # --- Main PSO loop ---
    t_start = time.perf_counter()

    for g in range(1, G + 1):
        omega_g = omega * (1.0 - 0.5 * (g / G))

        from app.core.pso import _match_nodes

        # Spatial nearest-neighbor matching for cognitive & social targets
        pbest_matched = np.array([_match_nodes(positions[i], pbest_pos[i]) for i in range(P)])
        gbest_matched = np.array([_match_nodes(positions[i], gbest_pos) for i in range(P)])

        velocities = (
            omega_g * velocities
            + c1 * r1 * (pbest_matched - positions)
            + c2 * r2 * (gbest_matched - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)
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

        fitnesses = _eval_fitness(positions)

        improved = fitnesses < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitnesses[improved]

        best_idx = int(np.argmin(pbest_fit))
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[best_idx])
            gbest_pos = pbest_pos[best_idx].copy()

        fitness_history.append(gbest_fit)

        if on_iteration is not None:
            clamped_positions = np.clip(positions, [0.0, 0.0], [area_W, area_H])
            clamped_gbest_pos = np.clip(gbest_pos, [0.0, 0.0], [area_W, area_H])
            on_iteration(g, clamped_positions, clamped_gbest_pos, gbest_fit)
            time.sleep(max(0.002, min(0.01, 3.0 / G)))

    compute_time = time.perf_counter() - t_start

    # --- Final metrics (CPU, on best deployment) ---
    best_clamped = np.clip(gbest_pos, [0.0, 0.0], [area_W, area_H])

    final_cov_map = compute_coverage_map(
        best_clamped, area_W, area_H, Rs,
        lam=lam, cell_size=cell_size, restricted_mask=restricted_mask,
    )

    if restricted_mask is not None:
        valid = ~restricted_mask
        coverage_ratio = float(np.mean(final_cov_map[valid])) if valid.any() else 0.0
    else:
        coverage_ratio = float(np.mean(final_cov_map))

    return {
        "best_positions": best_clamped,
        "fitness_history": fitness_history,
        "coverage_map": final_cov_map,
        "coverage_ratio": coverage_ratio,
        "connectivity_ratio": float(_connectivity_ratio(best_clamped, Rc, sink)),
        "avg_energy": float(_energy_cost(best_clamped, area_W, area_H)),
        "compute_time_seconds": compute_time,
        "iterations_run": G,
        "gpu_used": True,
    }
