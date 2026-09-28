"""
core/city_templates.py
----------------------
Generates 20 pre-defined city grid templates (NumPy arrays) for training and evaluation.

Cell types:
  0: Road cell (normal priority)
  1: Building cell (no-install zone)
  2: Low-priority cell (parks / empty lots)
  3: Intersection cell (high-priority road cell, double coverage weight)
"""

import numpy as np


def generate_grid_city(L: int = 20, W: int = 20, block_size: int = 4, seed: int = 0) -> np.ndarray:
    """Manhattan-style grid pattern with regular streets and intersections."""
    rng = np.random.RandomState(seed)
    grid = np.ones((L, W), dtype=np.int32)  # Default to buildings (1)

    # Place vertical and horizontal road lines
    road_cols = list(range(1, W, block_size))
    road_rows = list(range(1, L, block_size))

    for r in road_rows:
        grid[r, :] = 0  # Road cell
    for c in road_cols:
        grid[:, c] = 0  # Road cell

    # Mark intersections where road rows and cols cross
    for r in road_rows:
        for c in road_cols:
            grid[r, c] = 3  # Intersection cell

    # Place a few low-priority parks in building blocks
    for r in range(0, L - block_size + 1, block_size):
        for c in range(0, W - block_size + 1, block_size):
            if rng.rand() < 0.25:
                # Turn a block into low-priority park (2)
                for pr in range(r + 2, min(r + block_size, L)):
                    for pc in range(c + 2, min(c + block_size, W)):
                        if grid[pr, pc] == 1:
                            grid[pr, pc] = 2

    return grid


def generate_radial_city(L: int = 20, W: int = 20, seed: int = 0) -> np.ndarray:
    """Radial city pattern with concentric rings and radiating avenues."""
    grid = np.ones((L, W), dtype=np.int32)
    cy, cx = L / 2.0, W / 2.0
    max_r = min(L, W) / 2.0

    # Concentric rings
    rings = [max_r * 0.3, max_r * 0.6, max_r * 0.85]
    
    # 8 Radial avenues (angles 0, 45, 90, 135, 180, 225, 270, 315 deg)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    for r in range(L):
        for c in range(W):
            dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
            # Check ring roads
            is_ring = any(abs(dist - ring_r) < 1.0 for ring_r in rings)

            # Check radial roads
            angle = np.arctan2(r - cy, c - cx) % (2 * np.pi)
            is_radial = any(abs((angle - a + np.pi) % (2 * np.pi) - np.pi) < 0.25 for a in angles)

            if is_ring and is_radial:
                grid[r, c] = 3  # Intersection
            elif is_ring or is_radial:
                grid[r, c] = 0  # Road
            elif dist > max_r * 0.9:
                grid[r, c] = 2  # Low-priority park / outer ring

    return grid


def generate_random_city(L: int = 20, W: int = 20, num_walks: int = 6, seed: int = 0) -> np.ndarray:
    """Random walk road generation with intersections where walks cross."""
    rng = np.random.RandomState(seed)
    grid = np.ones((L, W), dtype=np.int32)
    visit_count = np.zeros((L, W), dtype=np.int32)

    for _ in range(num_walks):
        r = rng.randint(0, L)
        c = rng.randint(0, W)
        dr, dc = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
        if dr == 0 and dc == 0:
            dr = 1

        steps = rng.randint(L // 2, L * 2)
        for _ in range(steps):
            r = max(0, min(L - 1, r + dr))
            c = max(0, min(W - 1, c + dc))
            visit_count[r, c] += 1
            if rng.rand() < 0.2:  # Turn
                dr, dc = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])

    # Assign cell types based on visit count
    grid[visit_count == 1] = 0  # Normal road
    grid[visit_count >= 2] = 3  # Intersection where paths cross

    # Add a few parks (2) on unvisited border cells
    grid[(visit_count == 0) & (rng.rand(L, W) < 0.15)] = 2

    # Ensure at least some road cells exist
    if np.sum((grid == 0) | (grid == 3)) < 10:
        grid[L // 2, :] = 0
        grid[:, W // 2] = 0

    return grid


def generate_20_city_templates(L: int = 20, W: int = 20) -> list[np.ndarray]:
    """Generates 20 diverse city grid templates (mix of Grid, Radial, Random)."""
    templates = []
    # 7 Grid cities
    for i in range(7):
        templates.append(generate_grid_city(L, W, block_size=3 + (i % 3), seed=i * 10))

    # 7 Radial cities
    for i in range(7):
        templates.append(generate_radial_city(L, W, seed=i * 20 + 1))

    # 6 Random cities
    for i in range(6):
        templates.append(generate_random_city(L, W, seed=i * 30 + 2))

    return templates


# Pre-generated default 20 templates (20x20)
CITY_TEMPLATES_20X20 = generate_20_city_templates(20, 20)


def get_city_template(index: int = None, L: int = 20, W: int = 20, seed: int = None) -> np.ndarray:
    """Retrieve or generate a city grid template."""
    if L == 20 and W == 20:
        if index is not None:
            return CITY_TEMPLATES_20X20[index % len(CITY_TEMPLATES_20X20)].copy()
        elif seed is not None:
            return CITY_TEMPLATES_20X20[seed % len(CITY_TEMPLATES_20X20)].copy()
        else:
            idx = np.random.randint(0, len(CITY_TEMPLATES_20X20))
            return CITY_TEMPLATES_20X20[idx].copy()
    else:
        # Generate on the fly for custom sizes
        t_type = (seed if seed is not None else 0) % 3
        if t_type == 0:
            return generate_grid_city(L, W, seed=seed or 0)
        elif t_type == 1:
            return generate_radial_city(L, W, seed=seed or 0)
        else:
            return generate_random_city(L, W, seed=seed or 0)
