"""
validators/optimize_validator.py
---------------------------------
Input validation for POST /api/v1/optimize.

Enforces ALL cross-field business rules that Pydantic's per-field constraints
cannot express alone. Returns a list of human-readable error strings so the
controller can return a clean 422 with descriptive messages.

Rules enforced here:
  1. weights.w1 + w2 + w3 must equal 1.0 (within tolerance)
  2. sensing_radius (Rs) must be < comm_radius (Rc)
  3. area width and height must be > 0 (belt-and-suspenders; Pydantic also checks)
  4. num_nodes must fit meaningfully within the area
  5. Every RestrictedArea rect must be within field bounds and x1<x2, y1<y2
  6. cell_size must be <= min(area.width, area.height)
  7. strategy must be one of the allowed values

No I/O, no HTTP — pure logic returning error lists.
"""

from __future__ import annotations

from app.models.config import OptimizationConfig

ALLOWED_STRATEGIES = {"pso", "pso_vdcoa", "random", "grid", "rl"}
ALLOWED_METHODS = {"pso", "rl"}
WEIGHT_TOLERANCE = 1e-6


def validate_optimize_config(config: OptimizationConfig) -> list[str]:
    """
    Validate cross-field rules for an OptimizationConfig.

    Args:
        config: A fully-parsed OptimizationConfig (Pydantic has already
                validated individual field types and ranges).

    Returns:
        List of error message strings. Empty list means the config is valid.
    """
    errors: list[str] = []

    # 1. Weights must sum to 1
    weight_sum = config.weights.w1 + config.weights.w2 + config.weights.w3
    if abs(weight_sum - 1.0) > WEIGHT_TOLERANCE:
        errors.append(
            f"weights.w1 + w2 + w3 must equal 1.0, got {weight_sum:.6f}. "
            f"(w1={config.weights.w1}, w2={config.weights.w2}, w3={config.weights.w3})"
        )

    # 2. Rs must be strictly less than Rc
    if config.sensing_radius >= config.comm_radius:
        errors.append(
            f"sensing_radius (Rs={config.sensing_radius}) must be strictly less than "
            f"comm_radius (Rc={config.comm_radius}). A node cannot communicate "
            f"without also sensing."
        )

    # 3. Area dimensions sanity (Pydantic gt=0 already handles zero/negative,
    #    but we add a meaningful upper-bound check)
    if config.area.width > 100_000 or config.area.height > 100_000:
        errors.append(
            "Area dimensions must be <= 100,000 metres in each axis. "
            f"Got width={config.area.width}, height={config.area.height}."
        )

    # 4. num_nodes must be at least 1 (Pydantic ge=1 handles this, but add
    #    an upper sanity bound)
    if config.num_nodes > 10_000:
        errors.append(
            f"num_nodes={config.num_nodes} exceeds maximum of 10,000."
        )

    # 5. Restricted areas must be valid rectangles within field bounds
    W, H = config.area.width, config.area.height
    for i, ra in enumerate(config.restricted_areas):
        if ra.x1 >= ra.x2:
            errors.append(
                f"restricted_areas[{i}]: x1 ({ra.x1}) must be < x2 ({ra.x2})."
            )
        if ra.y1 >= ra.y2:
            errors.append(
                f"restricted_areas[{i}]: y1 ({ra.y1}) must be < y2 ({ra.y2})."
            )
        if ra.x1 < 0 or ra.y1 < 0 or ra.x2 > W or ra.y2 > H:
            errors.append(
                f"restricted_areas[{i}] ({ra.x1},{ra.y1})-({ra.x2},{ra.y2}) "
                f"extends outside the field bounds (0,0)-({W},{H})."
            )

    # 6. cell_size must be <= smallest field dimension
    min_dim = min(config.area.width, config.area.height)
    if config.cell_size > min_dim:
        errors.append(
            f"cell_size ({config.cell_size}) must be <= the smallest field "
            f"dimension ({min_dim})."
        )

    # 7. Strategy and Method must be known values
    if config.strategy not in ALLOWED_STRATEGIES:
        errors.append(
            f"strategy '{config.strategy}' is not recognised. "
            f"Allowed values: {sorted(ALLOWED_STRATEGIES)}."
        )

    if config.method not in ALLOWED_METHODS:
        errors.append(
            f"method '{config.method}' is not recognised. "
            f"Allowed values: {sorted(ALLOWED_METHODS)}."
        )

    return errors


def is_valid_optimize_config(config: OptimizationConfig) -> bool:
    """Convenience wrapper — returns True if there are no validation errors."""
    return len(validate_optimize_config(config)) == 0
