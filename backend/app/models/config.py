"""
models/config.py
----------------
Pydantic data shapes for optimization input configuration.

Pure data models — no methods, no logic, no HTTP knowledge.
Validation rules (weights sum to 1, Rs < Rc, etc.) live in validators/.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Area(BaseModel):
    """Dimensions of the deployment field in metres."""

    width: float = Field(..., gt=0, description="Field width in metres (must be > 0)")
    height: float = Field(..., gt=0, description="Field height in metres (must be > 0)")


class Weights(BaseModel):
    """Multi-objective fitness weights. Must sum to 1 — enforced in validator."""

    w1: float = Field(..., ge=0.0, le=1.0, description="Coverage weight")
    w2: float = Field(..., ge=0.0, le=1.0, description="Energy weight")
    w3: float = Field(..., ge=0.0, le=1.0, description="Connectivity weight")


class PSOParams(BaseModel):
    """Hyper-parameters for the PSO algorithm."""

    swarm_size: int = Field(30, ge=1, description="Number of independent particles")
    iterations: int = Field(500, ge=1, le=10_000, description="Total PSO iterations")
    inertia: float = Field(0.7, ge=0.0, le=1.0, description="Inertia weight ω")
    c1: float = Field(1.5, ge=0.0, description="Cognitive acceleration coefficient")
    c2: float = Field(1.5, ge=0.0, description="Social acceleration coefficient")


class RestrictedArea(BaseModel):
    """Axis-aligned rectangular region forbidden for sensor placement."""

    x1: float = Field(..., description="Left boundary (metres)")
    y1: float = Field(..., description="Bottom boundary (metres)")
    x2: float = Field(..., description="Right boundary (metres)")
    y2: float = Field(..., description="Top boundary (metres)")


class NonCriticalArea(BaseModel):
    """Axis-aligned rectangular region where coverage is not required."""

    x1: float
    y1: float
    x2: float
    y2: float


class OptimizationConfig(BaseModel):
    """
    Full input configuration for POST /api/v1/optimize.

    Cross-field rules (weights sum, Rs < Rc, RA within bounds) are enforced
    separately in validators/optimize_validator.py, not here.
    """

    area: Area
    num_nodes: int = Field(..., ge=1, description="Number of sensor nodes to deploy")
    sensing_radius: float = Field(..., gt=0.0, description="Reliable sensing radius Rs (metres)")
    comm_radius: float = Field(..., gt=0.0, description="Communication radius Rc (metres)")
    initial_energy: float = Field(1.0, gt=0.0, description="Initial node energy budget (J)")
    weights: Weights
    pso_params: PSOParams = Field(default_factory=PSOParams)
    use_gpu: bool = Field(False, description="Use CUDA GPU acceleration if available")
    use_vdcoa: bool = Field(False, description="Apply VDCOA chaos refinement after PSO")
    seed: int | None = Field(None, description="RNG seed for reproducibility")
    restricted_areas: list[RestrictedArea] = Field(
        default_factory=list, description="Regions forbidden for sensor placement"
    )
    non_critical_areas: list[NonCriticalArea] = Field(
        default_factory=list, description="Regions where coverage is not evaluated"
    )
    strategy: str = Field(
        "pso",
        description="Algorithm to run: 'pso', 'pso_vdcoa', 'random', 'grid', 'rl'",
    )
    method: str = Field("pso", description="Placement method: 'pso' or 'rl'")
    cell_size: float = Field(1.0, gt=0.0, description="Grid resolution (metres per cell)")
    intersections: list[RestrictedArea] = Field(
        default_factory=list, description="Intersection regions worth double coverage weight"
    )
