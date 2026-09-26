"""
services/surrogate_service.py
-------------------------------
Service layer for managing the PyTorch surrogate model lifecycle:
- Checkpoint location management
- Training threshold checks (≥ 50 runs, retrain every 10 runs)
- Status reporting for API/UI
- Loading surrogate model for PSO inference
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
import asyncio

from app.core.surrogate_model import (
    FitnessSurrogateMLP,
    load_checkpoint,
    train_surrogate,
)
from app.db.repositories import run_repository

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "surrogate.pt"
TRAINING_THRESHOLD = 50
RETRAIN_INTERVAL = 10


def get_checkpoint_path() -> Path:
    return CHECKPOINT_PATH


def is_surrogate_trained() -> bool:
    return CHECKPOINT_PATH.exists()


async def get_surrogate_status() -> dict:
    """
    Return status dictionary matching API contract:
    { trained: bool, runs_available: int, runs_needed: int, last_trained: str | None }
    """
    runs_available = await run_repository.count_runs()
    trained = is_surrogate_trained()
    runs_needed = max(0, TRAINING_THRESHOLD - runs_available)

    last_trained = None
    if trained:
        try:
            mtime = CHECKPOINT_PATH.stat().st_mtime
            last_trained = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            pass

    return {
        "trained": trained,
        "runs_available": runs_available,
        "runs_needed": runs_needed,
        "last_trained": last_trained,
    }


def load_surrogate_if_available() -> tuple[FitnessSurrogateMLP | None, dict]:
    """Load model if checkpoint exists, else return (None, {})."""
    if not is_surrogate_trained():
        return None, {}
    try:
        return load_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        logger.warning("Failed to load surrogate checkpoint: %s", e)
        return None, {}


async def run_training_pipeline(epochs: int = 50) -> dict:
    """
    Async pipeline to fetch samples from MongoDB, train surrogate MLP, and save checkpoint.
    """
    samples = await run_repository.get_training_samples(limit=500)
    if len(samples) < TRAINING_THRESHOLD:
        raise ValueError(f"Not enough training samples: {len(samples)} available, {TRAINING_THRESHOLD} required")

    # Run blocking PyTorch training in executor
    loop = asyncio.get_running_loop()

    def _train():
        model, metrics = train_surrogate(
            training_runs=samples,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            save_path=CHECKPOINT_PATH,
        )
        return metrics

    metrics = await loop.run_in_executor(None, _train)
    logger.info("Surrogate model trained successfully: %s", metrics)
    return metrics
