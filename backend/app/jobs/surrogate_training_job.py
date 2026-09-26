"""
jobs/surrogate_training_job.py
------------------------------
Background job wrapper for surrogate model training.
"""

import asyncio
import logging
from app.services.surrogate_service import run_training_pipeline

logger = logging.getLogger(__name__)


def execute_surrogate_training(epochs: int = 50) -> None:
    """Synchronous entrypoint for FastAPI BackgroundTasks."""
    try:
        metrics = asyncio.run(run_training_pipeline(epochs=epochs))
        logger.info("Surrogate background training job completed: %s", metrics)
    except Exception as e:
        logger.error("Surrogate background training failed: %s", e, exc_info=True)
