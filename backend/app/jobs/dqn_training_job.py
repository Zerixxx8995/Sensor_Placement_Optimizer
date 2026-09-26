"""
jobs/dqn_training_job.py
------------------------
Background task runner for DQN training loop.
"""

import asyncio
import logging
from app.services.redeployment_service import run_dqn_training_pipeline

logger = logging.getLogger(__name__)


def execute_dqn_training(episodes: int = 500) -> None:
    """Entrypoint for FastAPI BackgroundTasks."""
    try:
        metrics = asyncio.run(run_dqn_training_pipeline(episodes=episodes))
        logger.info("DQN background training finished: %s", metrics)
    except Exception as e:
        logger.error("DQN background training failed: %s", e, exc_info=True)
