"""
jobs/optimization_job.py
-------------------------
Background task wrapper that dispatches to the correct algorithm based on
the ``strategy`` field in config.

Strategy dispatch table
-----------------------
  "random"    → core/baselines.place_random()
  "grid"      → core/baselines.place_grid()
  "pso"       → core/pso.run_pso()  (GPU variant if use_gpu=True)
  "pso_vdcoa" → core/pso.run_pso() then core/vdcoa.run_vdcoa_refinement()

All algorithms return the same result-dict contract so job_store.set_complete
is algorithm-agnostic. The on_iteration callback is only meaningful for PSO
variants (baselines complete instantly, so no progress events are emitted).

No HTTP knowledge and no business logic — pure job lifecycle + dispatch only.
"""

import asyncio
import logging

from app.jobs import job_store
from app.core.pso import run_pso
from app.core.pso_gpu import run_pso_gpu
from app.core.vdcoa import run_vdcoa_refinement
from app.core.baselines import place_random, place_grid

logger = logging.getLogger(__name__)


def run_optimization_job(job_id: str, config: dict) -> None:
    """
    Execute the optimization run for the requested strategy as a background task.

    Called by FastAPI BackgroundTasks. Updates job_store at each lifecycle
    transition so the polling / SSE endpoints see live status.

    Args:
        job_id: The UUID registered in job_store by the service layer.
        config: Raw dict already shaped by optimization_service._to_pso_config().
                Must contain a ``strategy`` key.
    """
    job_store.set_running(job_id)

    def on_iteration(g: int, positions, gbest_pos, gbest_fit) -> None:
        """Publish per-iteration events for PSO strategies (ignored by baselines)."""
        job_store.publish_iteration(job_id, {
            "event": "iteration",
            "iteration": g,
            "best_positions": gbest_pos.tolist(),
            "best_fitness": float(gbest_fit),
            "particles": positions.tolist(),
        })

    try:
        strategy = config.get("strategy", "pso")
        method = config.get("method", "pso")

        # ---------------------------------------------------------------
        # RL strategy (PPO placement agent)
        # ---------------------------------------------------------------
        if method == "rl" or strategy == "rl":
            from app.services import ppo_placement_service
            from app.models.config import OptimizationConfig

            config_obj = OptimizationConfig.model_validate(config)
            result = ppo_placement_service.run_ppo_placement(config_obj, job_id=job_id)
            result["strategy"] = "rl"

        # ---------------------------------------------------------------
        # Baseline strategies  (instant, no iteration events)
        # ---------------------------------------------------------------
        elif strategy == "random":
            result = place_random(config)
            result["strategy"] = "random"

        elif strategy == "grid":
            result = place_grid(config)
            result["strategy"] = "grid"

        # ---------------------------------------------------------------
        # PSO-based strategies
        # ---------------------------------------------------------------
        else:
            # Check for trained surrogate model
            surrogate_model = None
            try:
                from app.services import surrogate_service
                surrogate_model, _ = surrogate_service.load_surrogate_if_available()
            except Exception as s_exc:
                logger.warning("Surrogate model load failed: %s", s_exc)

            # --- Phase 1: PSO ---
            if config.get("use_gpu"):
                pso_result = run_pso_gpu(config, on_iteration=on_iteration)
            else:
                pso_result = run_pso(config, on_iteration=on_iteration, surrogate_model=surrogate_model)

            # --- Phase 2: VDCOA refinement (pso_vdcoa only) ---
            if strategy == "pso_vdcoa":
                pso_iterations = config.get("pso_params", {}).get("iterations", 500)
                chaos_iterations = max(50, pso_iterations // 5)  # 20 % of PSO budget

                def on_vdcoa_iteration(i: int, positions, gbest_pos, gbest_fit) -> None:
                    g = pso_iterations + i + 1
                    job_store.publish_iteration(job_id, {
                        "event": "iteration",
                        "iteration": g,
                        "best_positions": gbest_pos.tolist(),
                        "best_fitness": float(gbest_fit),
                        "particles": positions.tolist(),
                    })

                result = run_vdcoa_refinement(
                    pso_result,
                    config,
                    chaos_iterations=chaos_iterations,
                    on_iteration=on_vdcoa_iteration,
                )
                result["strategy"] = "pso_vdcoa"
            else:
                # Plain PSO
                result = pso_result
                result["strategy"] = "pso"

        job_store.set_complete(job_id, result)

        # ---------------------------------------------------------------
        # Persist to MongoDB (best-effort — failure must not fail the job)
        # ---------------------------------------------------------------
        try:
            from app.db.repositories import run_repository  # local import avoids circular deps
            from app.services import surrogate_service

            asyncio.run(
                run_repository.save_run(
                    job_id=job_id,
                    config=config,
                    result=result,
                    surrogate_used=result.get("surrogate_used", False),
                    surrogate_switch_iteration=result.get("surrogate_switch_iteration"),
                )
            )

            # Auto-trigger surrogate training if threshold met
            async def _check_and_train():
                count = await run_repository.count_runs()
                if count >= surrogate_service.TRAINING_THRESHOLD:
                    if not surrogate_service.is_surrogate_trained() or count % surrogate_service.RETRAIN_INTERVAL == 0:
                        logger.info("Auto-triggering surrogate model training (%d runs stored)", count)
                        await surrogate_service.run_training_pipeline()

            asyncio.run(_check_and_train())

        except Exception as mongo_exc:  # noqa: BLE001
            logger.warning("MongoDB save / auto-train failed for job %s: %s", job_id, mongo_exc)

    except Exception as exc:  # noqa: BLE001
        job_store.set_failed(job_id, str(exc))

