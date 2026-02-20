"""
API endpoints for automation scheduler.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ..automation.scheduler import (
    FactoryScheduler,
    GenerationConfig,
    QualityConfig,
    create_scheduler,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global scheduler instance
_scheduler: Optional[FactoryScheduler] = None


def get_scheduler() -> FactoryScheduler:
    """Get or create scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = create_scheduler()
    return _scheduler


@router.post("/generate")
async def trigger_generation(
    count: int = Query(
        100, ge=1, le=10000, description="Number of circuits to generate"
    ),
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """
    Trigger a batch generation of new circuits.

    Args:
        count: Number of circuits to generate

    Returns:
        Generation results
    """
    try:
        scheduler = get_scheduler()
        result = scheduler.run_generation_batch(count=count)
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Generation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unroll")
async def trigger_unroll(
    max_circuits: int = Query(50, ge=1, le=1000, description="Max circuits to unroll"),
    equivalents_per: int = Query(
        100, ge=1, le=1000, description="Equivalents per circuit"
    ),
) -> Dict[str, Any]:
    """
    Trigger unrolling of representative circuits.

    Args:
        max_circuits: Maximum circuits to unroll
        equivalents_per: Max equivalents per circuit

    Returns:
        Unrolling results
    """
    try:
        scheduler = get_scheduler()
        result = scheduler.run_unroll_batch(
            max_circuits=max_circuits, equivalents_per_circuit=equivalents_per
        )
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Unroll trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter")
async def trigger_quality_filter() -> Dict[str, Any]:
    """
    Trigger quality filtering based on non-triviality score.

    Returns:
        Filtering results
    """
    try:
        scheduler = get_scheduler()
        result = scheduler.run_quality_filter()
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Filter trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full-cycle")
async def trigger_full_cycle() -> Dict[str, Any]:
    """
    Trigger a complete generation -> unroll -> filter cycle.

    Returns:
        Combined results from all steps
    """
    try:
        scheduler = get_scheduler()
        result = scheduler.run_full_cycle()
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Full cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_scheduler_stats() -> Dict[str, Any]:
    """
    Get automation scheduler statistics.

    Returns:
        Scheduler statistics
    """
    try:
        scheduler = get_scheduler()
        return scheduler.get_stats()
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
