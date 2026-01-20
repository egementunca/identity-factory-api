"""
Worker function for circuit generation in a separate process.
Must be in a separate module for pickle compatibility.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def generate_in_process(
    generator_name: str,
    width: int,
    gate_count: int,
    gate_set: str,
    max_circuits: Optional[int],
    config: Optional[Dict[str, Any]],
    progress_queue,  # multiprocessing.Queue
) -> Dict[str, Any]:
    """
    Run circuit generation in a subprocess.
    
    This function is called via ProcessPoolExecutor.
    It re-initializes the generator registry in the new process.
    """
    from ..generators import GenerationProgress, get_registry
    
    registry = get_registry()
    generator = registry.get_generator(generator_name)
    
    if not generator:
        return {
            "success": False,
            "error": f"Generator '{generator_name}' not found",
            "circuits": [],
        }
    
    started_at = datetime.now()
    
    def progress_callback(progress: GenerationProgress):
        """Send progress updates through the queue."""
        try:
            progress_queue.put_nowait({
                "status": progress.status.value,
                "circuits_found": progress.circuits_found,
                "circuits_stored": progress.circuits_stored,
                "duplicates_skipped": progress.duplicates_skipped,
                "current_gate_count": progress.current_gate_count,
                "current_width": progress.current_width,
                "elapsed_seconds": progress.elapsed_seconds,
                "estimated_remaining_seconds": progress.estimated_remaining_seconds,
                "circuits_per_second": progress.circuits_per_second,
                "current_status": progress.current_status,
                "error": progress.error,
            })
        except Exception:
            pass  # Queue full or other issue, skip update
    
    try:
        result = generator.generate(
            width=width,
            gate_count=gate_count,
            gate_set=gate_set,
            max_circuits=max_circuits,
            config=config,
            progress_callback=progress_callback,
        )
        
        return {
            "success": result.success,
            "total_circuits": result.total_circuits,
            "new_circuits": result.new_circuits,
            "duplicates": result.duplicates,
            "total_seconds": result.total_seconds,
            "error": result.error,
            "circuits": result.circuits or [],
        }
        
    except Exception as e:
        logger.exception("Generation failed in worker")
        return {
            "success": False,
            "error": str(e),
            "circuits": [],
        }
