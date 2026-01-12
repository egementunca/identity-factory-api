"""
Generator API endpoints.
Provides REST API for managing and running circuit generators.
"""

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ..database import CircuitDatabase, CircuitRecord, DimGroupRecord
from ..generators import GenerationProgress, GeneratorStatus, get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generators", tags=["generators"])

# Database path for circuit storage
_db_path = Path.home() / ".identity_factory" / "circuits.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)

# In-memory storage for run tracking
_active_runs: Dict[str, Dict[str, Any]] = {}
_run_lock = threading.Lock()


# === Request/Response Models ===


class GeneratorInfoResponse(BaseModel):
    """Generator information."""

    name: str
    display_name: str
    description: str
    gate_sets: List[str]
    supports_pause: bool
    supports_incremental: bool
    config_schema: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    """Request to start circuit generation."""

    generator_name: str = Field(..., description="Name of generator to use")
    width: int = Field(..., ge=2, le=10, description="Number of wires/qubits")
    gate_count: int = Field(..., ge=1, le=100, description="Number of gates")
    gate_set: str = Field("mcx", description="Gate set to use")
    max_circuits: Optional[int] = Field(None, description="Max circuits to generate")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Generator-specific config"
    )


class RunStatusResponse(BaseModel):
    """Current status of a generation run."""

    run_id: str
    generator_name: str
    status: str
    circuits_found: int = 0
    circuits_stored: int = 0
    duplicates_skipped: int = 0
    current_gate_count: Optional[int] = None
    current_width: Optional[int] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    circuits_per_second: float = 0.0
    current_status: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class GenerateResultResponse(BaseModel):
    """Result of completed generation."""

    success: bool
    run_id: str
    generator_name: str
    total_circuits: int = 0
    new_circuits: int = 0
    duplicates: int = 0
    width: Optional[int] = None
    gate_count: Optional[int] = None
    gate_set: str = "mcx"
    total_seconds: float = 0.0
    error: Optional[str] = None


# === Background task for generation ===


def run_generation_task(
    run_id: str,
    generator_name: str,
    width: int,
    gate_count: int,
    gate_set: str,
    max_circuits: Optional[int],
    config: Optional[Dict[str, Any]],
):
    """Background task to run generation."""
    registry = get_registry()
    generator = registry.get_generator(generator_name)

    if not generator:
        with _run_lock:
            _active_runs[run_id]["status"] = "failed"
            _active_runs[run_id]["error"] = f"Generator '{generator_name}' not found"
        return

    def progress_callback(progress: GenerationProgress):
        with _run_lock:
            if run_id in _active_runs:
                _active_runs[run_id].update(
                    {
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
                    }
                )

    try:
        result = generator.generate(
            width=width,
            gate_count=gate_count,
            gate_set=gate_set,
            max_circuits=max_circuits,
            config=config,
            progress_callback=progress_callback,
        )

        # Store circuits in database if any were generated
        stored_count = 0
        stored_ids = []
        if result.success and result.circuits:
            try:
                db = CircuitDatabase(str(_db_path))

                # Get or create dimension group
                dim_group = db.get_dim_group(width, gate_count)
                if not dim_group:
                    dim_group = DimGroupRecord(
                        id=None,
                        width=width,
                        gate_count=gate_count,
                        circuit_count=0,
                        is_processed=False,
                    )
                    db.store_dim_group(dim_group)
                    dim_group = db.get_dim_group(width, gate_count)

                # Store each circuit
                for circuit_data in result.circuits:
                    circuit_width = circuit_data.get("width", width)
                    circuit_gates = circuit_data.get("gates", [])
                    circuit_perm = circuit_data.get("permutation", [])

                    # Convert gates to tuple format if needed
                    gates_list = []
                    for g in circuit_gates:
                        if isinstance(g, (list, tuple)):
                            gates_list.append(tuple(g))
                        else:
                            gates_list.append(g)

                    record = CircuitRecord(
                        id=None,
                        width=circuit_width,
                        gate_count=len(gates_list),
                        gates=gates_list,
                        permutation=(
                            circuit_perm
                            if circuit_perm
                            else list(range(2**circuit_width))
                        ),
                        dim_group_id=dim_group.id if dim_group else None,
                    )

                    stored_id = db.store_circuit(record)
                    if stored_id:
                        stored_ids.append(stored_id)
                        stored_count += 1
                        if dim_group:
                            db.add_circuit_to_dim_group(dim_group.id, stored_id)

                logger.info(f"Stored {stored_count} circuits in database")

            except Exception as db_error:
                logger.error(f"Failed to store circuits: {db_error}")

        with _run_lock:
            if run_id in _active_runs:
                _active_runs[run_id].update(
                    {
                        "status": "completed" if result.success else "failed",
                        "circuits_stored": stored_count,
                        "result": {
                            "success": result.success,
                            "total_circuits": result.total_circuits,
                            "new_circuits": stored_count,
                            "duplicates": result.duplicates,
                            "total_seconds": result.total_seconds,
                            "error": result.error,
                            "circuit_ids": stored_ids,
                        },
                        "completed_at": datetime.now().isoformat(),
                    }
                )

    except Exception as e:
        logger.exception(f"Generation run {run_id} failed")
        with _run_lock:
            if run_id in _active_runs:
                _active_runs[run_id].update(
                    {
                        "status": "failed",
                        "error": str(e),
                        "completed_at": datetime.now().isoformat(),
                    }
                )


# === Endpoints ===


@router.get("/", response_model=List[GeneratorInfoResponse])
async def list_generators():
    """List all available generators."""
    registry = get_registry()
    generators = registry.list_generators()
    return [
        GeneratorInfoResponse(
            name=g.name,
            display_name=g.display_name,
            description=g.description,
            gate_sets=g.gate_sets,
            supports_pause=g.supports_pause,
            supports_incremental=g.supports_incremental,
            config_schema=g.config_schema,
        )
        for g in generators
    ]


@router.get("/{generator_name}", response_model=GeneratorInfoResponse)
async def get_generator(generator_name: str):
    """Get information about a specific generator."""
    registry = get_registry()
    generator = registry.get_generator(generator_name)

    if not generator:
        raise HTTPException(
            status_code=404, detail=f"Generator '{generator_name}' not found"
        )

    info = generator.get_info()
    return GeneratorInfoResponse(
        name=info.name,
        display_name=info.display_name,
        description=info.description,
        gate_sets=info.gate_sets,
        supports_pause=info.supports_pause,
        supports_incremental=info.supports_incremental,
        config_schema=info.config_schema,
    )


@router.get("/gate-sets/")
async def list_gate_sets():
    """List all supported gate sets."""
    registry = get_registry()
    return {"gate_sets": registry.get_all_gate_sets()}


@router.post("/run", response_model=RunStatusResponse)
async def start_generation(request: GenerateRequest):
    """
    Start a new generation run.

    Returns immediately with a run_id that can be used to track progress.
    """
    registry = get_registry()
    generator = registry.get_generator(request.generator_name)

    if not generator:
        raise HTTPException(
            status_code=404, detail=f"Generator '{request.generator_name}' not found"
        )

    if not generator.supports_gate_set(request.gate_set):
        raise HTTPException(
            status_code=400,
            detail=f"Generator '{request.generator_name}' does not support gate set '{request.gate_set}'",
        )

    # Create run ID
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now()

    # Initialize run tracking
    with _run_lock:
        _active_runs[run_id] = {
            "run_id": run_id,
            "generator_name": request.generator_name,
            "status": "running",
            "circuits_found": 0,
            "circuits_stored": 0,
            "duplicates_skipped": 0,
            "current_gate_count": request.gate_count,
            "current_width": request.width,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "circuits_per_second": 0.0,
            "current_status": "Starting...",
            "error": None,
            "started_at": started_at.isoformat(),
            "completed_at": None,
            "result": None,
        }

    # Start in a separate thread for reliable execution
    import threading

    thread = threading.Thread(
        target=run_generation_task,
        kwargs={
            "run_id": run_id,
            "generator_name": request.generator_name,
            "width": request.width,
            "gate_count": request.gate_count,
            "gate_set": request.gate_set,
            "max_circuits": request.max_circuits,
            "config": request.config,
        },
        daemon=True,
    )
    thread.start()
    logger.info(f"Started generation thread for run {run_id}")

    return RunStatusResponse(
        run_id=run_id,
        generator_name=request.generator_name,
        status="running",
        current_gate_count=request.gate_count,
        current_width=request.width,
        current_status="Starting...",
        started_at=started_at.isoformat(),
    )


@router.get("/runs/", response_model=List[RunStatusResponse])
async def list_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    generator_name: Optional[str] = Query(None, description="Filter by generator"),
):
    """List all generation runs."""
    with _run_lock:
        runs = list(_active_runs.values())

    # Apply filters
    if status:
        runs = [r for r in runs if r.get("status") == status]
    if generator_name:
        runs = [r for r in runs if r.get("generator_name") == generator_name]

    return [
        RunStatusResponse(
            run_id=r["run_id"],
            generator_name=r["generator_name"],
            status=r["status"],
            circuits_found=r.get("circuits_found", 0),
            circuits_stored=r.get("circuits_stored", 0),
            duplicates_skipped=r.get("duplicates_skipped", 0),
            current_gate_count=r.get("current_gate_count"),
            current_width=r.get("current_width"),
            elapsed_seconds=r.get("elapsed_seconds", 0.0),
            estimated_remaining_seconds=r.get("estimated_remaining_seconds"),
            circuits_per_second=r.get("circuits_per_second", 0.0),
            current_status=r.get("current_status", ""),
            error=r.get("error"),
            started_at=r.get("started_at"),
            completed_at=r.get("completed_at"),
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get status of a specific generation run."""
    with _run_lock:
        run = _active_runs.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return RunStatusResponse(
        run_id=run["run_id"],
        generator_name=run["generator_name"],
        status=run["status"],
        circuits_found=run.get("circuits_found", 0),
        circuits_stored=run.get("circuits_stored", 0),
        duplicates_skipped=run.get("duplicates_skipped", 0),
        current_gate_count=run.get("current_gate_count"),
        current_width=run.get("current_width"),
        elapsed_seconds=run.get("elapsed_seconds", 0.0),
        estimated_remaining_seconds=run.get("estimated_remaining_seconds"),
        circuits_per_second=run.get("circuits_per_second", 0.0),
        current_status=run.get("current_status", ""),
        error=run.get("error"),
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
    )


@router.get("/runs/{run_id}/result", response_model=GenerateResultResponse)
async def get_run_result(run_id: str):
    """Get result of a completed generation run."""
    with _run_lock:
        run = _active_runs.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if run["status"] not in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Run is still in progress")

    result = run.get("result", {})

    return GenerateResultResponse(
        success=result.get("success", False),
        run_id=run_id,
        generator_name=run["generator_name"],
        total_circuits=result.get("total_circuits", 0),
        new_circuits=result.get("new_circuits", 0),
        duplicates=result.get("duplicates", 0),
        width=run.get("current_width"),
        gate_count=run.get("current_gate_count"),
        total_seconds=result.get("total_seconds", 0.0),
        error=result.get("error"),
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel a running generation."""
    with _run_lock:
        run = _active_runs.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if run["status"] != "running":
        raise HTTPException(status_code=400, detail="Run is not active")

    # Get generator and request cancellation
    registry = get_registry()
    generator = registry.get_generator(run["generator_name"])

    if generator:
        generator.cancel()
        with _run_lock:
            _active_runs[run_id]["status"] = "cancelled"
            _active_runs[run_id]["current_status"] = "Cancellation requested"

    return {"message": "Cancellation requested", "run_id": run_id}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a run from history (only completed/failed/cancelled runs)."""
    with _run_lock:
        run = _active_runs.get(run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        if run["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot delete running task")

        del _active_runs[run_id]

    return {"message": "Run deleted", "run_id": run_id}
