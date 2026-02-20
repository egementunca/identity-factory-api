"""
API endpoints for the Identity Circuit Factory.
Updated for simplified structure focusing on seed generation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ..database import CircuitDatabase
from ..debris_cancellation import DebrisCancellationAnalyzer
from ..factory_manager import FactoryConfig, IdentityFactory
from ..seed_generator import SeedGenerator
from ..unroller import CircuitUnroller
from .models import (
    AdvancedSearchRequest,
    BatchCircuitRequest,
    BatchGenerationResultResponse,
    CircuitRequest,
    CircuitResponse,
    CircuitsByCompositionResponse,
    CircuitVisualizationResponse,
    DebrisAnalyzeResponse,
    DimGroupResponse,
    ErrorResponse,
    FactoryStatsResponse,
    GenerationResultResponse,
    GenerationStatsResponse,
    HealthResponse,
    PaginatedResponse,
    SearchParams,
    UnrollRequest,
    UnrollResponse,
)
from .db_paths import resolve_identity_db_path

logger = logging.getLogger(__name__)

# Database path - shared default (cluster if present, else ~/.identity_factory)
_db_path = resolve_identity_db_path()
_db_path.parent.mkdir(parents=True, exist_ok=True)

# Global instances
_database: Optional[CircuitDatabase] = None
_seed_generator: Optional[SeedGenerator] = None
_unroller: Optional[CircuitUnroller] = None


def get_database() -> CircuitDatabase:
    """Get or create the global database instance."""
    global _database
    if _database is None:
        _database = CircuitDatabase(str(_db_path))
    return _database


def get_seed_generator() -> SeedGenerator:
    """Get or create the global seed generator instance."""
    global _seed_generator
    if _seed_generator is None:
        database = get_database()
        _seed_generator = SeedGenerator(database)
    return _seed_generator


def get_unroller() -> CircuitUnroller:
    """Get or create the global unroller instance."""
    global _unroller
    if _unroller is None:
        database = get_database()
        _unroller = CircuitUnroller(database)
    return _unroller


# Create router
router = APIRouter()


@router.post("/generate", response_model=GenerationResultResponse)
async def generate_circuit(
    request: CircuitRequest,
    database: CircuitDatabase = Depends(get_database),
    seed_generator: SeedGenerator = Depends(get_seed_generator),
) -> GenerationResultResponse:
    """
    Generate an identity circuit for specified dimensions.

    Args:
        request: Generation parameters
        database: Database instance
        seed_generator: Seed generator instance

    Returns:
        Generation result with circuit information
    """
    try:
        logger.info(
            f"API: Generating circuit (width={request.width}, forward_length={request.forward_length})"
        )

        # Generate circuit using simplified seed generator
        result = seed_generator.generate_seed(
            width=request.width,
            forward_length=request.forward_length,
            max_attempts=request.max_attempts or 10,
        )

        if not result.success:
            raise HTTPException(
                status_code=500, detail=f"Generation failed: {result.error_message}"
            )

        return GenerationResultResponse(
            success=True,
            circuit_id=result.circuit_id,
            dim_group_id=result.dim_group_id,
            forward_gates=result.forward_gates,
            inverse_gates=result.inverse_gates,
            identity_gates=result.identity_gates,
            gate_composition=result.gate_composition,
            total_time=(
                result.metrics.get("generation_time", 0.0) if result.metrics else 0.0
            ),
            metrics=result.metrics,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate", response_model=BatchGenerationResultResponse)
async def batch_generate(
    request: BatchCircuitRequest,
    background_tasks: BackgroundTasks,
    database: CircuitDatabase = Depends(get_database),
    seed_generator: SeedGenerator = Depends(get_seed_generator),
) -> BatchGenerationResultResponse:
    """
    Generate multiple circuits for different dimensions.

    Args:
        request: Batch generation parameters
        background_tasks: FastAPI background tasks
        database: Database instance
        seed_generator: Seed generator instance

    Returns:
        Batch generation results
    """
    try:
        logger.info(f"API: Batch generating {len(request.dimensions)} circuits")
        start_time = time.time()

        results = []
        successful = 0
        failed = 0

        for width, forward_length in request.dimensions:
            try:
                result = seed_generator.generate_seed(
                    width=width,
                    forward_length=forward_length,
                    max_attempts=request.max_attempts or 10,
                )

                generation_result = GenerationResultResponse(
                    success=result.success,
                    circuit_id=result.circuit_id,
                    dim_group_id=result.dim_group_id,
                    forward_gates=result.forward_gates,
                    inverse_gates=result.inverse_gates,
                    identity_gates=result.identity_gates,
                    gate_composition=result.gate_composition,
                    total_time=(
                        result.metrics.get("generation_time", 0.0)
                        if result.metrics
                        else 0.0
                    ),
                    error_message=result.error_message,
                    metrics=result.metrics,
                )

                results.append(generation_result)

                if result.success:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(
                    f"Failed to generate circuit for ({width}, {forward_length}): {e}"
                )
                results.append(
                    GenerationResultResponse(
                        success=False, total_time=0.0, error_message=str(e)
                    )
                )
                failed += 1

        total_time = time.time() - start_time

        return BatchGenerationResultResponse(
            total_requested=len(request.dimensions),
            successful_generations=successful,
            failed_generations=failed,
            results=results,
            total_time=total_time,
        )

    except Exception as e:
        logger.error(f"API batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits", response_model=PaginatedResponse)  # TODO finish filtering
async def search_circuits(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    width: Optional[int] = Query(None, ge=1, le=10, description="Filter by width"),
    gate_count: Optional[int] = Query(
        None, ge=1, le=100, description="Filter by gate count"
    ),
    is_representative: Optional[bool] = Query(
        None, description="Filter by representative status"
    ),
    gate_composition: Optional[str] = Query(
        None, description="Filter by gate composition (e.g., '2,1,0')"
    ),
    sort_by: Optional[str] = Query("id", description="Sort field"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order"),
    database: CircuitDatabase = Depends(get_database),
) -> PaginatedResponse:
    """
    Search circuits with comprehensive filtering and pagination.

    Args:
        page: Page number
        size: Page size
        width: Filter by circuit width
        gate_count: Filter by gate count
        is_representative: Filter by representative status
        gate_composition: Filter by gate composition (format: "x,cx,ccx")
        sort_by: Sort field
        sort_order: Sort order (asc/desc)
        database: Database instance

    Returns:
        Paginated circuit results
    """
    try:
        # Parse gate composition filter if provided
        composition_filter = None
        if gate_composition:
            try:
                parts = gate_composition.split(",")
                if len(parts) == 3:
                    composition_filter = tuple(int(p) for p in parts)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid gate composition format. Use 'x,cx,ccx'",
                )

        # Get all circuits (we'll implement proper filtering later)
        all_circuits = database.get_all_circuits()

        # Apply filters
        filtered_circuits = []
        for circuit in all_circuits:
            # Width filter
            if width is not None and circuit.width != width:
                continue

            # Gate count filter
            if gate_count is not None and circuit.gate_count != gate_count:
                continue

            # Representative status filter
            if is_representative is not None:
                circuit_is_rep = circuit.representative_id == circuit.id
                if is_representative != circuit_is_rep:
                    continue

            # Gate composition filter
            if composition_filter is not None:
                circuit_comp = circuit.get_gate_composition()
                if circuit_comp != composition_filter:
                    continue

            filtered_circuits.append(circuit)

        # Sort circuits
        reverse = sort_order == "desc"
        if sort_by == "id":
            filtered_circuits.sort(key=lambda c: c.id, reverse=reverse)
        elif sort_by == "width":
            filtered_circuits.sort(key=lambda c: c.width, reverse=reverse)
        elif sort_by == "gate_count":
            filtered_circuits.sort(key=lambda c: c.gate_count, reverse=reverse)

        # Calculate pagination
        total = len(filtered_circuits)
        start_idx = (page - 1) * size
        end_idx = start_idx + size

        page_circuits = filtered_circuits[start_idx:end_idx]

        # Convert to response format
        circuit_responses = [
            CircuitResponse.from_circuit_record(c) for c in page_circuits
        ]

        return PaginatedResponse(
            items=circuit_responses,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API search circuits failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/circuits/advanced-search", response_model=PaginatedResponse
)  # TODO finish filtering
async def advanced_search_circuits(
    search_request: AdvancedSearchRequest,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=1000),
    database: CircuitDatabase = Depends(get_database),
) -> PaginatedResponse:
    """
    Advanced circuit search with complex filtering.

    Args:
        search_request: Advanced search parameters
        page: Page number
        size: Page size
        database: Database instance

    Returns:
        Paginated search results
    """
    try:
        all_circuits = database.get_all_circuits()

        # Apply advanced filters
        filtered_circuits = []
        for circuit in all_circuits:
            # Width range filter
            if search_request.width_range:
                min_w, max_w = search_request.width_range
                if not (min_w <= circuit.width <= max_w):
                    continue

            # Gate count range filter
            if search_request.gate_count_range:
                min_gc, max_gc = search_request.gate_count_range
                if not (min_gc <= circuit.gate_count <= max_gc):
                    continue

            # Has equivalents filter
            if search_request.has_equivalents is not None:
                has_equiv = any(
                    c.representative_id == circuit.id and c.id != circuit.id
                    for c in all_circuits
                )
                if search_request.has_equivalents != has_equiv:
                    continue

            # Gate types filter
            if search_request.gate_types:
                circuit_gate_types = set(gate[0] for gate in circuit.gates)
                required_types = set(search_request.gate_types)
                if not required_types.issubset(circuit_gate_types):
                    continue

            # Gate composition filters
            composition = circuit.get_gate_composition()

            if search_request.min_composition:
                min_x, min_cx, min_ccx = search_request.min_composition
                if not (
                    composition[0] >= min_x
                    and composition[1] >= min_cx
                    and composition[2] >= min_ccx
                ):
                    continue

            if search_request.max_composition:
                max_x, max_cx, max_ccx = search_request.max_composition
                if not (
                    composition[0] <= max_x
                    and composition[1] <= max_cx
                    and composition[2] <= max_ccx
                ):
                    continue

            filtered_circuits.append(circuit)

        # Pagination
        total = len(filtered_circuits)
        start_idx = (page - 1) * size
        end_idx = start_idx + size

        page_circuits = filtered_circuits[start_idx:end_idx]
        circuit_responses = [
            CircuitResponse.from_circuit_record(c) for c in page_circuits
        ]

        return PaginatedResponse(
            items=circuit_responses,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size,
        )

    except Exception as e:
        logger.error(f"API advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dim-groups", response_model=List[DimGroupResponse])
async def list_dimension_groups(
    width: Optional[int] = Query(None, description="Filter by width"),
    gate_count: Optional[int] = Query(None, description="Filter by gate count"),
    processed_only: bool = Query(False, description="Only processed groups"),
    database: CircuitDatabase = Depends(get_database),
) -> List[DimGroupResponse]:
    """
    List dimension groups with optional filtering.

    Args:
        width: Optional width filter
        gate_count: Optional gate count filter
        processed_only: Only return processed groups
        database: Database instance

    Returns:
        List of dimension groups
    """
    try:
        # Get dimension groups with filtering
        if width and gate_count:
            dim_group = database.get_dim_group(width, gate_count)
            dim_groups = [dim_group] if dim_group else []
        elif width:
            dim_groups = [
                dg for dg in database.get_all_dim_groups() if dg.width == width
            ]
        else:
            dim_groups = database.get_all_dim_groups()

        # Filter processed if requested
        if processed_only:
            dim_groups = [dg for dg in dim_groups if dg.is_processed]

        # Convert to response format
        responses = []
        for dg in dim_groups:
            # Get representatives and equivalents counts
            all_circuits = database.get_circuits_in_dim_group(dg.id)
            representatives = [c for c in all_circuits if c.representative_id == c.id]

            responses.append(
                DimGroupResponse(
                    id=dg.id,
                    width=dg.width,
                    gate_count=dg.gate_count,
                    circuit_count=dg.circuit_count,
                    representative_count=len(representatives),
                    is_processed=dg.is_processed,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"API list dimension groups failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dim-groups/{dim_group_id}", response_model=DimGroupResponse)
async def get_dimension_group(
    dim_group_id: int, database: CircuitDatabase = Depends(get_database)
) -> DimGroupResponse:
    """
    Get detailed information about a specific dimension group.

    Args:
        dim_group_id: Dimension group ID
        database: Database instance

    Returns:
        Dimension group details
    """
    try:
        # Get dimension group
        dim_group = database.get_dim_group_by_id(dim_group_id)
        if not dim_group:
            raise HTTPException(status_code=404, detail="Dimension group not found")

        # Get circuit counts
        all_circuits = database.get_circuits_in_dim_group(dim_group_id)
        representatives = [c for c in all_circuits if c.representative_id == c.id]

        return DimGroupResponse(
            id=dim_group.id,
            width=dim_group.width,
            gate_count=dim_group.gate_count,
            circuit_count=dim_group.circuit_count,
            representative_count=len(representatives),
            is_processed=dim_group.is_processed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API get dimension group failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits/{circuit_id}", response_model=CircuitResponse)
async def get_circuit(
    circuit_id: int, database: CircuitDatabase = Depends(get_database)
) -> CircuitResponse:
    """
    Get detailed information about a specific circuit.

    Args:
        circuit_id: Circuit ID
        database: Database instance

    Returns:
        Circuit details
    """
    try:
        circuit = database.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit not found")

        return CircuitResponse.from_circuit_record(circuit)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API get circuit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuits/{circuit_id}/unroll", response_model=UnrollResponse)
async def unroll_circuit(
    circuit_id: int,
    request: UnrollRequest = None,
    database: CircuitDatabase = Depends(get_database),
    unroller: CircuitUnroller = Depends(get_unroller),
) -> UnrollResponse:
    """
    Unroll a circuit to generate equivalent circuits.

    Uses comprehensive unrolling including swap space exploration,
    rotations, reverse, and permutations.

    Args:
        circuit_id: Circuit ID to unroll
        request: Optional unroll parameters (max_equivalents)
        database: Database instance
        unroller: CircuitUnroller instance

    Returns:
        Unroll result with equivalent circuits
    """
    try:
        # Get the circuit
        circuit = database.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit not found")

        # Get max_equivalents from request or use default
        max_equivalents = 100
        if request:
            max_equivalents = request.max_equivalents

        logger.info(
            f"Unrolling circuit {circuit_id} with max_equivalents={max_equivalents}"
        )

        # Perform unrolling
        result = unroller.unroll_circuit(circuit, max_equivalents)

        return UnrollResponse(
            success=result.get("success", False),
            circuit_id=circuit_id,
            equivalents=result.get("equivalents", []),
            total_generated=result.get("total_generated", 0),
            unique_equivalents=result.get("unique_equivalents", 0),
            stored_equivalents=result.get("stored_equivalents", 0),
            fully_unrolled=result.get("fully_unrolled", False),
            error=result.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API unroll circuit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/circuits/{circuit_id}/analyze-debris", response_model=DebrisAnalyzeResponse
)
async def analyze_debris_cancellation(
    circuit_id: int, database: CircuitDatabase = Depends(get_database)
) -> DebrisAnalyzeResponse:
    """
    Analyze a circuit for debris cancellation opportunities.

    Finds debris gates that can be inserted to enable new cancellation paths.
    Returns non-triviality score indicating how hard the circuit is to simplify.

    Args:
        circuit_id: Circuit ID to analyze
        database: Database instance

    Returns:
        Debris analysis result with improvement details
    """
    try:
        # Get the circuit
        circuit_record = database.get_circuit(circuit_id)
        if not circuit_record:
            raise HTTPException(status_code=404, detail="Circuit not found")

        logger.info(f"Analyzing debris cancellation for circuit {circuit_id}")

        # Convert to sat_revsynth Circuit
        from circuit.circuit import Circuit

        circuit = Circuit(circuit_record.width)

        for gate in circuit_record.gates:
            if isinstance(gate, (list, tuple)):
                # NCT format: ('X', t) or ('CX', c, t) or ('CCX', c1, c2, t)
                if len(gate) >= 2 and isinstance(gate[0], str):
                    gate_type = gate[0]
                    if gate_type == "X":
                        circuit = circuit.x(gate[1])
                    elif gate_type == "CX":
                        circuit = circuit.cx(gate[1], gate[2])
                    elif gate_type == "CCX":
                        circuit = circuit.mcx([gate[1], gate[2]], gate[3])
                # Controls/target format: ([c1, c2], t)
                elif len(gate) == 2 and isinstance(gate[0], (list, tuple)):
                    controls, target = gate
                    if len(controls) == 0:
                        circuit = circuit.x(target)
                    elif len(controls) == 1:
                        circuit = circuit.cx(controls[0], target)
                    else:
                        circuit = circuit.mcx(list(controls), target)
                # ECA57 format: [c1, c2, t]
                elif len(gate) == 3 and all(isinstance(x, int) for x in gate):
                    ctrl1, ctrl2, target = gate
                    circuit = circuit.mcx([ctrl1, ctrl2], target)

        original_gate_count = len(circuit.gates())

        # Create analyzer and analyze
        analyzer = DebrisCancellationAnalyzer(max_debris_gates=5)
        cancellation_path = analyzer.analyze_circuit(circuit)

        # Compute non-triviality score
        non_triviality_score = analyzer.compute_non_triviality_score(circuit)

        if (
            cancellation_path
            and cancellation_path.final_gate_count < original_gate_count
        ):
            return DebrisAnalyzeResponse(
                success=True,
                circuit_id=circuit_id,
                original_gate_count=original_gate_count,
                final_gate_count=cancellation_path.final_gate_count,
                improvement_found=True,
                non_triviality_score=non_triviality_score,
                debris_gates=cancellation_path.debris_gates,
                cancellation_path=cancellation_path.path,
            )
        else:
            return DebrisAnalyzeResponse(
                success=True,
                circuit_id=circuit_id,
                original_gate_count=original_gate_count,
                final_gate_count=original_gate_count,
                improvement_found=False,
                non_triviality_score=non_triviality_score,
                debris_gates=[],
                cancellation_path=[],
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API analyze debris failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dim-groups/{dim_group_id}/circuits", response_model=List[CircuitResponse])
async def list_circuits_in_dim_group(
    dim_group_id: int,
    representatives_only: bool = Query(
        False, description="Only return representative circuits"
    ),
    database: CircuitDatabase = Depends(get_database),
) -> List[CircuitResponse]:
    """
    List circuits in a dimension group.

    Args:
        dim_group_id: Dimension group ID
        representatives_only: Whether to return only representatives
        database: Database instance

    Returns:
        List of circuits
    """
    try:
        if representatives_only:
            circuits = database.get_representatives_in_dim_group(dim_group_id)
        else:
            circuits = database.get_circuits_in_dim_group(dim_group_id)

        return [CircuitResponse.from_circuit_record(circuit) for circuit in circuits]

    except Exception as e:
        logger.error(f"API list circuits in dim group failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/dim-groups/{dim_group_id}/compositions",
    response_model=List[CircuitsByCompositionResponse],
)
async def get_circuits_by_composition(
    dim_group_id: int, database: CircuitDatabase = Depends(get_database)
) -> List[CircuitsByCompositionResponse]:
    """
    Get circuits grouped by gate composition within a dimension group.

    Args:
        dim_group_id: Dimension group ID
        database: Database instance

    Returns:
        Circuits grouped by gate composition
    """
    try:
        all_circuits = database.get_circuits_in_dim_group(dim_group_id)

        # Group circuits by gate composition
        compositions = {}
        for circuit in all_circuits:
            comp = circuit.get_gate_composition()
            if comp not in compositions:
                compositions[comp] = []
            compositions[comp].append(circuit)

        # Convert to response format
        responses = []
        for comp, circuits in compositions.items():
            responses.append(
                CircuitsByCompositionResponse(
                    gate_composition=comp,
                    circuits=[CircuitResponse.from_circuit_record(c) for c in circuits],
                    total_count=len(circuits),
                )
            )

        return responses

    except Exception as e:
        logger.error(f"API get circuits by composition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/circuits/{circuit_id}/visualization", response_model=CircuitVisualizationResponse
)
async def get_circuit_visualization(
    circuit_id: int, database: CircuitDatabase = Depends(get_database)
) -> CircuitVisualizationResponse:
    """
    Get visualization data for a circuit.

    Args:
        circuit_id: Circuit ID
        database: Database instance

    Returns:
        Circuit visualization data
    """
    try:
        circuit = database.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit not found")

        # Generate ASCII diagram
        ascii_diagram = _generate_ascii_diagram(circuit.gates, circuit.width)

        # Generate gate descriptions
        gate_descriptions = []
        for i, gate in enumerate(circuit.gates):
            desc = _describe_gate(gate, i)
            gate_descriptions.append(desc)

        # Generate permutation table
        permutation_table = _generate_permutation_table(
            circuit.permutation, circuit.width
        )

        return CircuitVisualizationResponse(
            circuit_id=circuit_id,
            ascii_diagram=ascii_diagram,
            gate_descriptions=gate_descriptions,
            permutation_table=permutation_table,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API get circuit visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=FactoryStatsResponse)
async def get_stats(
    database: CircuitDatabase = Depends(get_database),
) -> FactoryStatsResponse:
    """
    Get factory statistics.

    Args:
        database: Database instance

    Returns:
        Factory statistics
    """
    try:
        stats = database.get_database_stats()

        return FactoryStatsResponse(
            total_circuits=stats["total_circuits"],
            total_dim_groups=stats["total_dim_groups"],
            total_representatives=stats["total_representatives"],
            total_equivalents=stats["total_equivalents"],
            pending_jobs=stats["pending_jobs"],
        )

    except Exception as e:
        logger.error(f"API get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generator/stats", response_model=GenerationStatsResponse)
async def get_generation_stats(
    seed_generator: SeedGenerator = Depends(get_seed_generator),
) -> GenerationStatsResponse:
    """
    Get seed generation statistics.

    Args:
        seed_generator: Seed generator instance

    Returns:
        Generation statistics
    """
    try:
        stats = seed_generator.get_generation_stats()

        return GenerationStatsResponse(
            total_attempts=stats["total_attempts"],
            successful_generations=stats["successful_generations"],
            failed_generations=stats["failed_generations"],
            success_rate_percent=stats["success_rate_percent"],
            total_generation_time=stats["total_generation_time"],
            average_generation_time=stats["average_generation_time"],
        )

    except Exception as e:
        logger.error(f"API get generation stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Test database connection
        database = get_database()
        database_connected = True
        try:
            database.get_database_stats()
        except:
            database_connected = False

        # Test SAT solver availability
        sat_solver_available = True
        try:
            from synthesizers.circuit_synthesizer import CircuitSynthesizer
        except:
            sat_solver_available = False

        return HealthResponse(
            status=(
                "healthy" if database_connected and sat_solver_available else "degraded"
            ),
            timestamp=datetime.now(),
            version="1.0.0",
            database_connected=database_connected,
            sat_solver_available=sat_solver_available,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions


def _generate_ascii_diagram(gates: List[Tuple], width: int) -> str:
    """Generate ASCII diagram for a circuit."""
    try:
        from circuit.circuit import Circuit

        circuit = Circuit(width)
        for gate in gates:
            if gate[0] == "X":
                circuit.x(gate[1])
            elif gate[0] == "CX":
                circuit.cx(gate[1], gate[2])
            elif gate[0] == "CCX":
                circuit.mcx([gate[1], gate[2]], gate[3])
            else:
                # Handle unknown gate types gracefully
                logger.warning(f"Unknown gate type: {gate}")

        # Get the string representation with proper formatting
        diagram = str(circuit)

        # Clean up the diagram - ensure proper spacing and formatting
        if diagram and len(diagram.strip()) > 10:
            # Remove any extra whitespace and ensure consistent formatting
            lines = diagram.split("\n")
            # Filter out empty lines and normalize spacing
            cleaned_lines = []
            for line in lines:
                if line.strip():  # Only include non-empty lines
                    cleaned_lines.append(line.rstrip())  # Remove trailing whitespace

            # Join with proper newlines
            diagram = "\n".join(cleaned_lines)

            # Ensure the diagram ends with a newline for proper display
            if not diagram.endswith("\n"):
                diagram += "\n"

            return diagram

        # Fallback if diagram generation fails
        fallback = f"Circuit with {len(gates)} gates on {width} qubits:\n"
        for i, gate in enumerate(gates):
            fallback += f"  {i+1}. {gate}\n"
        return fallback

    except Exception as e:
        # Provide detailed error information and fallback
        error_msg = f"Error generating circuit diagram: {e}\n\n"
        error_msg += f"Circuit details:\n"
        error_msg += f"  Width: {width} qubits\n"
        error_msg += f"  Gates: {len(gates)}\n"
        for i, gate in enumerate(gates):
            error_msg += f"    {i+1}. {gate}\n"
        return error_msg


def _describe_gate(gate: Tuple, index: int) -> str:
    """Generate human-readable description of a gate."""
    if gate[0] == "X":
        return f"Gate {index + 1}: X (NOT) on qubit {gate[1]}"
    elif gate[0] == "CX":
        return f"Gate {index + 1}: CX (CNOT) from qubit {gate[1]} to qubit {gate[2]}"
    elif gate[0] == "CCX":
        return f"Gate {index + 1}: CCX (Toffoli) with controls {gate[1]}, {gate[2]} and target {gate[3]}"
    else:
        return f"Gate {index + 1}: {gate}"


def _generate_permutation_table(permutation: List[int], width: int) -> List[List[int]]:
    """Generate permutation table showing input -> output mapping."""
    table = []
    for i in range(2**width):
        input_binary = [int(b) for b in format(i, f"0{width}b")]
        output_index = permutation[i]
        output_binary = [int(b) for b in format(output_index, f"0{width}b")]
        # Format: [In#, In0, In1, ..., Out#, Out0, Out1, ...] (corrected to show proper permutation)
        table.append([i] + input_binary + [output_index] + output_binary)
    return table
