"""
Pydantic models for API request/response schemas.
Updated for simplified database structure.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Types of jobs that can be processed."""

    SEED_GENERATION = "seed_generation"
    UNROLLING = "unrolling"
    POST_PROCESSING = "post_processing"


class JobStatus(str, Enum):
    """Job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CircuitRequest(BaseModel):
    """Request model for circuit generation."""

    width: int = Field(..., ge=1, le=10, description="Number of qubits")
    forward_length: int = Field(
        ..., ge=1, le=20, description="Number of gates in forward circuit"
    )
    max_inverse_gates: Optional[int] = Field(
        40, ge=1, le=100, description="Maximum inverse gates"
    )
    max_attempts: Optional[int] = Field(
        10, ge=1, le=50, description="Maximum generation attempts"
    )


class BatchCircuitRequest(BaseModel):
    """Request model for batch circuit generation."""

    dimensions: List[Tuple[int, int]] = Field(
        ..., description="List of (width, forward_length) tuples"
    )
    max_inverse_gates: Optional[int] = Field(40, ge=1, le=100)
    max_attempts: Optional[int] = Field(10, ge=1, le=50)


class CircuitResponse(BaseModel):
    """Response model for circuit data."""

    id: int
    width: int
    gate_count: int  # Total length of the identity circuit
    gates: List[Tuple]
    permutation: List[int]
    complexity_walk: Optional[List[int]] = None
    circuit_hash: Optional[str] = None
    dim_group_id: Optional[int] = None
    representative_id: Optional[int] = (
        None  # Points to self if this is a representative
    )
    is_representative: bool = False  # True if representative_id == id

    @classmethod
    def from_circuit_record(cls, circuit_record):
        """Create response from CircuitRecord."""
        return cls(
            id=circuit_record.id,
            width=circuit_record.width,
            gate_count=circuit_record.gate_count,
            gates=circuit_record.gates,
            permutation=circuit_record.permutation,
            complexity_walk=circuit_record.complexity_walk,
            circuit_hash=circuit_record.circuit_hash,
            dim_group_id=circuit_record.dim_group_id,
            representative_id=circuit_record.representative_id,
            is_representative=(circuit_record.representative_id == circuit_record.id),
        )


class DimGroupResponse(BaseModel):
    """Response model for dimension group data."""

    id: int
    width: int
    gate_count: int  # Length/number of gates
    circuit_count: int
    representative_count: int  # Number of representative circuits

    is_processed: bool


class CircuitsByCompositionResponse(BaseModel):
    """Response model for circuits grouped by gate composition."""

    gate_composition: Tuple[int, int, int]  # (NOT, CNOT, CCNOT)
    circuits: List[CircuitResponse]
    total_count: int


class GenerationResultResponse(BaseModel):
    """Response model for generation results."""

    success: bool
    circuit_id: Optional[int] = None
    dim_group_id: Optional[int] = None
    forward_gates: Optional[List[Tuple]] = None
    inverse_gates: Optional[List[Tuple]] = None
    identity_gates: Optional[List[Tuple]] = None
    gate_composition: Optional[Tuple[int, int, int]] = None
    total_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class BatchGenerationResultResponse(BaseModel):
    """Response model for batch generation results."""

    total_requested: int
    successful_generations: int
    failed_generations: int
    results: List[GenerationResultResponse]
    total_time: float


class JobResponse(BaseModel):
    """Response model for job data."""

    id: int
    job_type: str
    status: str
    priority: int
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def from_job_record(cls, job_record):
        """Create response from JobRecord."""
        return cls(
            id=job_record.id,
            job_type=job_record.job_type,
            status=job_record.status,
            priority=job_record.priority,
            parameters=job_record.parameters,
            result=job_record.result,
            error_message=job_record.error_message,
            started_at=job_record.started_at,
            completed_at=job_record.completed_at,
        )


class FactoryStatsResponse(BaseModel):
    """Response model for factory statistics."""

    total_circuits: int
    total_dim_groups: int
    total_representatives: int
    total_equivalents: int
    pending_jobs: int
    generation_time: float = 0.0
    database_size_mb: Optional[float] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(50, ge=1, le=1000, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(
        "asc", pattern="^(asc|desc)$", description="Sort order"
    )


class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: List[Any]
    total: int
    page: int
    size: int
    pages: int


class SearchParams(BaseModel):
    """Search parameters."""

    width: Optional[int] = Field(None, ge=1, le=10, description="Filter by width")
    gate_count: Optional[int] = Field(
        None, ge=1, le=100, description="Filter by gate count"
    )
    is_representative: Optional[bool] = Field(
        None, description="Filter by representative status"
    )
    gate_composition: Optional[str] = Field(
        None, description="Filter by gate composition (e.g., '2,1,0')"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    database_connected: bool
    sat_solver_available: bool


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None


# Additional specialized responses for the frontend


class CircuitVisualizationResponse(BaseModel):
    """Response model for circuit visualization."""

    circuit_id: int
    ascii_diagram: str
    gate_descriptions: List[str]
    permutation_table: List[List[int]]


class DimGroupSummaryResponse(BaseModel):
    """Summary response for dimension group overview."""

    id: int
    width: int
    gate_count: int
    circuit_count: int
    compositions: List[Dict[str, Any]]  # List of gate compositions and their counts


class GenerationStatsResponse(BaseModel):
    """Response for generation statistics."""

    total_attempts: int
    successful_generations: int
    failed_generations: int
    success_rate_percent: float
    total_generation_time: float
    average_generation_time: float


# Request models for advanced operations


class AdvancedSearchRequest(BaseModel):
    """Advanced search request for circuits."""

    width_range: Optional[Tuple[int, int]] = None
    gate_count_range: Optional[Tuple[int, int]] = None
    has_equivalents: Optional[bool] = None
    gate_types: Optional[List[str]] = None  # ["X", "CX", "CCX"]
    min_composition: Optional[Tuple[int, int, int]] = None
    max_composition: Optional[Tuple[int, int, int]] = None


class UnrollRequest(BaseModel):
    """Request for circuit unrolling."""

    max_equivalents: int = Field(
        100, ge=1, le=10000, description="Maximum equivalent circuits to generate"
    )


class UnrollResponse(BaseModel):
    """Response for circuit unrolling."""

    success: bool
    circuit_id: int
    equivalents: List[List[Tuple]] = []  # List of equivalent circuit gate lists
    total_generated: int = 0
    unique_equivalents: int = 0
    stored_equivalents: int = 0
    fully_unrolled: bool = (
        False  # True if all equivalents were found (didn't hit limit)
    )
    error: Optional[str] = None


class DebrisAnalyzeResponse(BaseModel):
    """Response for debris cancellation analysis."""

    success: bool
    circuit_id: int
    original_gate_count: int
    final_gate_count: int = 0
    improvement_found: bool = False
    non_triviality_score: float = 0.0
    debris_gates: List[Tuple] = []
    cancellation_path: List[int] = []
    error: Optional[str] = None


class CompressRequest(BaseModel):
    """Request for circuit compression using rainbow table."""

    trials: int = Field(
        1000, ge=1, le=100000, description="Number of compression trials"
    )


class CompressResponse(BaseModel):
    """Response for circuit compression."""

    success: bool
    circuit_id: int
    original_gate_count: int
    compressed_gate_count: int
    compression_ratio: float = 0.0
    improved: bool = False
    compressed_gates: List[Tuple] = []
    trials_used: int = 0
    error: Optional[str] = None


class IdentityCircuitInfo(BaseModel):
    """Info about a pre-computed identity circuit."""

    gates: List[Tuple]
    gate_count: int
    gate_composition: Optional[Tuple[int, int, int]] = None  # (X, CX, CCX)


class IdentitiesResponse(BaseModel):
    """Response for pre-computed identity circuits."""

    width: int
    gate_count: int
    total_circuits: int
    circuits: List[IdentityCircuitInfo]


class SynthesizeRequest(BaseModel):
    """Request for optimal circuit synthesis."""

    permutation: List[int] = Field(
        ..., description="Target permutation as list of output values"
    )
    max_gates: int = Field(10, ge=1, le=20, description="Maximum gates to try")
    find_all: bool = Field(False, description="Find all minimum-gate solutions")
    max_solutions: int = Field(
        10, ge=1, le=100, description="Max solutions if find_all=True"
    )


class SynthesizeResponse(BaseModel):
    """Response for optimal circuit synthesis."""

    success: bool
    permutation: List[int]
    gate_count: int = 0
    circuits: List[List[Tuple]] = []  # List of gate lists
    total_solutions: int = 0
    attempts: List[int] = []  # Gate counts tried
    error: Optional[str] = None
