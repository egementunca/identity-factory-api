"""
Pydantic models for Local Mixing API endpoints.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class LoadCircuitRequest(BaseModel):
    """Request to load and parse a circuit string."""

    circuit_string: str = Field(
        ..., description="Circuit in format '[0,1,2] [1,2,0] ...'"
    )
    num_wires: int = Field(
        ..., ge=3, le=64, description="Number of wires in the circuit"
    )


class LoadCircuitResponse(BaseModel):
    """Response with parsed circuit information."""

    gates: List[Tuple[int, int, int]]
    gate_count: int
    num_wires: int
    permutation: List[int]


class CanonicalizeRequest(BaseModel):
    """Request to canonicalize a circuit."""

    circuit_string: str = Field(
        ..., description="Circuit in format '[0,1,2] [1,2,0] ...'"
    )
    num_wires: int = Field(..., ge=3, le=64)


class CanonicalizeResponse(BaseModel):
    """Response with canonicalization results."""

    original_gates: List[Tuple[int, int, int]]
    canonical_gates: List[Tuple[int, int, int]]
    original_count: int
    canonical_count: int
    gates_removed: int
    trace: Optional[List[Dict[str, Any]]] = None


class InflatePreviewRequest(BaseModel):
    """Request to preview inflation (kneading) step."""

    circuit_string: str = Field(
        ..., description="Circuit in format '[0,1,2] [1,2,0] ...'"
    )
    num_wires: int = Field(..., ge=3, le=64)
    sample_count: int = Field(5, ge=1, le=20, description="Number of gates to sample")
    random_id_size_range: Tuple[int, int] = Field(
        (50, 100), description="Range for random identity size"
    )


class InflateSample(BaseModel):
    """Sample of a single gate's inflation."""

    gate_index: int
    original_gate: Tuple[int, int, int]
    inflated_size: int
    r_size: int
    compressed_size: Optional[int] = None


class InflatePreviewResponse(BaseModel):
    """Response with inflation preview."""

    original_gate_count: int
    samples: List[InflateSample]
    estimated_total_inflated_size: int
    estimated_compression_ratio: Optional[float] = None


class HeatmapRequest(BaseModel):
    """Request to generate a heatmap between two circuits."""

    circuit_one: str = Field(..., description="First circuit string")
    circuit_two: str = Field(..., description="Second circuit string")
    num_wires: int = Field(..., ge=3, le=64)
    num_inputs: int = Field(100, ge=10, le=10000, description="Random inputs to test")
    x_slice: Optional[Tuple[int, int]] = Field(
        None, description="Optional (start, end) slice for circuit one"
    )
    y_slice: Optional[Tuple[int, int]] = Field(
        None, description="Optional (start, end) slice for circuit two"
    )
    canonicalize: bool = Field(True, description="Canonicalize circuits before heatmap")


class HeatmapResponse(BaseModel):
    """Response with heatmap data."""

    heatmap_data: List[List[float]]  # 2D array [x][y] = overlap value
    x_size: int
    y_size: int
    mean_overlap: float
    std_dev: float
    circuit_one_size: int
    circuit_two_size: int


class RandomCircuitRequest(BaseModel):
    """Request to generate a random circuit."""

    num_wires: int = Field(..., ge=3, le=64)
    num_gates: int = Field(..., ge=1, le=1000)


class RandomCircuitResponse(BaseModel):
    """Response with random circuit."""

    circuit_string: str
    gates: List[Tuple[int, int, int]]
    permutation: List[int]


class ExampleCircuit(BaseModel):
    """An example circuit for testing."""

    name: str
    description: str
    circuit_string: str
    num_wires: int


class ExamplesResponse(BaseModel):
    """Response with example circuits."""

    examples: List[ExampleCircuit]
