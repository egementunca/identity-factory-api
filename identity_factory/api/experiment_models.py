"""
Pydantic models for Experiment API endpoints.

Mirrors the ObfuscationConfig from local_mixing/src/config.rs
and provides experiment presets.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentType(str, Enum):
    """Pre-defined experiment types."""

    EXPANSION = "expansion"
    ANNEALED = "annealed"
    INFLATION_ONLY = "inflation_only"
    SAT_COMPRESSION = "sat_compression"
    CUSTOM = "custom"


class ExperimentStatus(str, Enum):
    """Experiment job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObfuscationStrategy(str, Enum):
    """Obfuscation strategy types."""

    ABBUTTERFLY = "abbutterfly"
    BBUTTERFLY = "bbutterfly"
    BUTTERFLY = "butterfly"


class ObfuscationParams(BaseModel):
    """
    Obfuscation parameters matching local_mixing's ObfuscationConfig.

    See: local_mixing/src/config.rs
    """

    # --- Strategy ---
    strategy: ObfuscationStrategy = Field(
        ObfuscationStrategy.ABBUTTERFLY, description="Obfuscation strategy to use"
    )
    bookendless: bool = Field(
        False, description="Enable bookendless mode (abbutterfly only)"
    )

    # --- Structural Parameters (Mixing) ---
    structure_block_size_min: int = Field(
        10,
        ge=3,
        le=100,
        description="Minimum size of the random identity structure (R⋅R⁻¹)",
    )
    structure_block_size_max: int = Field(
        30, ge=3, le=200, description="Maximum size of the random identity structure"
    )

    # --- Obfuscation Intensity ---
    shooting_count: int = Field(
        1000,
        ge=0,
        le=10_000_000,
        description="Gate reordering passes - randomly moves gates left/right as far as possible without changing circuit semantics (scrambles gate order)",
    )
    shooting_count_inner: int = Field(
        0,
        ge=0,
        le=100_000,
        description="Gate reordering passes inside each butterfly block",
    )
    single_gate_replacements: int = Field(
        500, ge=0, description="Number of single gate replacements before main loop"
    )
    rounds: int = Field(
        3, ge=1, le=100, description="Number of butterfly obfuscation rounds"
    )

    # --- Modes ---
    sat_mode: bool = Field(
        True,
        description="Use SAT solver to find optimal gate sequences during compression (slower but more effective than rainbow table lookup)",
    )
    no_ancilla_mode: bool = Field(
        False,
        description="Disable ancilla expansion - when off, subcircuits are expanded to use extra wires to find equivalent circuits via rainbow table lookup",
    )
    single_gate_mode: bool = Field(
        False, description="Enable single gate replacement pass"
    )
    skip_compression: bool = Field(
        False,
        description="Skip compression entirely - circuit only grows larger (inflation only mode)",
    )

    # --- Compression/Optimization ---
    compression_window_size: int = Field(
        100,
        ge=10,
        le=10000,
        description="Window size for peephole optimization (normal mode)",
    )
    compression_window_size_sat: int = Field(
        10, ge=1, le=100, description="Window size for SAT-based optimization"
    )
    compression_sat_limit: int = Field(
        1000, ge=100, le=100000, description="Timeout/conflict limit for SAT solver"
    )
    final_stability_threshold: int = Field(
        12,
        ge=1,
        le=50,
        description="Number of stable passes required to stop final compression",
    )
    chunk_split_base: int = Field(
        1500, ge=100, description="Denominator for splitting circuit into chunks"
    )


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    # --- Experiment Metadata ---
    name: str = Field(
        ..., min_length=1, max_length=100, description="Experiment name/identifier"
    )
    experiment_type: ExperimentType = Field(
        ExperimentType.CUSTOM, description="Experiment preset type"
    )
    description: Optional[str] = Field(
        None, description="Optional description of the experiment"
    )

    # --- Circuit Parameters ---
    wires: int = Field(8, ge=3, le=64, description="Number of circuit wires")
    initial_gates: int = Field(
        20, ge=1, le=10000, description="Number of gates in the initial circuit"
    )

    # --- Obfuscation Parameters ---
    obfuscation: ObfuscationParams = Field(
        default_factory=ObfuscationParams, description="Obfuscation pipeline parameters"
    )

    # --- Paths (optional) ---
    lmdb_path: Optional[str] = Field(
        None, description="Path to LMDB database for template lookup"
    )
    input_circuit_path: Optional[str] = Field(
        None, description="Path to input circuit file (if not generating)"
    )


class ExperimentPreset(BaseModel):
    """A pre-configured experiment preset."""

    id: str
    name: str
    description: str
    experiment_type: ExperimentType
    config: ExperimentConfig
    tags: List[str] = []


class StartExperimentRequest(BaseModel):
    """Request to start an experiment."""

    config: ExperimentConfig


class StartExperimentResponse(BaseModel):
    """Response after starting an experiment."""

    job_id: str
    status: ExperimentStatus
    message: str
    started_at: datetime


class ExperimentProgress(BaseModel):
    """Progress update from a running experiment."""

    job_id: str
    status: ExperimentStatus
    progress_percent: float = 0.0
    current_round: Optional[int] = None
    total_rounds: Optional[int] = None
    current_gates: Optional[int] = None
    elapsed_seconds: float = 0.0
    log_lines: List[str] = []


class ExperimentResults(BaseModel):
    """Results from a completed experiment."""

    job_id: str
    status: ExperimentStatus
    config: ExperimentConfig

    # --- Timing ---
    started_at: datetime
    completed_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0

    # --- Results ---
    initial_gates: int
    final_gates: int
    expansion_factor: float

    # --- Heatmap (if generated) ---
    heatmap_data: Optional[List[List[float]]] = None
    heatmap_x_size: Optional[int] = None
    heatmap_y_size: Optional[int] = None

    # --- Output files ---
    output_circuit_path: Optional[str] = None
    results_json_path: Optional[str] = None

    # --- Logs ---
    log_output: Optional[str] = None


class PresetsResponse(BaseModel):
    """Response with available experiment presets."""

    presets: List[ExperimentPreset]


class ConfigSchemaResponse(BaseModel):
    """Response with JSON schema for experiment configuration."""

    schema: Dict[str, Any]
    parameter_descriptions: Dict[str, str]


class ExperimentStatusResponse(BaseModel):
    """Response with experiment status."""

    job_id: str
    status: ExperimentStatus
    progress: Optional[ExperimentProgress] = None
    results: Optional[ExperimentResults] = None
