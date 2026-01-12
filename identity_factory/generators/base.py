"""
Base interfaces for circuit generators.
All generator backends must implement these interfaces.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GeneratorStatus(Enum):
    """Status of a generator run."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GeneratorInfo:
    """Information about a generator backend."""

    name: str
    display_name: str
    description: str
    gate_sets: List[str]  # Supported gate sets: 'mcx', 'eca57', etc.
    supports_pause: bool = False
    supports_incremental: bool = False
    config_schema: Optional[Dict[str, Any]] = None  # JSON schema for config


@dataclass
class GenerationProgress:
    """Real-time progress information for a generation run."""

    run_id: str
    generator_name: str
    status: GeneratorStatus

    # Progress metrics
    circuits_found: int = 0
    circuits_stored: int = 0
    duplicates_skipped: int = 0

    # Current work
    current_gate_count: Optional[int] = None
    current_width: Optional[int] = None

    # Timing
    started_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    # Rates
    circuits_per_second: float = 0.0

    # Messages
    current_status: str = ""
    log_messages: List[str] = field(default_factory=list)

    # Error info
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of a generation run."""

    success: bool
    run_id: str
    generator_name: str

    # What was generated
    total_circuits: int = 0
    new_circuits: int = 0
    duplicates: int = 0

    # Parameters used
    width: Optional[int] = None
    gate_count: Optional[int] = None
    gate_set: str = "mcx"
    config: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_seconds: float = 0.0

    # Error info
    error: Optional[str] = None

    # Database info
    dim_group_id: Optional[int] = None
    circuit_ids: List[int] = field(default_factory=list)

    # Generated circuits data (for database storage)
    # Each circuit is a dict with 'width', 'gates', 'permutation' etc.
    circuits: List[Dict[str, Any]] = field(default_factory=list)


class CircuitGenerator(ABC):
    """
    Abstract base class for circuit generators.

    All generator backends (SAT forward/inverse, ECA57 miner, Go enumeration)
    must implement this interface.
    """

    def __init__(self):
        self._status = GeneratorStatus.IDLE
        self._current_run_id: Optional[str] = None
        self._progress = None
        self._progress_callbacks: List[Callable[[GenerationProgress], None]] = []
        self._cancel_requested = False
        self._lock = threading.Lock()

    @abstractmethod
    def get_info(self) -> GeneratorInfo:
        """Return information about this generator."""
        pass

    @abstractmethod
    def generate(
        self,
        width: int,
        gate_count: int,
        gate_set: str = "mcx",
        max_circuits: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> GenerationResult:
        """
        Generate identity circuits.

        Args:
            width: Number of wires/qubits
            gate_count: Number of gates in circuit
            gate_set: Gate set to use ('mcx', 'eca57', etc.)
            max_circuits: Maximum circuits to generate (None = no limit)
            config: Generator-specific configuration
            progress_callback: Callback for progress updates

        Returns:
            GenerationResult with generated circuits info
        """
        pass

    @abstractmethod
    def supports_gate_set(self, gate_set: str) -> bool:
        """Check if this generator supports a given gate set."""
        pass

    def get_status(self) -> GeneratorStatus:
        """Get current generator status."""
        return self._status

    def get_progress(self) -> Optional[GenerationProgress]:
        """Get current progress if running."""
        return self._progress

    def cancel(self) -> bool:
        """
        Request cancellation of current generation.
        Returns True if cancellation was requested, False if nothing to cancel.
        """
        with self._lock:
            if self._status == GeneratorStatus.RUNNING:
                self._cancel_requested = True
                return True
            return False

    def is_cancel_requested(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested

    def _update_progress(self, progress: GenerationProgress):
        """Update progress and notify callbacks."""
        self._progress = progress
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception:
                pass  # Don't let callback errors break generation

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def _reset_state(self):
        """Reset state for new run."""
        with self._lock:
            self._cancel_requested = False
            self._progress = None
