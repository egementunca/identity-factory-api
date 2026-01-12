"""
Automation scheduler for Identity Circuit Factory.
Handles scheduled generation, unrolling, and quality filtering.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database import CircuitDatabase, CircuitRecord
from ..debris_cancellation import DebrisCancellationAnalyzer
from ..seed_generator import SeedGenerator
from ..unroller import CircuitUnroller

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for batch generation."""

    width_range: tuple = (3, 6)  # 3 to 5 wires
    gate_count_range: tuple = (4, 12)  # 4 to 10 gates (even only)
    circuits_per_dimension: int = 50
    max_total_circuits: int = 1000


@dataclass
class QualityConfig:
    """Configuration for quality filtering."""

    min_non_triviality_score: float = 1.5
    min_wire_spread: int = 2  # Use at least 2 target wires
    max_compression_ratio: float = 0.9  # Keep if compression achieves < 10% reduction


@dataclass
class SchedulerStats:
    """Statistics for scheduler runs."""

    total_generated: int = 0
    total_unrolled: int = 0
    total_filtered: int = 0
    total_kept: int = 0
    last_run: Optional[datetime] = None
    runs_completed: int = 0


class FactoryScheduler:
    """
    Automated scheduler for identity circuit factory.

    Handles:
    - Batch generation of new circuits
    - Unrolling representatives to equivalents
    - Quality filtering based on non-triviality score
    """

    def __init__(
        self,
        database: CircuitDatabase,
        generation_config: Optional[GenerationConfig] = None,
        quality_config: Optional[QualityConfig] = None,
    ):
        self.database = database
        self.gen_config = generation_config or GenerationConfig()
        self.quality_config = quality_config or QualityConfig()

        self.seed_generator = SeedGenerator(database)
        self.unroller = CircuitUnroller(database)
        self.debris_analyzer = DebrisCancellationAnalyzer()

        self.stats = SchedulerStats()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info("FactoryScheduler initialized")

    def run_generation_batch(self, count: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a batch of circuit generation.

        Args:
            count: Optional override for circuits to generate

        Returns:
            Dictionary with generation results
        """
        count = count or self.gen_config.max_total_circuits
        generated = 0
        failed = 0

        width_min, width_max = self.gen_config.width_range
        gate_min, gate_max = self.gen_config.gate_count_range

        logger.info(f"Starting generation batch: target={count} circuits")

        for width in range(width_min, width_max + 1):
            if generated >= count:
                break

            for gate_count in range(gate_min, gate_max + 1, 2):  # Even only
                if generated >= count:
                    break

                per_dim = min(self.gen_config.circuits_per_dimension, count - generated)

                logger.info(
                    f"Generating {per_dim} circuits for {width}w x {gate_count}g"
                )

                for _ in range(per_dim):
                    try:
                        result = self.seed_generator.generate_seed(
                            width=width, forward_length=gate_count // 2
                        )

                        # SeedGenerationResult is a dataclass, check .success attribute
                        if result and result.success:
                            generated += 1
                        else:
                            failed += 1

                    except Exception as e:
                        logger.warning(f"Generation failed: {e}")
                        failed += 1

        self.stats.total_generated += generated
        self.stats.last_run = datetime.now()
        self.stats.runs_completed += 1

        logger.info(
            f"Generation batch complete: {generated} generated, {failed} failed"
        )

        return {
            "generated": generated,
            "failed": failed,
            "total_in_db": self.database.get_database_stats().get("total_circuits", 0),
        }

    def run_unroll_batch(
        self, max_circuits: int = 100, equivalents_per_circuit: int = 100
    ) -> Dict[str, Any]:
        """
        Unroll representative circuits to generate equivalents.

        Args:
            max_circuits: Maximum circuits to unroll
            equivalents_per_circuit: Max equivalents per circuit

        Returns:
            Dictionary with unrolling results
        """
        logger.info(f"Starting unroll batch: max_circuits={max_circuits}")

        # Get representatives that haven't been fully unrolled
        representatives = self.database.get_all_representatives()[:max_circuits]

        total_equivalents = 0
        circuits_processed = 0

        for rep in representatives:
            try:
                result = self.unroller.unroll_circuit(
                    rep, max_equivalents=equivalents_per_circuit
                )

                if result.get("success"):
                    total_equivalents += result.get("stored_equivalents", 0)
                    circuits_processed += 1

            except Exception as e:
                logger.warning(f"Unroll failed for circuit {rep.id}: {e}")

        self.stats.total_unrolled += total_equivalents

        logger.info(
            f"Unroll batch complete: {circuits_processed} circuits, {total_equivalents} equivalents"
        )

        return {
            "circuits_processed": circuits_processed,
            "equivalents_generated": total_equivalents,
        }

    def run_quality_filter(self) -> Dict[str, Any]:
        """
        Filter circuits based on non-triviality score.

        Returns:
            Dictionary with filtering results
        """
        logger.info("Starting quality filter")

        # Get all circuits
        stats = self.database.get_database_stats()
        total = stats.get("total_circuits", 0)

        kept = 0
        filtered = 0
        analyzed = 0

        # Process in batches
        batch_size = 100
        offset = 0

        while offset < total:
            circuits = self.database.get_circuits_batch(offset, batch_size)

            for circuit in circuits:
                try:
                    # Convert to sat_revsynth Circuit for analysis
                    from circuit.circuit import Circuit

                    c = Circuit(circuit.width)
                    for gate in circuit.gates:
                        if isinstance(gate, (list, tuple)) and len(gate) == 3:
                            if all(isinstance(x, int) for x in gate):
                                ctrl1, ctrl2, target = gate
                                c = c.mcx([ctrl1, ctrl2], target)

                    # Analyze
                    score = self.debris_analyzer.compute_non_triviality_score(c)
                    analyzed += 1

                    if score >= self.quality_config.min_non_triviality_score:
                        kept += 1
                        # TODO: Mark as premium in database
                    else:
                        filtered += 1
                        # TODO: Mark as trivial in database

                except Exception as e:
                    logger.warning(
                        f"Quality analysis failed for circuit {circuit.id}: {e}"
                    )

            offset += batch_size

        self.stats.total_filtered += filtered
        self.stats.total_kept += kept

        logger.info(f"Quality filter complete: {kept} kept, {filtered} filtered")

        return {
            "analyzed": analyzed,
            "kept": kept,
            "filtered": filtered,
            "kept_ratio": kept / analyzed if analyzed > 0 else 0,
        }

    def run_full_cycle(self) -> Dict[str, Any]:
        """
        Run a complete generation -> unroll -> filter cycle.

        Returns:
            Combined results from all steps
        """
        logger.info("Starting full automation cycle")
        start_time = time.time()

        # Step 1: Generate new circuits
        gen_result = self.run_generation_batch()

        # Step 2: Unroll representatives
        unroll_result = self.run_unroll_batch()

        # Step 3: Quality filter
        filter_result = self.run_quality_filter()

        elapsed = time.time() - start_time

        logger.info(f"Full cycle complete in {elapsed:.1f}s")

        return {
            "generation": gen_result,
            "unrolling": unroll_result,
            "filtering": filter_result,
            "elapsed_seconds": elapsed,
            "stats": {
                "total_generated": self.stats.total_generated,
                "total_unrolled": self.stats.total_unrolled,
                "total_kept": self.stats.total_kept,
                "runs_completed": self.stats.runs_completed,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_generated": self.stats.total_generated,
            "total_unrolled": self.stats.total_unrolled,
            "total_filtered": self.stats.total_filtered,
            "total_kept": self.stats.total_kept,
            "runs_completed": self.stats.runs_completed,
            "last_run": (
                self.stats.last_run.isoformat() if self.stats.last_run else None
            ),
        }


def create_scheduler(db_path: Optional[str] = None) -> FactoryScheduler:
    """
    Create a scheduler instance with default configuration.

    Args:
        db_path: Optional database path override

    Returns:
        Configured FactoryScheduler
    """
    if db_path is None:
        db_path = str(Path.home() / ".identity_factory" / "circuits.db")

    database = CircuitDatabase(db_path)

    return FactoryScheduler(
        database=database,
        generation_config=GenerationConfig(
            width_range=(3, 5),
            gate_count_range=(4, 10),
            circuits_per_dimension=20,
            max_total_circuits=200,
        ),
        quality_config=QualityConfig(min_non_triviality_score=1.5),
    )
