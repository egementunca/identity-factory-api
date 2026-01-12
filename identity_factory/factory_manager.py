"""
Identity Circuit Factory Manager.
Main orchestrator for generating, unrolling, and managing identity circuits.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .database import CircuitDatabase
from .debris_cancellation import DebrisCancellationManager
from .job_queue import JobQueueManager
from .ml_features import MLFeatureManager
from .post_processor import PostProcessor
from .seed_generator import SeedGenerator
from .unroller import CircuitUnroller

logger = logging.getLogger(__name__)


@dataclass
class FactoryConfig:
    """Configuration for the identity factory."""

    db_path: str = "identity_circuits.db"
    max_inverse_gates: int = 40
    max_equivalents: int = 10000
    sat_solver: str = "minisat-gh"
    log_level: str = "INFO"
    enable_post_processing: bool = True
    enable_unrolling: bool = True
    enable_debris_analysis: bool = True
    enable_ml_features: bool = True
    enable_job_queue: bool = True
    max_workers: int = 4
    max_debris_gates: int = 5
    redis_url: str = "redis://localhost:6379"
    parquet_storage: bool = False


@dataclass
class FactoryStats:
    """Statistics about the factory operations."""

    total_dim_groups: int = 0
    total_circuits: int = 0
    total_representatives: int = 0
    total_equivalents: int = 0
    total_simplifications: int = 0
    total_debris_analyses: int = 0
    total_ml_analyses: int = 0
    active_jobs: int = 0
    generation_time: float = 0.0
    unroll_time: float = 0.0
    post_process_time: float = 0.0
    debris_analysis_time: float = 0.0
    ml_analysis_time: float = 0.0


class IdentityFactory:
    """Main factory for identity circuit generation and management."""

    def __init__(self, config: Optional[FactoryConfig] = None):
        self.config = config or FactoryConfig()
        self._setup_logging()

        # Initialize database
        self.db = CircuitDatabase(self.config.db_path)

        # Initialize components
        self.seed_generator = SeedGenerator(self.db, self.config.max_inverse_gates)
        self.unroller = CircuitUnroller(self.db, self.config.max_equivalents)
        self.post_processor = PostProcessor(self.db)
        self.debris_manager = DebrisCancellationManager(
            self.db, self.config.max_debris_gates
        )
        self.ml_manager = MLFeatureManager(self.db)

        # Initialize job queue if enabled
        self.job_queue = None
        if self.config.enable_job_queue:
            try:
                self.job_queue = JobQueueManager(
                    database=self.db, max_workers=self.config.max_workers
                )
                self._register_job_handlers()
            except Exception as e:
                logger.warning(f"Failed to initialize job queue: {e}")
                self.job_queue = None

        logger.info("IdentityFactory initialized successfully")

    def _setup_logging(self):
        """Set up logging configuration."""
        # Set more detailed logging for debugging circuit generation issues
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # For debugging circuit generation, we want to see debug messages
        if self.config.log_level.upper() == "DEBUG":
            log_level = logging.DEBUG

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override any existing logging configuration
        )

        # Make sure our specific modules show debug messages when needed
        if log_level == logging.DEBUG:
            logging.getLogger("identity_factory.seed_generator").setLevel(logging.DEBUG)
            logging.getLogger("identity_factory.database").setLevel(logging.DEBUG)
            logging.getLogger("identity_factory.unroller").setLevel(logging.DEBUG)

        logger.info(f"Logging set up with level: {self.config.log_level}")
        logger.info(f"Debug logging enabled for circuit generation debugging")

    def _register_job_handlers(self):
        """Register job handlers with the job queue."""
        if self.job_queue:
            from .job_queue import JobType

            self.job_queue.register_job_handler(
                JobType.SEED_GENERATION, self._handle_seed_generation_job
            )
            self.job_queue.register_job_handler(
                JobType.UNROLLING, self._handle_unrolling_job
            )
            self.job_queue.register_job_handler(
                JobType.POST_PROCESSING, self._handle_post_processing_job
            )
            self.job_queue.register_job_handler(
                JobType.DEBRIS_ANALYSIS, self._handle_debris_analysis_job
            )
            self.job_queue.register_job_handler(
                JobType.ML_FEATURE_EXTRACTION, self._handle_ml_feature_job
            )

    async def _handle_seed_generation_job(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle seed generation job."""
        width = params["width"]
        gate_count = params["gate_count"]
        # Map legacy 'sequential' -> 'enforce_double_length' if provided
        seed_kwargs = {}
        if "max_attempts" in params:
            seed_kwargs["max_attempts"] = params["max_attempts"]
        if "enforce_double_length" in params:
            seed_kwargs["enforce_double_length"] = params["enforce_double_length"]
        # Ignore obsolete 'sequential' parameter silently for backward compatibility
        seed_result = self.seed_generator.generate_seed(
            width, gate_count, **seed_kwargs
        )
        return seed_result.__dict__

    async def start_job_queue(self):
        """Start the job queue workers."""
        if self.job_queue:
            await self.job_queue.start_workers()

    async def stop_job_queue(self):
        """Stop the job queue workers."""
        if self.job_queue:
            await self.job_queue.stop_workers()

    def generate_identity_circuit(
        self,
        width: int,
        gate_count: int,
        enable_unrolling: bool = True,
        enable_post_processing: bool = True,
        enable_debris_analysis: bool = True,
        enable_ml_analysis: bool = True,
        use_job_queue: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a complete identity circuit with optional processing steps.

        Args:
            width: Number of qubits
            gate_count: Number of gates in forward circuit
            enable_unrolling: Whether to generate equivalent circuits
            enable_post_processing: Whether to apply simplifications
            enable_debris_analysis: Whether to analyze for debris cancellation
            enable_ml_analysis: Whether to extract ML features
            use_job_queue: Whether to use distributed processing
            **kwargs: Additional parameters

        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        result = {
            "success": False,
            "width": width,
            "gate_count": gate_count,
            "total_time": 0.0,
        }

        try:
            logger.info(f"Generating identity circuit ({width}, {gate_count})")

            # Step 1: Generate seed circuit
            # Filter kwargs to only include parameters that generate_seed accepts
            seed_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["max_attempts", "enforce_double_length"]
            }
            seed_result = self.seed_generator.generate_seed(
                width, gate_count, **seed_kwargs
            )
            result["seed_generation"] = seed_result

            if not seed_result.success:
                result["error"] = seed_result.error_message
                return result

            circuit_id = seed_result.circuit_id
            dim_group_id = seed_result.dim_group_id
            representative_id = seed_result.representative_id

            # Step 2: Unroll to generate equivalents (if enabled and representative was created)
            if enable_unrolling and representative_id:
                if use_job_queue and self.job_queue:
                    # Submit unrolling job (would need async version)
                    # For now, just mark as queued without actual submission
                    result["unrolling"] = {"status": "queued"}
                else:
                    unroll_result = self.unroller.unroll_dimension_group(dim_group_id)
                    result["unrolling"] = unroll_result

            # Step 3: Post-processing (if enabled)
            if enable_post_processing and dim_group_id:
                if use_job_queue and self.job_queue:
                    # Submit post-processing job (would need async version)
                    result["post_processing"] = {"status": "queued"}
                else:
                    post_result = self.post_processor.simplify_dimension_group(
                        dim_group_id
                    )
                    result["post_processing"] = post_result
                    result["successful_simplifications"] = len(
                        [r for r in post_result.values() if r.success]
                    )

            # Step 4: Debris analysis (if enabled)
            if enable_debris_analysis and circuit_id and dim_group_id:
                if use_job_queue and self.job_queue:
                    # Submit debris analysis job (would need async version)
                    result["debris_analysis"] = {"status": "queued"}
                else:
                    debris_result = (
                        self.debris_manager.analyze_dim_group_representative(
                            dim_group_id, circuit_id
                        )
                    )
                    result["debris_analysis"] = debris_result

            # Step 5: ML feature extraction (if enabled)
            if enable_ml_analysis and circuit_id and dim_group_id:
                if use_job_queue and self.job_queue:
                    # Submit ML feature extraction job (would need async version)
                    result["ml_analysis"] = {"status": "queued"}
                else:
                    circuit_record = self.db.get_circuit(circuit_id)
                    if circuit_record:
                        circuit = self._record_to_circuit(circuit_record)
                        ml_result = self.ml_manager.analyze_circuit(
                            circuit_id, dim_group_id, circuit
                        )
                        result["ml_analysis"] = ml_result

            result["success"] = True

        except Exception as e:
            logger.error(f"Circuit generation failed: {e}")
            result["error"] = str(e)

        finally:
            result["total_time"] = time.time() - start_time

        return result

    def _record_to_circuit(self, record):
        """Helper to convert a CircuitRecord from the DB to a sat_revsynth.Circuit object."""
        from circuit.circuit import Circuit

        circuit = Circuit(record.width)
        for gate in record.gates:
            controls, target = gate
            if len(controls) == 0:
                circuit = circuit.x(target)
            elif len(controls) == 1:
                circuit = circuit.cx(controls[0], target)
            elif len(controls) == 2:
                circuit = circuit.mcx(controls, target)
            else:
                circuit = circuit.mcx(controls, target)
        return circuit

    def _handle_unrolling_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unrolling job."""
        dim_group_id = parameters["dim_group_id"]
        unroll_types = parameters.get("unroll_types")

        result = self.unroller.unroll_dimension_group(dim_group_id, unroll_types)
        return result.__dict__

    def _handle_post_processing_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle post-processing job."""
        dim_group_id = parameters["dim_group_id"]
        simplification_types = parameters.get("simplification_types")

        results = self.post_processor.simplify_dimension_group(
            dim_group_id, simplification_types
        )

        # Convert results to serializable format
        serializable_results = {}
        for circuit_id, result in results.items():
            serializable_results[str(circuit_id)] = result.__dict__

        return serializable_results

    def _handle_debris_analysis_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle debris analysis job."""
        dim_group_id = parameters["dim_group_id"]
        circuit_id = parameters["circuit_id"]

        result = self.debris_manager.analyze_dim_group_representative(
            dim_group_id, circuit_id
        )
        return result or {}

    def _handle_ml_feature_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ML feature extraction job."""
        circuit_id = parameters["circuit_id"]
        dim_group_id = parameters["dim_group_id"]

        circuit_record = self.db.get_circuit(circuit_id)
        if not circuit_record:
            return {"error": "Circuit not found"}

        circuit = self._record_to_circuit(circuit_record)
        result = self.ml_manager.analyze_circuit(circuit_id, dim_group_id, circuit)
        return result

    def batch_generate(
        self, dimensions: List[Tuple[int, int]], use_job_queue: bool = False, **kwargs
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Generate multiple identity circuits for different dimensions.

        Args:
            dimensions: List of (width, gate_count) tuples
            use_job_queue: Whether to use distributed processing
            **kwargs: Additional parameters for generation

        Returns:
            Dictionary mapping dimensions to generation results
        """
        results = {}

        logger.info(f"Batch generating {len(dimensions)} dimension groups")

        for width, gate_count in dimensions:
            logger.info(f"Processing dimension ({width}, {gate_count})")

            try:
                result = self.generate_identity_circuit(
                    width, gate_count, use_job_queue=use_job_queue, **kwargs
                )
                results[(width, gate_count)] = result

            except Exception as e:
                logger.error(f"Failed to generate ({width}, {gate_count}): {e}")
                results[(width, gate_count)] = {
                    "success": False,
                    "error": str(e),
                    "width": width,
                    "gate_count": gate_count,
                }

        return results

    def get_factory_stats(self) -> FactoryStats:
        """Get comprehensive factory statistics."""
        db_stats = self.db.get_database_stats()

        return FactoryStats(
            total_dim_groups=db_stats.get("total_dim_groups", 0),
            total_circuits=db_stats.get("total_circuits", 0),
            total_representatives=db_stats.get("total_representatives", 0),
            total_equivalents=db_stats.get("total_equivalents", 0),
            total_simplifications=0,  # Would need to be tracked
            total_debris_analyses=db_stats.get("debris_cancellations", 0),
            total_ml_analyses=db_stats.get("ml_features", 0),
            active_jobs=sum(db_stats.get("job_status_counts", {}).values()),
            generation_time=self.seed_generator.total_generation_time,
            unroll_time=self.unroller.total_unroll_time,
            post_process_time=0.0,  # Would need to be tracked
            debris_analysis_time=0.0,  # Would need to be tracked
            ml_analysis_time=0.0,  # Would need to be tracked
        )

    def get_dimension_group_analysis(self, dim_group_id: int) -> Dict[str, Any]:
        """Get comprehensive analysis of a dimension group."""
        dim_group = self.db.get_dim_group_by_id(dim_group_id)
        if not dim_group:
            return {"error": "Dimension group not found"}

        # Get representatives
        representatives = self.db.get_representatives_for_dim_group(dim_group_id)

        # Get all equivalents
        equivalents = self.db.get_all_equivalents_for_dim_group(dim_group_id)

        # Analyze gate compositions
        composition_analysis = self._analyze_gate_compositions(equivalents)

        analysis = {
            "dim_group_id": dim_group_id,
            "width": dim_group.width,
            "gate_count": dim_group.gate_count,
            "circuit_count": dim_group.circuit_count,
            "total_equivalents": len(equivalents),
            "is_processed": dim_group.is_processed,
            "representatives": [
                {
                    "id": rep.id,
                    "circuit_id": rep.circuit_id,
                    "gate_composition": rep.gate_composition,
                    "is_primary": rep.is_primary,
                }
                for rep in representatives
            ],
            "equivalents": {
                "total": len(equivalents),
                "by_unroll_type": {},
                "by_gate_composition": composition_analysis,
            },
        }

        # Group equivalents by unroll type
        for equiv in equivalents:
            unroll_type = equiv.get("unroll_type", "unknown")
            if unroll_type not in analysis["equivalents"]["by_unroll_type"]:
                analysis["equivalents"]["by_unroll_type"][unroll_type] = 0
            analysis["equivalents"]["by_unroll_type"][unroll_type] += 1

        return analysis

    def _analyze_gate_compositions(
        self, equivalents: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze gate compositions in equivalent circuits."""
        compositions = {}
        for equiv in equivalents:
            comp = equiv.get("gate_composition")
            if comp:
                comp_str = str(comp)
                compositions[comp_str] = compositions.get(comp_str, 0) + 1
        return compositions

    def export_dimension_group(self, dim_group_id: int, output_path: str) -> bool:
        """Export a dimension group to a file."""
        try:
            analysis = self.get_dimension_group_analysis(dim_group_id)
            if "error" in analysis:
                return False

            import json

            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"Exported dimension group {dim_group_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_dimension_group(self, import_path: str) -> bool:
        """Import a dimension group from a file."""
        try:
            import json

            with open(import_path, "r") as f:
                data = json.load(f)

            # This would implement the actual import logic
            # For now, just log the attempt
            logger.info(f"Imported dimension group from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False

    def cleanup_old_circuits(self, max_age_days: int = 30) -> int:
        """Clean up old circuits and associated data."""
        # This would implement cleanup logic
        # For now, return 0
        logger.info(f"Cleanup completed (placeholder)")
        return 0

    def get_recommended_dimensions(
        self, target_width: int, max_gate_count: int = 20
    ) -> List[Tuple[int, int]]:
        """Get recommended dimensions for exploration."""
        recommendations = []

        # Simple recommendation logic
        for gate_count in range(2, max_gate_count + 1, 2):
            recommendations.append((target_width, gate_count))

        return recommendations[:10]  # Limit to 10 recommendations

    def enable_debug_logging(self):
        """Enable debug logging for troubleshooting circuit generation issues."""
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("identity_factory.seed_generator").setLevel(logging.DEBUG)
        logging.getLogger("identity_factory.database").setLevel(logging.DEBUG)
        logging.getLogger("identity_factory.unroller").setLevel(logging.DEBUG)
        logger.info("🔍 Debug logging enabled for troubleshooting")

    def disable_debug_logging(self):
        """Restore normal logging level."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        logging.getLogger("identity_factory.seed_generator").setLevel(log_level)
        logging.getLogger("identity_factory.database").setLevel(log_level)
        logging.getLogger("identity_factory.unroller").setLevel(log_level)
        logger.info("📝 Normal logging level restored")
