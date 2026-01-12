"""
Simplified seed generation system for identity circuit factory.
Generates identity circuits using forward + inverse synthesis.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for SAT synthesis (not required for database operations)
try:
    from circuit.circuit import Circuit
    from synthesizers.circuit_synthesizer import CircuitSynthesizer

    SAT_SYNTHESIS_AVAILABLE = True
except ImportError:
    Circuit = None
    CircuitSynthesizer = None
    SAT_SYNTHESIS_AVAILABLE = False

from .database import CircuitDatabase, CircuitRecord, DimGroupRecord

logger = logging.getLogger(__name__)


@dataclass
class SeedGenerationResult:
    """Result of seed generation process."""

    success: bool
    circuit_id: Optional[int] = None
    dim_group_id: Optional[int] = None
    forward_gates: Optional[List[Tuple]] = None
    inverse_gates: Optional[List[Tuple]] = None
    identity_gates: Optional[List[Tuple]] = None
    permutation: Optional[List[int]] = None
    complexity_walk: Optional[List[int]] = None
    gate_composition: Optional[Tuple[int, int, int]] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class SeedGenerator:
    """Simplified seed generator using forward + inverse synthesis."""

    def __init__(self, database: CircuitDatabase, max_inverse_gates: int = 40):
        self.database = database
        self.max_inverse_gates = max_inverse_gates
        self.generation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_generation_time = 0.0
        self.sat_available = False

        logger.info(
            f"SeedGenerator initialized with max_inverse_gates={max_inverse_gates}"
        )

        # Test dependencies (optional - API can work without SAT synthesis)
        try:
            from circuit.circuit import Circuit
            from sat.solver import Solver
            from synthesizers.circuit_synthesizer import CircuitSynthesizer

            # Test SAT solver availability
            solver = Solver("minisat-gh")
            self.sat_available = True
            logger.info("✓ All SAT synthesis dependencies available")
        except Exception as e:
            logger.warning(f"⚠ SAT synthesis not available: {e}")
            logger.warning(
                "Database operations will still work, but seed generation is disabled"
            )

    def generate_seed(
        self, width: int, forward_length: int, max_attempts: int = 10
    ) -> SeedGenerationResult:
        """
        Generate an identity circuit for the given dimensions.

        Args:
            width: Number of qubits
            forward_length: Number of gates in the forward circuit
            max_attempts: Maximum number of generation attempts

        Returns:
            SeedGenerationResult with the generated circuit information
        """
        start_time = time.time()

        logger.info(
            f"Generating seed for width={width}, forward_length={forward_length}"
        )

        for attempt in range(max_attempts):
            logger.info(f"🔄 Starting generation attempt {attempt + 1}/{max_attempts}")
            try:
                result = self._attempt_generation(width, forward_length)
                if result.success:
                    logger.info(
                        f"✅ Successfully generated seed circuit {result.circuit_id} on attempt {attempt + 1}"
                    )

                    generation_time = time.time() - start_time
                    self.total_generation_time += generation_time
                    self.generation_count += 1
                    self.success_count += 1

                    result.metrics = {
                        "generation_time": generation_time,
                        "attempts": attempt + 1,
                        "forward_length": forward_length,
                        "total_length": (
                            len(result.identity_gates) if result.identity_gates else 0
                        ),
                    }

                    return result
                else:
                    logger.warning(
                        f"❌ Attempt {attempt + 1} failed: {result.error_message}"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Generation attempt {attempt + 1} failed with exception: {e}"
                )
                import traceback

                logger.error(f"Exception traceback: {traceback.format_exc()}")
                continue

        # All attempts failed
        generation_time = time.time() - start_time
        self.total_generation_time += generation_time
        self.generation_count += 1
        self.failure_count += 1

        logger.error(
            f"Failed to generate seed for width={width}, forward_length={forward_length} after {max_attempts} attempts"
        )

        return SeedGenerationResult(
            success=False,
            error_message=f"Failed to generate after {max_attempts} attempts",
            metrics={"generation_time": generation_time, "attempts": max_attempts},
        )

    def _attempt_generation(
        self, width: int, forward_length: int
    ) -> SeedGenerationResult:
        """Attempt to generate one identity circuit."""
        try:
            # Step 1: Generate random forward circuit
            logger.info(
                f"Step 1: Generating random forward circuit with {forward_length} gates"
            )
            try:
                forward_circuit = self._generate_random_circuit(width, forward_length)
                forward_gates = self._convert_circuit_gates_to_tuples(
                    forward_circuit.gates()
                )
                logger.info(f"Forward circuit generated: {forward_gates}")
            except Exception as e:
                logger.error(f"Step 1 FAILED - Forward circuit generation: {e}")
                return SeedGenerationResult(
                    success=False,
                    error_message=f"Forward circuit generation failed: {e}",
                )

            # Step 2: Get permutation from forward circuit
            logger.info("Step 2: Getting permutation from forward circuit")
            try:
                permutation = forward_circuit.tt().values()
                logger.info(f"Forward circuit permutation: {permutation}")
            except Exception as e:
                logger.error(f"Step 2 FAILED - Permutation calculation: {e}")
                return SeedGenerationResult(
                    success=False, error_message=f"Permutation calculation failed: {e}"
                )

            # Step 3: Synthesize inverse circuit using SAT solver with optimal gate count
            logger.info(
                "Step 3: Synthesizing inverse circuit using SAT solver with optimal gate count"
            )
            try:
                inverse_circuit = self._synthesize_inverse_circuit(forward_circuit)
                if not inverse_circuit:
                    logger.error("Step 3 FAILED - SAT solver returned None")
                    return SeedGenerationResult(
                        success=False,
                        error_message="SAT solver failed to find inverse circuit",
                    )

                inverse_gates = self._convert_circuit_gates_to_tuples(
                    inverse_circuit.gates()
                )
                logger.info(
                    f"Inverse circuit synthesized: {inverse_gates} (length: {len(inverse_gates)})"
                )
            except Exception as e:
                logger.error(f"Step 3 FAILED - Inverse synthesis: {e}")
                return SeedGenerationResult(
                    success=False, error_message=f"Inverse synthesis failed: {e}"
                )

            # Step 4: Combine forward + inverse to get identity circuit
            logger.info("Step 4: Combining forward and inverse circuits")
            try:
                identity_circuit = self._combine_circuits(
                    forward_circuit, inverse_circuit
                )
                # Convert circuit's internal format to our gate composition format
                identity_gates = self._convert_circuit_gates_to_tuples(
                    identity_circuit.gates()
                )
                logger.info(
                    f"Identity circuit created: {len(identity_gates)} total gates"
                )
            except Exception as e:
                logger.error(f"Step 4 FAILED - Circuit combination: {e}")
                return SeedGenerationResult(
                    success=False, error_message=f"Circuit combination failed: {e}"
                )

            # Step 5: Verify it's actually an identity
            logger.info("Step 5: Verifying identity circuit")
            try:
                identity_permutation = identity_circuit.tt().values()
                expected_identity = list(
                    range(2**width)
                )  # Fixed: should be 2^width, not width
                if identity_permutation != expected_identity:
                    logger.error(
                        f"Step 5 FAILED - Identity verification: got {identity_permutation}, expected {expected_identity}"
                    )
                    return SeedGenerationResult(
                        success=False,
                        error_message=f"Combined circuit is not identity: {identity_permutation} != {expected_identity}",
                    )
                logger.info("Identity verification PASSED")
            except Exception as e:
                logger.error(f"Step 5 FAILED - Identity verification: {e}")
                return SeedGenerationResult(
                    success=False, error_message=f"Identity verification failed: {e}"
                )

            # Step 6: Check if this circuit already exists in database
            existing_circuit = self._check_circuit_exists(width, identity_gates)
            if existing_circuit:
                logger.info(
                    f"Circuit already exists as ID {existing_circuit.id}, skipping duplicate"
                )
                return SeedGenerationResult(
                    success=True,
                    circuit_id=existing_circuit.id,
                    dim_group_id=existing_circuit.dim_group_id,
                    forward_gates=forward_gates,
                    inverse_gates=inverse_gates,
                    identity_gates=identity_gates,
                    permutation=expected_identity,
                    gate_composition=existing_circuit.get_gate_composition(),
                )

            # Step 6: Store new circuit
            total_length = len(identity_gates)
            complexity_walk = self._generate_complexity_walk(identity_gates, width)
            gate_composition = self._calculate_gate_composition(identity_gates)

            # Get or create dimension group
            dim_group = self.database.get_dim_group(width, total_length)
            if not dim_group:
                logger.info(
                    f"Creating new dimension group for ({width}, {total_length})"
                )
                dim_group = DimGroupRecord(
                    id=None,
                    width=width,
                    gate_count=total_length,
                    circuit_count=0,
                    is_processed=False,
                )
                dim_group_id = self.database.store_dim_group(dim_group)
            else:
                dim_group_id = dim_group.id
                logger.info(
                    f"Using existing dimension group {dim_group_id} for ({width}, {total_length})"
                )

            # Store the circuit (representative_id will be set to itself by database)
            circuit_record = CircuitRecord(
                id=None,
                width=width,
                gate_count=total_length,
                gates=identity_gates,
                permutation=expected_identity,
                complexity_walk=complexity_walk,
                dim_group_id=dim_group_id,
                representative_id=None,  # Will be set to circuit_id by database
            )
            circuit_id = self.database.store_circuit(circuit_record)

            # Add circuit to dimension group
            self.database.add_circuit_to_dim_group(dim_group_id, circuit_id)

            logger.info(
                f"Stored new identity circuit {circuit_id} in dimension group {dim_group_id}"
            )

            return SeedGenerationResult(
                success=True,
                circuit_id=circuit_id,
                dim_group_id=dim_group_id,
                forward_gates=forward_gates,
                inverse_gates=inverse_gates,
                identity_gates=identity_gates,
                permutation=expected_identity,
                complexity_walk=complexity_walk,
                gate_composition=gate_composition,
            )

        except Exception as e:
            logger.error(f"Error during generation attempt: {e}")
            return SeedGenerationResult(success=False, error_message=str(e))

    def _generate_random_circuit(self, width: int, gate_count: int) -> Circuit:
        """Generate a random quantum circuit."""
        circuit = Circuit(width)
        gate_pool = self._build_gate_pool(width)

        for _ in range(gate_count):
            gate_type, *params = random.choice(gate_pool)

            if gate_type == "X":
                circuit.x(params[0])
            elif gate_type == "CX":
                circuit.cx(params[0], params[1])
            elif gate_type == "CCX":
                circuit.mcx(params[:2], params[2])

        return circuit

    def _build_gate_pool(self, width: int) -> List[Tuple]:
        """Build pool of possible gates for given width with normalized CCX gates."""
        gates = []

        # X gates on each qubit
        for i in range(width):
            gates.append(("X", i))

        # CX gates (all possible control/target combinations)
        for control in range(width):
            for target in range(width):
                if control != target:
                    gates.append(("CX", control, target))

        # CCX gates (Toffoli) for 3+ qubits - normalized with sorted controls
        if width >= 3:
            for control1 in range(width):
                for control2 in range(
                    control1 + 1, width
                ):  # Ensure control1 < control2
                    for target in range(width):
                        if target != control1 and target != control2:
                            # Store controls in sorted order for normalization
                            gates.append(("CCX", control1, control2, target))

        return gates

    def _synthesize_inverse_circuit(
        self, forward_circuit: Circuit
    ) -> Optional[Circuit]:
        """Synthesize inverse circuit using SAT solver with optimal gate count."""
        try:
            logger.info("Starting SAT-based inverse synthesis...")
            from sat.solver import Solver
            from synthesizers.optimal_synthesizer import OptimalSynthesizer
            from truth_table.truth_table import TruthTable

            # Get the permutation from the forward circuit
            logger.info("Getting forward permutation...")
            forward_permutation = forward_circuit.tt().values()
            logger.info(f"Forward permutation: {forward_permutation}")

            # Calculate the inverse permutation
            logger.info("Calculating inverse permutation...")
            inverse_permutation = [0] * len(forward_permutation)
            for i, val in enumerate(forward_permutation):
                inverse_permutation[val] = i
            logger.info(f"Inverse permutation: {inverse_permutation}")

            # Convert to TruthTable format expected by OptimalSynthesizer
            width = forward_circuit.width()
            logger.info(f"Converting to truth table format for {width} qubits...")

            # Create truth table directly from permutation
            truth_table = TruthTable(width, inverse_permutation)
            logger.info(f"Truth table created for {width} qubits")

            # Use OptimalSynthesizer to find the shortest circuit up to max_inverse_gates
            logger.info(
                f"Initializing SAT solver (minisat-gh) for optimal synthesis..."
            )
            try:
                solver = Solver("minisat-gh")  # Use built-in minisat SAT solver
                logger.info("SAT solver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SAT solver: {e}")
                raise

            try:
                # Use OptimalSynthesizer to find shortest circuit from 0 to max_inverse_gates
                optimal_synthesizer = OptimalSynthesizer(
                    output=truth_table,
                    lower_gc=0,  # Start from 0 gates
                    upper_gc=self.max_inverse_gates,  # Up to max_inverse_gates
                    solver=solver,
                )
                logger.info("Optimal synthesizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize optimal synthesizer: {e}")
                raise

            # Try to synthesize the inverse circuit (finds shortest solution)
            logger.info("Starting optimal SAT synthesis...")
            try:
                inverse_circuit = optimal_synthesizer.solve()
                logger.info(
                    f"Optimal SAT synthesis completed. Result: {inverse_circuit is not None}"
                )
            except Exception as e:
                logger.error(f"Optimal SAT synthesis failed with error: {e}")
                raise

            if inverse_circuit:
                logger.info(
                    f"✓ Successfully synthesized inverse circuit with {len(inverse_circuit)} gates (optimal length)"
                )
                return inverse_circuit
            else:
                logger.error(
                    f"✗ SAT solver found no solution within {self.max_inverse_gates} gates"
                )
                return None

        except ImportError as e:
            logger.error(f"Import error in SAT synthesis: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in SAT synthesis: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _combine_circuits(self, forward: Circuit, inverse: Circuit) -> Circuit:
        """Combine forward and inverse circuits to create identity."""
        combined = forward + inverse
        return combined

    def _convert_circuit_gates_to_tuples(
        self, circuit_gates: List[Tuple]
    ) -> List[Tuple]:
        """Convert from circuit's internal format (controls_list, target) to our tuple format."""
        converted_gates = []

        for controls, target in circuit_gates:
            if len(controls) == 0:
                # NOT gate
                converted_gates.append(("X", target))
            elif len(controls) == 1:
                # CNOT gate
                converted_gates.append(("CX", controls[0], target))
            elif len(controls) == 2:
                # CCNOT gate - ensure controls are sorted for consistency
                sorted_controls = sorted(controls)
                converted_gates.append(
                    ("CCX", sorted_controls[0], sorted_controls[1], target)
                )
            else:
                # Multi-controlled gates (should not happen in our use case, but handle gracefully)
                logger.warning(
                    f"Unsupported gate with {len(controls)} controls, treating as CCNOT with first two controls"
                )
                sorted_controls = sorted(controls[:2])
                converted_gates.append(
                    ("CCX", sorted_controls[0], sorted_controls[1], target)
                )

        return converted_gates

    def _calculate_gate_composition(self, gates: List[Tuple]) -> Tuple[int, int, int]:
        """Calculate gate composition (NOT, CNOT, CCNOT counts)."""
        not_count = sum(1 for gate in gates if gate[0] == "X")
        cnot_count = sum(1 for gate in gates if gate[0] == "CX")
        ccnot_count = sum(1 for gate in gates if gate[0] == "CCX")
        return (not_count, cnot_count, ccnot_count)

    def _generate_complexity_walk(self, gates: List[Tuple], width: int) -> List[int]:
        """Generate complexity walk using Hamming distance from identity after each gate."""
        N = 1 << width  # 2^width
        mapping = list(range(N))
        complexity_walk = []

        for gate in gates:
            gate_type = gate[0]

            if gate_type == "X":  # NOT gate
                qubit = gate[1]
                mask = 1 << qubit
                for i in range(N):
                    mapping[i] ^= mask

            elif gate_type == "CX":  # CNOT gate
                control, target = gate[1], gate[2]
                mask = 1 << target
                for i in range(N):
                    if mapping[i] & (1 << control):
                        mapping[i] ^= mask

            elif gate_type == "CCX":  # CCNOT (Toffoli) gate
                control1, control2, target = gate[1], gate[2], gate[3]
                mask = 1 << target
                for i in range(N):
                    if (mapping[i] & (1 << control1)) and (
                        mapping[i] & (1 << control2)
                    ):
                        mapping[i] ^= mask

            # Calculate Hamming distance from identity
            hamming_distance = sum((i ^ mapping[i]).bit_count() for i in range(N))
            complexity_walk.append(hamming_distance)

        return complexity_walk

    def _check_circuit_exists(
        self, width: int, gates: List[Tuple]
    ) -> Optional[CircuitRecord]:
        """Check if a circuit with the same gates already exists."""
        # Compute hash for the circuit
        identity_permutation = list(range(width))
        circuit_hash = self.database._compute_circuit_hash(gates, identity_permutation)

        # Check if circuit with this hash exists
        existing_circuit = self.database.get_circuit_by_hash(circuit_hash)
        return existing_circuit

    def generate_multiple_seeds(
        self, width: int, forward_length: int, count: int = 1, **kwargs
    ) -> List[SeedGenerationResult]:
        """Generate multiple seed circuits."""
        results = []
        for i in range(count):
            logger.info(f"Generating seed {i+1}/{count}")
            result = self.generate_seed(width, forward_length, **kwargs)
            results.append(result)

            # Log progress
            if result.success:
                logger.info(f"Seed {i+1}: SUCCESS - Circuit ID {result.circuit_id}")
            else:
                logger.warning(f"Seed {i+1}: FAILED - {result.error_message}")

        return results

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        avg_time = self.total_generation_time / max(self.generation_count, 1)
        success_rate = self.success_count / max(self.generation_count, 1) * 100

        return {
            "total_attempts": self.generation_count,
            "successful_generations": self.success_count,
            "failed_generations": self.failure_count,
            "success_rate_percent": success_rate,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_time,
        }
