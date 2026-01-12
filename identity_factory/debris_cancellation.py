"""
Debris cancellation system for identity circuit factory.
Handles insertion of debris gates to enable new cancellation paths and computes non-triviality scores.
"""

import heapq
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional import for SAT synthesis
try:
    from circuit.circuit import Circuit

    CIRCUIT_AVAILABLE = True
except ImportError:
    Circuit = None
    CIRCUIT_AVAILABLE = False

from .database import CircuitRecord

@dataclass
class DebrisCancellationRecord:
    """Record of a debris cancellation analysis."""
    id: Optional[int]
    circuit_id: int
    dim_group_id: int
    debris_gates: List[Tuple]
    cancellation_path: List[int]
    non_triviality_score: float
    final_gate_count: int
    cancellation_metrics: Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class CancellationPath:
    """Represents a path of gate cancellations."""

    path: List[int]  # Gate indices in cancellation order
    final_gate_count: int
    complexity_score: float
    debris_gates: List[Tuple]  # Gates inserted as debris


@dataclass
class DebrisInsertion:
    """Represents a debris gate insertion."""

    position: int  # Where to insert the debris gate
    gate: Tuple  # The debris gate (controls, target)
    expected_cancellations: int  # Expected number of cancellations this enables


class DebrisCancellationAnalyzer:
    """Analyzes circuits for debris cancellation opportunities."""

    def __init__(self, max_debris_gates: int = 5, max_search_depth: int = 100):
        self.max_debris_gates = max_debris_gates
        self.max_search_depth = max_search_depth

    def analyze_circuit(self, circuit: Circuit) -> Optional[CancellationPath]:
        """
        Analyze a circuit for debris cancellation opportunities.
        Returns the best cancellation path found, or None if no improvement possible.
        """
        original_gates = circuit.gates()
        original_count = len(original_gates)

        # First try basic swap and cancellation
        basic_result = self._basic_cancellation(circuit)
        if basic_result.final_gate_count == 0:
            return basic_result  # Already fully cancelled

        # If basic cancellation doesn't work, try debris insertion
        best_path = self._find_best_debris_path(circuit)

        if best_path and best_path.final_gate_count < original_count:
            return best_path

        return None

    def _basic_cancellation(self, circuit: Circuit) -> CancellationPath:
        """Perform basic swap and cancellation without debris."""
        try:
            # Use the existing reduction method from sat_revsynth
            reduced_circuit, metrics = circuit.reduce_by_swaps_and_cancellation()
            final_gates = reduced_circuit.gates()

            return CancellationPath(
                path=[],  # No specific path for basic cancellation
                final_gate_count=len(final_gates),
                complexity_score=0.0,  # Basic cancellation is trivial
                debris_gates=[],
            )
        except Exception as e:
            logger.warning(f"Basic cancellation failed: {e}")
            return CancellationPath(
                path=[],
                final_gate_count=len(circuit.gates()),
                complexity_score=float("inf"),
                debris_gates=[],
            )

    def _find_best_debris_path(self, circuit: Circuit) -> Optional[CancellationPath]:
        """Find the best cancellation path using debris insertion."""
        original_gates = circuit.gates()
        width = circuit.width()

        # Generate potential debris insertions
        debris_insertions = self._generate_debris_insertions(original_gates, width)

        # Use A* search to find the best cancellation path
        best_path = self._astar_search(circuit, debris_insertions)

        return best_path

    def _generate_debris_insertions(
        self, gates: List[Tuple], width: int
    ) -> List[DebrisInsertion]:
        """Generate potential debris gate insertions."""
        insertions = []

        # Try inserting NOT gates (single-qubit)
        for pos in range(len(gates) + 1):
            for target in range(width):
                insertion = DebrisInsertion(
                    position=pos,
                    gate=([], target),  # NOT gate
                    expected_cancellations=self._estimate_cancellations(
                        gates, pos, ([], target)
                    ),
                )
                insertions.append(insertion)

        # Try inserting CNOT gates (two-qubit)
        for pos in range(len(gates) + 1):
            for control in range(width):
                for target in range(width):
                    if control != target:
                        insertion = DebrisInsertion(
                            position=pos,
                            gate=([control], target),  # CNOT gate
                            expected_cancellations=self._estimate_cancellations(
                                gates, pos, ([control], target)
                            ),
                        )
                        insertions.append(insertion)

        # Sort by expected cancellations and limit
        insertions.sort(key=lambda x: x.expected_cancellations, reverse=True)
        return insertions[: self.max_debris_gates * 10]  # Keep top candidates

    def _estimate_cancellations(
        self, gates: List[Tuple], position: int, debris_gate: Tuple
    ) -> int:
        """Estimate how many cancellations a debris gate might enable."""
        controls, target = debris_gate

        # Look for potential cancellation partners
        cancellations = 0

        # Check gates before insertion point
        for i in range(position - 1, max(0, position - 5), -1):
            if i < len(gates):
                gate_controls, gate_target = gates[i]
                if gate_controls == controls and gate_target == target:
                    cancellations += 1
                    break

        # Check gates after insertion point
        for i in range(position, min(len(gates), position + 5)):
            if i < len(gates):
                gate_controls, gate_target = gates[i]
                if gate_controls == controls and gate_target == target:
                    cancellations += 1
                    break

        return cancellations

    def _astar_search(
        self, circuit: Circuit, debris_insertions: List[DebrisInsertion]
    ) -> Optional[CancellationPath]:
        """Use A* search to find the best cancellation path."""
        original_gates = circuit.gates()
        original_count = len(original_gates)

        # Priority queue: (heuristic_score, depth, circuit_state, path, debris_gates)
        queue = [(0, 0, original_gates, [], [])]
        visited = set()

        best_result = None
        best_score = float("inf")

        while queue and len(visited) < self.max_search_depth:
            heuristic_score, depth, current_gates, path, debris_gates = heapq.heappop(
                queue
            )

            # Create circuit from current state
            current_circuit = self._gates_to_circuit(current_gates, circuit.width())

            # Try basic cancellation
            try:
                reduced_circuit, _ = current_circuit.reduce_by_swaps_and_cancellation()
                final_count = len(reduced_circuit.gates())

                if final_count < original_count:
                    # Found a better result
                    complexity_score = self._compute_complexity_score(
                        depth, len(debris_gates), final_count, original_count
                    )

                    if complexity_score < best_score:
                        best_score = complexity_score
                        best_result = CancellationPath(
                            path=path,
                            final_gate_count=final_count,
                            complexity_score=complexity_score,
                            debris_gates=debris_gates,
                        )
            except Exception:
                pass

            # If we've reached max debris gates, don't continue
            if len(debris_gates) >= self.max_debris_gates:
                continue

            # Try inserting debris gates
            for insertion in debris_insertions:
                if insertion.position <= len(current_gates):
                    new_gates = current_gates.copy()
                    new_gates.insert(insertion.position, insertion.gate)

                    # Create state hash
                    state_hash = self._hash_gate_sequence(new_gates)
                    if state_hash in visited:
                        continue

                    visited.add(state_hash)

                    # Compute heuristic (lower is better)
                    heuristic = len(new_gates) + len(debris_gates) * 2 + depth

                    new_path = path + [insertion.position]
                    new_debris = debris_gates + [insertion.gate]

                    heapq.heappush(
                        queue, (heuristic, depth + 1, new_gates, new_path, new_debris)
                    )

        return best_result

    def _gates_to_circuit(self, gates: List[Tuple], width: int) -> Circuit:
        """Convert a list of gates back to a Circuit object."""
        circuit = Circuit(width)
        for controls, target in gates:
            if len(controls) == 0:
                circuit = circuit.x(target)
            elif len(controls) == 1:
                circuit = circuit.cx(controls[0], target)
            elif len(controls) == 2:
                circuit = circuit.mcx(controls, target)
            else:
                circuit = circuit.mcx(controls, target)
        return circuit

    def _hash_gate_sequence(self, gates: List[Tuple]) -> str:
        """Create a hash for a sequence of gates."""
        return str(gates)  # Simple string representation for now

    def _compute_complexity_score(
        self, depth: int, debris_count: int, final_count: int, original_count: int
    ) -> float:
        """Compute a non-triviality score for the cancellation path."""
        # Factors that make cancellation more complex:
        # 1. More debris gates inserted
        # 2. Deeper search required
        # 3. Less reduction achieved

        reduction_ratio = (original_count - final_count) / original_count
        debris_penalty = debris_count * 0.5
        depth_penalty = depth * 0.1

        # Higher score = more complex/non-trivial
        complexity_score = debris_penalty + depth_penalty + (1.0 - reduction_ratio)

        return complexity_score

    def compute_non_triviality_score(self, circuit: Circuit) -> float:
        """
        Compute a non-triviality score for a circuit.
        Higher scores indicate circuits that are harder to simplify.
        """
        # Try basic cancellation first
        basic_result = self._basic_cancellation(circuit)

        if basic_result.final_gate_count == 0:
            return 0.0  # Trivial - fully cancellable

        # Try debris cancellation
        debris_result = self._find_best_debris_path(circuit)

        if (
            debris_result
            and debris_result.final_gate_count < basic_result.final_gate_count
        ):
            return debris_result.complexity_score
        else:
            # No improvement with debris, but still not fully cancellable
            return basic_result.complexity_score + 1.0


class DebrisCancellationManager:
    """Manages debris cancellation analysis for the factory."""

    def __init__(self, database, max_debris_gates: int = 5):
        self.database = database
        self.analyzer = DebrisCancellationAnalyzer(max_debris_gates=max_debris_gates)

    def analyze_dim_group_representative(
        self, dim_group_id: int, circuit_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze the representative circuit of a dimension group for debris cancellation.
        Returns analysis results if significant improvement is found.
        """
        circuit_record = self.database.get_circuit(circuit_id)
        if not circuit_record:
            return None

        # Convert to Circuit object
        circuit = self._record_to_circuit(circuit_record)

        # Analyze for debris cancellation
        cancellation_path = self.analyzer.analyze_circuit(circuit)

        if cancellation_path and cancellation_path.final_gate_count < len(
            circuit.gates()
        ):
            # Store the analysis
            debris_record = DebrisCancellationRecord(
                id=None,
                circuit_id=circuit_id,
                dim_group_id=dim_group_id,
                debris_gates=cancellation_path.debris_gates,
                cancellation_path=cancellation_path.path,
                non_triviality_score=cancellation_path.complexity_score,
                final_gate_count=cancellation_path.final_gate_count,
                cancellation_metrics={
                    "original_gate_count": len(circuit.gates()),
                    "reduction_ratio": (
                        len(circuit.gates()) - cancellation_path.final_gate_count
                    )
                    / len(circuit.gates()),
                    "debris_gate_count": len(cancellation_path.debris_gates),
                    "search_depth": len(cancellation_path.path),
                },
            )

            self.database.store_debris_cancellation(debris_record)

            return {
                "improvement_found": True,
                "original_gate_count": len(circuit.gates()),
                "final_gate_count": cancellation_path.final_gate_count,
                "non_triviality_score": cancellation_path.complexity_score,
                "debris_gates": cancellation_path.debris_gates,
                "cancellation_path": cancellation_path.path,
            }

        return {"improvement_found": False}

    def get_high_complexity_circuits(
        self, threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Get circuits with high non-triviality scores."""
        # This would require a more sophisticated query in a real implementation
        # For now, we'll return a placeholder
        return []

    def _record_to_circuit(self, record: "CircuitRecord") -> Circuit:
        """Convert a CircuitRecord to a Circuit object."""
        circuit = Circuit(record.width)

        # Debug: check what type record.gates is
        logger.info(
            f"record.gates type: {type(record.gates)}, content: {record.gates[:3] if hasattr(record.gates, '__getitem__') else record.gates}"
        )

        for gate in record.gates:
            gate_type = gate[0]
            if gate_type == "NOT":
                circuit = circuit.x(gate[1])
            elif gate_type == "CNOT":
                circuit = circuit.cx(gate[1], gate[2])
            elif gate_type == "TOFFOLI":
                circuit = circuit.mcx([gate[1], gate[2]], gate[3])
            else:
                # Handle multi-controlled gates (convert to TOFFOLI chains)
                # This is a simplified approach - could be optimized
                controls = gate[
                    1:-1
                ]  # All elements except first (type) and last (target)
                target = gate[-1]
                for i in range(len(controls) - 1):
                    circuit = circuit.mcx([controls[i], controls[i + 1]], target)
        return circuit
