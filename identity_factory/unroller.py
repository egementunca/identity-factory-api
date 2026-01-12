"""
Circuit unrolling system for identity circuit factory.
Generates equivalent circuits from representative circuits using various transformations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional import for SAT synthesis
try:
    from circuit.circuit import Circuit

    CIRCUIT_AVAILABLE = True
except ImportError:
    Circuit = None
    CIRCUIT_AVAILABLE = False

from .database import CircuitDatabase, CircuitRecord

logger = logging.getLogger(__name__)


@dataclass
class UnrollResult:
    """Result of unrolling operation."""

    success: bool
    dim_group_id: Optional[int] = None
    total_equivalents: int = 0
    new_circuits: int = 0
    unroll_types: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class CircuitUnroller:
    """Unrolls representative circuits to generate equivalent circuits."""

    def __init__(self, database: CircuitDatabase, max_equivalents: int = 10000):
        self.database = database
        self.max_equivalents = max_equivalents
        self.unroll_count = 0
        self.total_unroll_time = 0.0
        self.unroll_stats = {
            "total_unrolled": 0,
            "average_equivalents_per_group": 0.0,
            "largest_group": 0,
            "smallest_group": float("inf"),
        }

        logger.info(
            f"CircuitUnroller initialized with max_equivalents={max_equivalents}"
        )

    def unroll_dimension_group(
        self, dim_group_id: int, unroll_types: Optional[List[str]] = None
    ) -> UnrollResult:
        """
        Unroll all representatives in a dimension group to generate equivalent circuits.

        Args:
            dim_group_id: ID of the dimension group to unroll
            unroll_types: List of unroll types to apply (e.g., 'sat_revsynth_unroll')

        Returns:
            UnrollResult with generation statistics
        """
        import time

        start_time = time.time()

        logger.info(f"Unrolling dimension group {dim_group_id}")

        dim_group = self.database.get_dim_group_by_id(dim_group_id)
        if not dim_group:
            return UnrollResult(
                success=False, error_message=f"Dimension group {dim_group_id} not found"
            )

        representatives = self.database.get_representatives_in_dim_group(dim_group_id)
        if not representatives:
            return UnrollResult(
                success=False,
                error_message=f"No representatives for group {dim_group_id}",
            )

        if unroll_types is None:
            unroll_types = ["sat_revsynth_unroll"]

        total_new_circuits = 0
        unroll_type_counts = {ut: 0 for ut in unroll_types}

        # Use the comprehensive unroll_circuit method for each representative so that all
        # DFS / ROTATE / MIRROR / PERMUTE equivalents from sat_revsynth are considered and
        # sub-seed merging logic (cleanup_representatives_after_unroll) is applied.
        for circuit_record in representatives:
            try:
                rep_result = self.unroll_circuit(
                    circuit_record, max_equivalents=self.max_equivalents
                )

                if rep_result.get("success"):
                    # unique_equivalents counts new circuits inserted (excluding original)
                    total_new_circuits += rep_result.get("unique_equivalents", 0)

                    # Merge unroll type counts (currently only 'comprehensive')
                    for ut, count in rep_result.get("unroll_types", {}).items():
                        unroll_type_counts[ut] = unroll_type_counts.get(ut, 0) + count
                else:
                    logger.warning(
                        f"Failed to unroll representative {circuit_record.id}: {rep_result.get('error')}"
                    )
            except Exception as e:
                logger.error(
                    f"Critical failure processing representative {circuit_record.id}: {e}"
                )

        self.database.mark_dim_group_processed(dim_group_id)

        # Calculate equivalent count manually in simplified structure
        all_circuits = self.database.get_circuits_in_dim_group(dim_group_id)
        equivalent_count = len([c for c in all_circuits if c.representative_id != c.id])

        # Final stats update
        unroll_time = time.time() - start_time
        logger.info(
            f"Finished unrolling group {dim_group_id} in {unroll_time:.2f}s, generated {total_new_circuits} circuits."
        )

        return UnrollResult(
            success=True,
            dim_group_id=dim_group_id,
            total_equivalents=equivalent_count,
            new_circuits=total_new_circuits,
            unroll_types=unroll_type_counts,
        )

    def _perform_unrolling(
        self, circuit: Circuit, representative: CircuitRecord, unroll_types: List[str]
    ) -> UnrollResult:
        """Perform unrolling on a single representative using specified methods."""
        all_equivalents = []
        unroll_type_counts = {ut: 0 for ut in unroll_types}

        if "sat_revsynth_unroll" in unroll_types:
            all_equivalents = circuit.unroll([])
            unroll_type_counts["sat_revsynth_unroll"] = len(all_equivalents)

        if len(all_equivalents) > self.max_equivalents:
            all_equivalents = all_equivalents[: self.max_equivalents]

        stored_count = 0
        for equiv_circuit in all_equivalents:
            try:
                # Create equivalent circuit record in simplified structure
                from .seed_generator import normalize_circuit_gates

                equiv_gates = normalize_circuit_gates(equiv_circuit.gates())
                equiv_hash = self.database._compute_circuit_hash(
                    equiv_gates, equiv_circuit.width()
                )

                # Check if this equivalent already exists
                if self.database.circuit_exists(equiv_hash):
                    continue

                # Create new circuit record as equivalent
                equiv_record = CircuitRecord(
                    id=0,  # Will be set by database
                    width=equiv_circuit.width(),
                    gate_count=len(equiv_gates),
                    gates=equiv_gates,
                    permutation=list(
                        range(2 ** equiv_circuit.width())
                    ),  # Identity permutation
                    complexity_walk=None,
                    circuit_hash=equiv_hash,
                    dim_group_id=representative.dim_group_id,
                    representative_id=representative.id,  # Point to the representative
                )

                circuit_id = self.database.store_circuit(equiv_record)
                if circuit_id > 0:
                    stored_count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to store equivalent for rep {representative.id}: {e}"
                )

        logger.info(
            f"Stored {stored_count} new equivalents for representative {representative.id}"
        )
        return UnrollResult(
            success=True, new_circuits=stored_count, unroll_types=unroll_type_counts
        )

    def _record_to_circuit(self, record: CircuitRecord) -> Circuit:
        """Converts a CircuitRecord from the database to a sat_revsynth Circuit object.

        Handles multiple gate formats:
        1. NCT tuples: ('X', t), ('CX', c, t), ('CCX', c1, c2, t)
        2. Controls/target format: ([c1, c2], t)
        3. ECA57 format: [c1, c2, t] - 3-element list where last is target
        """
        try:
            if not isinstance(record.gates, list):
                raise TypeError(f"Malformed gates for circuit {record.id}: not a list.")

            # Use the constructor of the Circuit class
            new_circuit = Circuit(record.width)

            for gate in record.gates:
                if isinstance(gate, (list, tuple)):
                    # NCT format: ('X', t) or ('CX', c, t) or ('CCX', c1, c2, t)
                    if len(gate) >= 2 and isinstance(gate[0], str):
                        gate_type = gate[0]
                        if gate_type == "X":
                            new_circuit = new_circuit.x(gate[1])
                        elif gate_type == "CX":
                            new_circuit = new_circuit.cx(gate[1], gate[2])
                        elif gate_type == "CCX":
                            new_circuit = new_circuit.mcx([gate[1], gate[2]], gate[3])
                        else:
                            logger.warning(f"Unknown gate type: {gate_type}")
                    # Controls/target format: ([c1, c2], t)
                    elif (
                        len(gate) == 2
                        and isinstance(gate[0], (list, tuple))
                        and isinstance(gate[1], int)
                    ):
                        controls, target = gate
                        if len(controls) == 0:
                            new_circuit = new_circuit.x(target)
                        elif len(controls) == 1:
                            new_circuit = new_circuit.cx(controls[0], target)
                        else:
                            new_circuit = new_circuit.mcx(list(controls), target)
                    # ECA57 format: [c1, c2, t] - all integers, 3-element list
                    elif len(gate) == 3 and all(isinstance(x, int) for x in gate):
                        ctrl1, ctrl2, target = gate
                        new_circuit = new_circuit.mcx([ctrl1, ctrl2], target)
                    else:
                        logger.warning(
                            f"Unknown gate format for circuit {record.id}: {gate}"
                        )
                else:
                    logger.warning(
                        f"Malformed gate data in DB for circuit {record.id}: {gate}"
                    )

            return new_circuit

        except Exception as e:
            logger.error(
                f"Failed to convert CircuitRecord {record.id} to Circuit object: {e}"
            )
            raise

    def unroll_circuit(
        self, circuit_record: CircuitRecord, max_equivalents: int = 100
    ) -> Dict[str, Any]:
        """
        Unroll a single circuit to generate all equivalent circuits.
        This uses the comprehensive unroll method from sat_revsynth which includes:
        - Swap space exploration (BFS)
        - Rotations
        - Reverse
        - Permutations
        """
        try:
            # Convert to sat_revsynth Circuit
            circuit = self._record_to_circuit(circuit_record)

            logger.info(
                f"Starting comprehensive unroll for circuit {circuit_record.id}"
            )

            # Use the comprehensive unroll from sat_revsynth
            # This includes swap_space_bfs + rotations + reverse + permutations
            equivalent_circuits = circuit.unroll()

            logger.info(
                f"Unroll generated {len(equivalent_circuits)} total equivalents"
            )

            # Check if we hit the limit (meaning we might not have ALL equivalents)
            hit_limit = len(equivalent_circuits) >= max_equivalents

            # Limit the number of equivalents if specified
            if hit_limit:
                logger.info(
                    f"Limiting equivalents from {len(equivalent_circuits)} to {max_equivalents}"
                )
                equivalent_circuits = equivalent_circuits[:max_equivalents]

            # Convert circuits back to gate lists AND STORE them in DB
            equivalents_as_gates: List[List[Tuple]] = []
            stored_count = 0
            for equiv_circuit in equivalent_circuits:
                gates = equiv_circuit.gates()
                # Skip the original circuit
                if gates == circuit_record.gates:
                    continue

                equivalents_as_gates.append(gates)

                try:
                    # Create equivalent circuit record in simplified structure
                    from .seed_generator import normalize_circuit_gates

                    equiv_gates = normalize_circuit_gates(gates)
                    equiv_hash = self.database._compute_circuit_hash(
                        equiv_gates, circuit_record.width
                    )

                    # Check if this equivalent already exists
                    if self.database.circuit_exists(equiv_hash):
                        continue

                    # Create new circuit record as equivalent
                    equiv_record = CircuitRecord(
                        id=0,  # Will be set by database
                        width=circuit_record.width,
                        gate_count=len(equiv_gates),
                        gates=equiv_gates,
                        permutation=list(
                            range(2**circuit_record.width)
                        ),  # Identity permutation
                        complexity_walk=None,
                        circuit_hash=equiv_hash,
                        dim_group_id=circuit_record.dim_group_id,
                        representative_id=circuit_record.id,  # Point to the representative
                    )

                    equiv_id = self.database.store_circuit(equiv_record)
                    if equiv_id > 0:
                        stored_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to store equivalent for circuit {circuit_record.id}: {e}"
                    )

            result = {
                "success": True,
                "equivalents": equivalents_as_gates,
                "total_generated": len(equivalent_circuits),
                "unique_equivalents": len(equivalents_as_gates),
                "stored_equivalents": stored_count,
                "original_excluded": len(equivalent_circuits)
                - len(equivalents_as_gates),
                "fully_unrolled": not hit_limit,  # True if we didn't hit the limit
                "unroll_types": {"comprehensive": len(equivalent_circuits)},
            }

            # Simplified cleanup since we don't have the complex representative management
            # Just return the basic result for now
            return result

        except Exception as e:
            logger.error(f"Unroll failed for circuit {circuit_record.id}: {e}")
            return {"success": False, "error": str(e), "equivalents": []}

    def get_unroll_stats(self) -> Dict[str, Any]:
        """Get statistics about unrolling operations."""
        circuits_unrolled = getattr(self, "circuits_unrolled", 0)
        total_equivalents_generated = getattr(self, "total_equivalents_generated", 0)

        return {
            "total_unroll_time": self.total_unroll_time,
            "circuits_unrolled": circuits_unrolled,
            "total_equivalents_generated": total_equivalents_generated,
            "average_unroll_time": self.total_unroll_time / max(1, circuits_unrolled),
            "average_equivalents_per_circuit": total_equivalents_generated
            / max(1, circuits_unrolled),
        }
