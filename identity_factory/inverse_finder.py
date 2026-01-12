"""
Inverse Circuit Finder.

Synthesizes inverse circuits for given forward circuits using SAT solvers.
"""

import logging
from typing import List, Optional, Tuple

from .irreducible_db import ForwardCircuit, InverseCircuit, IrreducibleDatabase

logger = logging.getLogger(__name__)


class InverseFinder:
    """
    Find inverse circuits using various methods.

    Methods:
    - 'reverse': Simply reverse the gate order (works for self-inverse gates)
    - 'sat_optimal': Use SAT solver to find minimum-gate inverse
    - 'sat_bounded': Use SAT solver with bounded gate count
    """

    def __init__(self, database: IrreducibleDatabase, method: str = "sat_optimal"):
        self.database = database
        self.method = method

        # Import SAT synthesis tools
        try:
            from eca57.optimal_synthesizer import ECA57OptimalSynthesizer
            from eca57.truth_table import ECA57TruthTable
            from sat.solver import Solver

            self.sat_available = True
        except ImportError:
            logger.warning("SAT synthesis not available, will use reverse method")
            self.sat_available = False
            self.method = "reverse"

    def find_inverse(self, forward: ForwardCircuit) -> InverseCircuit:
        """
        Find inverse circuit for given forward circuit.

        Args:
            forward: Forward circuit to invert

        Returns:
            InverseCircuit implementing the inverse permutation
        """
        if self.method == "reverse":
            return self._find_by_reverse(forward)
        elif self.method == "sat_optimal":
            return self._find_by_sat_optimal(forward)
        elif self.method == "sat_bounded":
            return self._find_by_sat_bounded(forward)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _find_by_reverse(self, forward: ForwardCircuit) -> InverseCircuit:
        """
        Find inverse by reversing gate order.

        Works for ECA57 gates which are self-inverse.
        """
        # ECA57 gates are self-inverse, so just reverse the order
        inverse_gates = list(reversed(forward.gates))

        inverse = InverseCircuit(
            id=None,
            forward_id=forward.id,
            gate_count=len(inverse_gates),
            gates=inverse_gates,
            synthesis_method="reverse",
        )

        logger.info(f"Found inverse by reverse: {inverse.gate_count} gates")
        return inverse

    def _invert_permutation(self, perm: List[int]) -> List[int]:
        """Compute inverse permutation."""
        n = len(perm)
        inv = [0] * n
        for i in range(n):
            inv[perm[i]] = i
        return inv

    def _find_by_sat_optimal(self, forward: ForwardCircuit) -> InverseCircuit:
        """
        Find optimal inverse using SAT solver.

        Searches for minimum-gate circuit implementing inverse permutation.
        """
        if not self.sat_available:
            logger.warning("SAT not available, falling back to reverse")
            return self._find_by_reverse(forward)

        from eca57.optimal_synthesizer import ECA57OptimalSynthesizer
        from eca57.truth_table import ECA57TruthTable
        from sat.solver import Solver

        # Compute inverse permutation
        inv_perm = self._invert_permutation(forward.permutation)

        # Create truth table
        tt = ECA57TruthTable(forward.width, values=inv_perm)

        # Create solver
        solver = Solver("minisat-gh")

        # Search for optimal circuit
        # Start from 1 gate up to forward_length + 5
        synth = ECA57OptimalSynthesizer(
            output=tt,
            lower_gc=1,
            upper_gc=forward.gate_count + 5,
            solver=solver,
            anti_trivial=True,
        )

        logger.info(
            f"Searching for optimal inverse (max {forward.gate_count + 5} gates)..."
        )
        circuit = synth.solve()

        if circuit is None:
            logger.warning("SAT synthesis failed, falling back to reverse")
            return self._find_by_reverse(forward)

        # Convert to our format
        inverse_gates = circuit.gates_as_tuples()

        inverse = InverseCircuit(
            id=None,
            forward_id=forward.id,
            gate_count=len(inverse_gates),
            gates=inverse_gates,
            synthesis_method="sat_optimal",
        )

        logger.info(
            f"Found optimal inverse: {inverse.gate_count} gates (vs forward {forward.gate_count})"
        )
        return inverse

    def _find_by_sat_bounded(
        self, forward: ForwardCircuit, extra_gates: int = 2
    ) -> InverseCircuit:
        """
        Find inverse using SAT with bounded gate count.

        Searches up to forward_length + extra_gates.
        """
        if not self.sat_available:
            logger.warning("SAT not available, falling back to reverse")
            return self._find_by_reverse(forward)

        from eca57.optimal_synthesizer import ECA57OptimalSynthesizer
        from eca57.truth_table import ECA57TruthTable
        from sat.solver import Solver

        # Compute inverse permutation
        inv_perm = self._invert_permutation(forward.permutation)

        # Create truth table
        tt = ECA57TruthTable(forward.width, values=inv_perm)

        # Create solver
        solver = Solver("minisat-gh")

        # Search with bounded gate count
        max_gates = forward.gate_count + extra_gates
        synth = ECA57OptimalSynthesizer(
            output=tt, lower_gc=1, upper_gc=max_gates, solver=solver, anti_trivial=True
        )

        logger.info(f"Searching for bounded inverse (max {max_gates} gates)...")
        circuit = synth.solve()

        if circuit is None:
            logger.warning("SAT synthesis failed, falling back to reverse")
            return self._find_by_reverse(forward)

        # Convert to our format
        inverse_gates = circuit.gates_as_tuples()

        inverse = InverseCircuit(
            id=None,
            forward_id=forward.id,
            gate_count=len(inverse_gates),
            gates=inverse_gates,
            synthesis_method="sat_bounded",
        )

        logger.info(f"Found bounded inverse: {inverse.gate_count} gates")
        return inverse

    def find_and_store(self, forward: ForwardCircuit) -> InverseCircuit:
        """
        Find inverse and store in database.

        Args:
            forward: Forward circuit (must have id set)

        Returns:
            InverseCircuit with id set
        """
        if forward.id is None:
            raise ValueError("Forward circuit must be stored first (id is None)")

        inverse = self.find_inverse(forward)
        inverse.id = self.database.store_inverse(inverse)

        return inverse

    def batch_find_inverses(
        self, forward_circuits: List[ForwardCircuit], store: bool = True
    ) -> List[InverseCircuit]:
        """
        Find inverses for multiple forward circuits.

        Args:
            forward_circuits: List of forward circuits
            store: Whether to store in database

        Returns:
            List of inverse circuits
        """
        inverses = []

        for i, fwd in enumerate(forward_circuits):
            try:
                if store:
                    inv = self.find_and_store(fwd)
                else:
                    inv = self.find_inverse(fwd)

                inverses.append(inv)

                if (i + 1) % 10 == 0:
                    logger.info(f"Found {i + 1}/{len(forward_circuits)} inverses")

            except Exception as e:
                logger.error(f"Failed to find inverse for circuit {fwd.id}: {e}")

        logger.info(f"Batch complete: {len(inverses)} inverses found")
        return inverses


def demo_inverse():
    """Demo inverse finding."""
    from .irreducible_generator import IrreducibleGenerator

    db = IrreducibleDatabase()
    generator = IrreducibleGenerator(db)
    finder = InverseFinder(db, method="reverse")

    # Generate a forward circuit
    forward = generator.generate(width=3, repetitions=2)
    forward.id = db.store_forward(forward)

    print(f"Forward: {forward.gate_count} gates")
    print(f"  Permutation: {forward.permutation}")
    print(f"  Gates: {forward.gates}")

    # Find inverse
    inverse = finder.find_and_store(forward)

    print(f"\nInverse: {inverse.gate_count} gates")
    print(f"  Method: {inverse.synthesis_method}")
    print(f"  Gates: {inverse.gates}")

    db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_inverse()
