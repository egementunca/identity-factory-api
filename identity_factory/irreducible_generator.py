"""
Irreducible Circuit Generator.

Generates forward circuits that touch all wires using a pattern-based approach.
"""

import logging
import random
from typing import List, Optional, Tuple

from .irreducible_db import ForwardCircuit, IrreducibleDatabase

logger = logging.getLogger(__name__)


class IrreducibleGenerator:
    """
    Generate irreducible circuits using width × repetitions pattern.

    Each repetition creates a pattern of 'width' gates where every wire
    is a target exactly once.
    """

    def __init__(self, database: IrreducibleDatabase):
        self.database = database

    def generate(self, width: int, repetitions: int) -> ForwardCircuit:
        """
        Generate circuit with 'repetitions' cycles of 'width' gates.
        Each cycle ensures all wires are touched.

        Args:
            width: Number of qubits
            repetitions: Number of times to repeat the all-wire pattern

        Returns:
            ForwardCircuit with width × repetitions gates
        """
        total_gates = width * repetitions

        # Generate all gates by repeating pattern
        all_gates = []
        for rep in range(repetitions):
            pattern = self._generate_all_wire_pattern(width)
            all_gates.extend(pattern)

        # Compute permutation
        permutation = self._compute_permutation(all_gates, width)

        # Create circuit object
        circuit = ForwardCircuit(
            id=None,
            width=width,
            gate_count=total_gates,
            gates=all_gates,
            permutation=permutation,
            permutation_hash=IrreducibleDatabase.compute_permutation_hash(permutation),
        )

        logger.info(f"Generated {width}w × {repetitions}r = {total_gates}g circuit")
        return circuit

    def _generate_all_wire_pattern(self, width: int) -> List[Tuple[int, int, int]]:
        """
        Generate 'width' gates that touch all wires.

        Each wire becomes a target exactly once in the pattern.
        Controls are chosen randomly from other wires.

        Args:
            width: Number of qubits

        Returns:
            List of gates [(c1, c2, target), ...]
        """
        gates = []
        targets = list(range(width))
        random.shuffle(targets)

        for target in targets:
            # Pick 2 other wires for controls
            controls = [w for w in range(width) if w != target]
            c1, c2 = random.sample(controls, 2)
            gates.append((c1, c2, target))

        return gates

    def _compute_permutation(
        self, gates: List[Tuple[int, int, int]], width: int
    ) -> List[int]:
        """
        Compute the permutation implemented by a sequence of ECA57 gates.

        ECA57 gate: target ^= (ctrl1 OR NOT ctrl2)

        Args:
            gates: List of gates [(c1, c2, target), ...]
            width: Number of qubits

        Returns:
            Permutation as list [output_0, output_1, ..., output_n-1]
        """
        n = 2**width
        perm = list(range(n))

        for c1, c2, target in gates:
            new_perm = perm.copy()
            for i in range(n):
                # Get bit values
                b_c1 = (i >> c1) & 1
                b_c2 = (i >> c2) & 1
                b_target = (i >> target) & 1

                # ECA57 control: c1 OR (NOT c2)
                control = b_c1 | (1 - b_c2)

                if control:
                    # Flip target bit
                    flipped = i ^ (1 << target)
                    new_perm[i] = perm[flipped]

            perm = new_perm

        return perm

    def generate_batch(
        self, width: int, repetitions: int, count: int, store: bool = True
    ) -> List[ForwardCircuit]:
        """
        Generate multiple circuits with same dimensions.

        Args:
            width: Number of qubits
            repetitions: Pattern repetitions
            count: Number of circuits to generate
            store: Whether to store in database

        Returns:
            List of generated circuits
        """
        circuits = []

        for i in range(count):
            circuit = self.generate(width, repetitions)

            if store:
                circuit.id = self.database.store_forward(circuit)

            circuits.append(circuit)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{count} circuits")

        logger.info(f"Batch complete: {count} circuits for {width}w × {repetitions}r")
        return circuits

    def generate_all_dimensions(
        self, dimensions: List[Tuple[int, int]], circuits_per_dimension: int = 1000
    ) -> dict:
        """
        Generate circuits for all specified dimensions.

        Args:
            dimensions: List of (width, repetitions) tuples
            circuits_per_dimension: How many circuits per dimension

        Returns:
            Statistics dict
        """
        stats = {"total_generated": 0, "by_dimension": {}}

        for width, reps in dimensions:
            logger.info(f"Starting dimension {width}w × {reps}r...")

            circuits = self.generate_batch(
                width=width, repetitions=reps, count=circuits_per_dimension, store=True
            )

            stats["total_generated"] += len(circuits)
            stats["by_dimension"][f"{width}w_{reps}r"] = len(circuits)

        logger.info(
            f"All dimensions complete: {stats['total_generated']} total circuits"
        )
        return stats


def demo_generate():
    """Demo circuit generation."""
    from pathlib import Path

    db = IrreducibleDatabase()
    generator = IrreducibleGenerator(db)

    # Generate one circuit for each dimension
    dimensions = [
        (3, 5),  # 15 gates
        (4, 4),  # 16 gates
        (5, 3),  # 15 gates
        (6, 2),  # 12 gates
        (7, 1),  # 7 gates
    ]

    for width, reps in dimensions:
        circuit = generator.generate(width, reps)
        print(f"{width}w × {reps}r = {circuit.gate_count}g")
        print(f"  Permutation: {circuit.permutation[:8]}...")
        print(f"  First 3 gates: {circuit.gates[:3]}")
        print()

    db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_generate()
