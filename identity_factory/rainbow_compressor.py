"""
Rainbow Table Compression for Large Circuits.

The key insight: Even a 40-wire, 500-gate circuit contains many small subcircuits
that only touch 3-4 wires. We can:
1. Extract subcircuit windows
2. Identify which wires they use
3. Remap to (0,1,2,...) for lookup
4. Replace with smaller equivalent
5. Remap back to original wires
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RainbowTableCompressor:
    """
    Compress large circuits using pre-computed rainbow tables.

    Works by:
    1. Sliding window over circuit
    2. Finding subcircuits that use few unique wires
    3. Remapping to canonical wire indices (0,1,2,...)
    4. Looking up in rainbow table
    5. Replacing with smaller equivalent
    """

    def __init__(self, max_lookup_wires: int = 3):
        """
        Initialize compressor.

        Args:
            max_lookup_wires: Maximum wires for rainbow table lookup (we have 3-wire tables)
        """
        self.max_lookup_wires = max_lookup_wires
        self.rainbow_table: Dict[Tuple, List[Tuple]] = {}
        self._load_rainbow_table()

    def _load_rainbow_table(self):
        """Load rainbow table from exported JSON file."""
        import json

        json_path = Path(__file__).parent.parent / "go-proj" / "rainbow_table_3w.json"

        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    entries = json.load(f)

                for entry in entries:
                    perm = tuple(entry["permutation"])
                    gates = [tuple(g) for g in entry["gates"]]

                    # Only keep smallest circuit for each permutation
                    if perm not in self.rainbow_table or len(gates) < len(
                        self.rainbow_table[perm]
                    ):
                        self.rainbow_table[perm] = gates

                logger.info(
                    f"Rainbow table loaded: {len(self.rainbow_table)} unique permutations"
                )
            except Exception as e:
                logger.warning(f"Failed to load rainbow table: {e}")
                # Fallback: just identity
                identity_3 = tuple(range(8))
                self.rainbow_table[identity_3] = []
        else:
            logger.warning(f"Rainbow table not found at {json_path}")
            # Fallback: just identity
            identity_3 = tuple(range(8))
            self.rainbow_table[identity_3] = []

    def extract_subcircuit_wires(
        self, gates: List[Tuple], start: int, end: int
    ) -> Set[int]:
        """Get all wires used by gates in range [start, end)."""
        wires = set()
        for i in range(start, min(end, len(gates))):
            gate = gates[i]
            if isinstance(gate, (list, tuple)):
                for element in gate:
                    if isinstance(element, int):
                        wires.add(element)
        return wires

    def remap_gates(self, gates: List[Tuple], wire_map: Dict[int, int]) -> List[Tuple]:
        """Remap wire indices in gates using the given mapping."""
        remapped = []
        for gate in gates:
            if isinstance(gate, (list, tuple)):
                new_gate = []
                for element in gate:
                    if isinstance(element, int):
                        new_gate.append(wire_map.get(element, element))
                    else:
                        new_gate.append(element)
                remapped.append(tuple(new_gate))
            else:
                remapped.append(gate)
        return remapped

    def compute_permutation(
        self, gates: List[Tuple], num_wires: int
    ) -> Tuple[int, ...]:
        """
        Compute the permutation implemented by a sequence of gates.

        Each gate is (ctrl1, ctrl2, target) implementing target ^= (ctrl1 OR NOT ctrl2)
        """
        n = 2**num_wires
        perm = list(range(n))

        for gate in gates:
            if len(gate) == 3:
                c1, c2, target = gate
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
                        new_target = 1 - b_target
                        new_val = i ^ (1 << target) if new_target != b_target else i
                        new_perm[i] = new_val
                perm = new_perm

        return tuple(perm)

    def find_replacement(
        self, gates: List[Tuple], num_wires: int
    ) -> Optional[List[Tuple]]:
        """
        Look up gates in rainbow table and return smaller replacement if exists.
        """
        perm = self.compute_permutation(gates, num_wires)

        if perm in self.rainbow_table:
            replacement = self.rainbow_table[perm]
            if len(replacement) < len(gates):
                return replacement

        return None

    def compress(
        self,
        gates: List[Tuple],
        circuit_width: int,
        window_sizes: List[int] = [4, 6, 8],
        trials: int = 10000,
    ) -> Tuple[List[Tuple], Dict]:
        """
        Compress a circuit using rainbow table lookup.

        Args:
            gates: List of gates, each as (ctrl1, ctrl2, target)
            circuit_width: Total number of wires in circuit
            window_sizes: Subcircuit window sizes to try
            trials: Number of random windows to try

        Returns:
            Tuple of (compressed_gates, metrics_dict)
        """
        current_gates = list(gates)
        original_len = len(gates)
        replacements_made = 0

        for trial in range(trials):
            if not current_gates:
                break

            # Pick random window size and position
            window_size = random.choice(window_sizes)
            if len(current_gates) < window_size:
                window_size = len(current_gates)

            start = random.randint(0, len(current_gates) - window_size)
            end = start + window_size

            # Get wires used by this subcircuit
            wires_used = self.extract_subcircuit_wires(current_gates, start, end)

            # Only try if subcircuit uses few enough wires for our table
            if len(wires_used) > self.max_lookup_wires:
                continue

            # Create wire mapping: original -> canonical (0,1,2,...)
            wire_list = sorted(wires_used)
            wire_map = {w: i for i, w in enumerate(wire_list)}
            inverse_map = {i: w for w, i in wire_map.items()}

            # Extract and remap subcircuit
            subcircuit = current_gates[start:end]
            remapped = self.remap_gates(subcircuit, wire_map)

            # Look up in rainbow table
            replacement = self.find_replacement(remapped, len(wires_used))

            if replacement is not None and len(replacement) < len(subcircuit):
                # Remap replacement back to original wires
                restored = self.remap_gates(replacement, inverse_map)

                # Replace in circuit
                current_gates = current_gates[:start] + restored + current_gates[end:]
                replacements_made += 1

                if trial % 1000 == 0:
                    logger.info(
                        f"Trial {trial}: {len(subcircuit)} -> {len(replacement)} gates"
                    )

        metrics = {
            "original_gates": original_len,
            "compressed_gates": len(current_gates),
            "compression_ratio": (
                len(current_gates) / original_len if original_len > 0 else 1.0
            ),
            "replacements_made": replacements_made,
            "trials_used": trials,
        }

        return current_gates, metrics


def demo_compress(width: int = 40, gate_count: int = 500):
    """Demo compression on a random large circuit."""
    import random

    # Generate random circuit
    gates = []
    for _ in range(gate_count):
        wires = random.sample(range(width), 3)
        gates.append(tuple(wires))

    print(f"Generated {gate_count}-gate, {width}-wire random circuit")

    # Compress
    compressor = RainbowTableCompressor(max_lookup_wires=3)
    compressed, metrics = compressor.compress(gates, width, trials=10000)

    print(f"Results:")
    print(f"  Original: {metrics['original_gates']} gates")
    print(f"  Compressed: {metrics['compressed_gates']} gates")
    print(f"  Ratio: {metrics['compression_ratio']:.2%}")
    print(f"  Replacements: {metrics['replacements_made']}")

    return compressed, metrics


if __name__ == "__main__":
    demo_compress()
