#!/usr/bin/env python3
"""Test script for local mixing utilities."""

import os
import sys

# Add parent to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random

# Import directly from the module file (not through __init__.py)
import re
from typing import List, Optional, Tuple


def parse_circuit_string(circuit_str: str) -> List[Tuple[int, int, int]]:
    """Parse a circuit string into a list of gate tuples."""
    gates = []
    pattern = r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]"
    matches = re.findall(pattern, circuit_str)
    for match in matches:
        gates.append((int(match[0]), int(match[1]), int(match[2])))
    return gates


def gates_to_string(gates: List[Tuple[int, int, int]]) -> str:
    """Convert a list of gate tuples back to circuit string format."""
    return " ".join(f"[{g[0]},{g[1]},{g[2]}]" for g in gates)


def compute_permutation(gates: List[Tuple[int, int, int]], num_wires: int) -> List[int]:
    """Compute the permutation implemented by a circuit of ECA57 gates."""
    size = 1 << num_wires
    perm = list(range(size))

    for input_val in range(size):
        output_val = input_val
        for active, c1, c2 in gates:
            bit_c1 = (output_val >> c1) & 1
            bit_c2 = (output_val >> c2) & 1
            if bit_c1 == 1 and bit_c2 == 0:
                output_val ^= 1 << active
        perm[input_val] = output_val

    return perm


def canonicalize_simple(
    gates: List[Tuple[int, int, int]],
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Simple Python canonicalization."""
    if not gates:
        return [], 0

    result = list(gates)
    removals = 0
    changed = True

    while changed:
        changed = False
        i = 1
        while i < len(result):
            g1, g2 = result[i - 1], result[i]
            collision = (
                g1[0] == g2[1] or g1[0] == g2[2] or g2[0] == g1[1] or g2[0] == g1[2]
            )
            if not collision and g2 < g1:
                result[i - 1], result[i] = result[i], result[i - 1]
                changed = True
            i += 1

        i = 0
        while i < len(result) - 1:
            if result[i] == result[i + 1]:
                del result[i : i + 2]
                removals += 2
                changed = True
                if i > 0:
                    i -= 1
            else:
                i += 1

    return result, removals


def generate_random_gate(num_wires: int) -> Tuple[int, int, int]:
    """Generate a random valid ECA57 gate."""
    wires = list(range(num_wires))
    random.shuffle(wires)
    return (wires[0], wires[1], wires[2])


def generate_random_circuit(
    num_wires: int, num_gates: int
) -> List[Tuple[int, int, int]]:
    """Generate a random circuit."""
    return [generate_random_gate(num_wires) for _ in range(num_gates)]


def generate_random_identity(
    num_wires: int, half_size: int
) -> List[Tuple[int, int, int]]:
    """Generate a random identity circuit (R followed by R^-1)."""
    r_gates = generate_random_circuit(num_wires, half_size)
    r_inv_gates = list(reversed(r_gates))
    return r_gates + r_inv_gates


if __name__ == "__main__":
    print("Testing local_mixing_utils functions...\n")

    # Test parse
    gates = parse_circuit_string("[0,1,2] [1,2,0] [2,0,1]")
    print(f"✓ Parsed gates: {gates}")
    assert gates == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    # Test string conversion
    s = gates_to_string(gates)
    print(f"✓ Gates to string: {s}")
    assert s == "[0,1,2] [1,2,0] [2,0,1]"

    # Test permutation
    perm = compute_permutation(gates, 3)
    print(f"✓ Permutation computed: {perm}")
    assert len(perm) == 8

    # Test canonicalize with identity circuit from COMPLETE_TRACE.md
    identity_gates = parse_circuit_string(
        "[1,0,3] [1,4,2] [0,4,1] [4,2,0] [4,3,0] [4,3,0] [4,2,0] [0,4,1] [1,4,2] [1,0,3]"
    )
    print(f"  Original: {len(identity_gates)} gates")
    canonical, removals = canonicalize_simple(identity_gates)
    print(
        f"✓ Canonicalization: {len(identity_gates)} → {len(canonical)} gates (removed {removals})"
    )

    # Test random circuit
    random_gates = generate_random_circuit(5, 10)
    print(f"✓ Random circuit: {len(random_gates)} gates")
    assert len(random_gates) == 10

    # Test random identity
    id_gates = generate_random_identity(4, 5)
    id_perm = compute_permutation(id_gates, 4)
    is_identity = all(id_perm[i] == i for i in range(len(id_perm)))
    print(f"✓ Random identity: {len(id_gates)} gates, is_identity={is_identity}")
    assert is_identity, f"Random identity should be identity permutation, got {id_perm}"

    print("\n✅ All local_mixing_utils tests passed!")
