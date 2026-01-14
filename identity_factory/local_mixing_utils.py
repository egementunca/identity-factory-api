"""
Utility functions for local mixing operations.

This module provides circuit parsing, permutation computation, and
CLI wrappers for the local_mixing Rust binary.
"""

import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_circuit_string(circuit_str: str) -> List[Tuple[int, int, int]]:
    """
    Parse a circuit string into a list of gate tuples.

    Format: "[0,1,2] [1,2,0] [2,0,1]"
    Each gate is (active, control1, control2)

    Args:
        circuit_str: Circuit in bracket notation

    Returns:
        List of (active, control1, control2) tuples
    """
    gates = []
    # Match patterns like [0,1,2] or [10, 20, 30]
    pattern = r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]"
    matches = re.findall(pattern, circuit_str)

    for match in matches:
        gates.append((int(match[0]), int(match[1]), int(match[2])))

    return gates


def gates_to_string(gates: List[Tuple[int, int, int]]) -> str:
    """
    Convert a list of gate tuples back to circuit string format.

    Args:
        gates: List of (active, control1, control2) tuples

    Returns:
        Circuit string in bracket notation
    """
    return " ".join(f"[{g[0]},{g[1]},{g[2]}]" for g in gates)


def compute_permutation(gates: List[Tuple[int, int, int]], num_wires: int) -> List[int]:
    """
    Compute the permutation implemented by a circuit of ECA57 gates.

    Each gate [a, c1, c2] flips wire 'a' if c1=1 and c2=0.
    This is the 57th elementary cellular automaton rule.

    Args:
        gates: List of (active, control1, control2) tuples
        num_wires: Number of wires in the circuit

    Returns:
        Permutation as list where perm[i] = output for input i
    """
    size = 1 << num_wires  # 2^num_wires
    perm = list(range(size))

    for input_val in range(size):
        output_val = input_val
        for active, c1, c2 in gates:
            # Check if control conditions are met
            bit_c1 = (output_val >> c1) & 1
            bit_c2 = (output_val >> c2) & 1
            if bit_c1 == 1 and bit_c2 == 0:
                # Flip the active wire
                output_val ^= 1 << active
        perm[input_val] = output_val

    return perm


def is_identity_permutation(perm: List[int]) -> bool:
    """Check if a permutation is the identity."""
    return perm == list(range(len(perm)))


def generate_random_gate(num_wires: int) -> Tuple[int, int, int]:
    """Generate a random valid ECA57 gate."""
    wires = list(range(num_wires))
    random.shuffle(wires)
    return (wires[0], wires[1], wires[2])


def generate_random_circuit(
    num_wires: int, num_gates: int
) -> List[Tuple[int, int, int]]:
    """
    Generate a random circuit.

    Args:
        num_wires: Number of wires
        num_gates: Number of gates

    Returns:
        List of random gates
    """
    return [generate_random_gate(num_wires) for _ in range(num_gates)]


def generate_random_identity(
    num_wires: int, half_size: int
) -> List[Tuple[int, int, int]]:
    """
    Generate a random identity circuit (R followed by R^-1).

    Args:
        num_wires: Number of wires
        half_size: Number of gates in R (total will be 2x)

    Returns:
        Identity circuit gates
    """
    r_gates = generate_random_circuit(num_wires, half_size)
    r_inv_gates = list(reversed(r_gates))
    return r_gates + r_inv_gates


def canonicalize_simple(
    gates: List[Tuple[int, int, int]],
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Simple Python canonicalization that moves non-colliding gates.

    Two gates collide if one's active wire equals another's control wire.
    Non-colliding gates can be reordered to bring duplicates together.

    Args:
        gates: List of gate tuples

    Returns:
        (canonicalized gates, number of removals)
    """
    if not gates:
        return [], 0

    result = list(gates)
    removals = 0
    changed = True

    while changed:
        changed = False

        # Try to move gates to canonical position
        i = 1
        while i < len(result):
            # Check if gate[i] can move before gate[i-1]
            g1, g2 = result[i - 1], result[i]

            # Check collision: active wire of one touches control of other
            collision = (
                g1[0] == g2[1]
                or g1[0] == g2[2]  # g1 active hits g2 control
                or g2[0] == g1[1]
                or g2[0] == g1[2]  # g2 active hits g1 control
            )

            if not collision:
                # Can reorder - check if g2 should come before g1 (lexicographic)
                if g2 < g1:
                    result[i - 1], result[i] = result[i], result[i - 1]
                    changed = True
            i += 1

        # Remove adjacent duplicates (g ⊙ g = identity)
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


# Path to local_mixing Rust binary
LOCAL_MIXING_DIR = Path(__file__).parent.parent.parent / "local_mixing"
LOCAL_MIXING_BIN = LOCAL_MIXING_DIR / "target" / "release" / "local_mixing_bin"


def get_rust_binary_path() -> Optional[Path]:
    """Get path to the local_mixing Rust binary if it exists."""
    if LOCAL_MIXING_BIN.exists():
        return LOCAL_MIXING_BIN

    # Try debug build
    debug_bin = LOCAL_MIXING_DIR / "target" / "debug" / "local_mixing_bin"
    if debug_bin.exists():
        return debug_bin

    return None


def write_circuit_to_temp_file(circuit_str: str) -> str:
    """Write circuit to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="circuit_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(circuit_str)
    except:
        os.close(fd)
        raise
    return path


def run_heatmap_cli(
    circuit_one: str,
    circuit_two: str,
    num_wires: int,
    num_inputs: int,
) -> Dict[str, Any]:
    """
    Run the heatmap CLI command and parse results.

    Returns raw heatmap data as a dict.
    """
    bin_path = get_rust_binary_path()
    if not bin_path:
        raise RuntimeError(
            "local_mixing binary not found. Run 'cargo build --release' in local_mixing directory."
        )

    # Write circuits to temp files
    c1_path = write_circuit_to_temp_file(circuit_one)
    c2_path = write_circuit_to_temp_file(circuit_two)

    try:
        # Note: The CLI heatmap command may need modification
        # For now, we'll use the Python binding if available
        result = subprocess.run(
            [str(bin_path), "heatmap", "-i", str(num_inputs), "-n", str(num_wires)],
            cwd=str(LOCAL_MIXING_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    finally:
        # Clean up temp files
        os.unlink(c1_path)
        os.unlink(c2_path)


def compute_heatmap_python(
    circuit_one: List[Tuple[int, int, int]],
    circuit_two: List[Tuple[int, int, int]],
    num_wires: int,
    num_inputs: int,
    x_slice: Optional[Tuple[int, int]] = None,
    y_slice: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Compute heatmap in pure Python (for smaller circuits).

    This computes the Hamming distance between circuit states at each position.

    Args:
        circuit_one: First circuit gates
        circuit_two: Second circuit gates
        num_wires: Number of wires
        num_inputs: Number of random inputs to sample
        x_slice: Optional (start, end) for circuit one
        y_slice: Optional (start, end) for circuit two

    Returns:
        Dict with heatmap data
    """
    c1_len = len(circuit_one)
    c2_len = len(circuit_two)

    x1, x2 = x_slice if x_slice else (0, c1_len)
    y1, y2 = y_slice if y_slice else (0, c2_len)

    x_range = range(x1, min(x2 + 1, c1_len + 1))
    y_range = range(y1, min(y2 + 1, c2_len + 1))

    # Initialize heatmap
    heatmap = [[0.0 for _ in y_range] for _ in x_range]

    # Sample random inputs
    max_val = (1 << num_wires) - 1

    for _ in range(num_inputs):
        input_val = random.randint(0, max_val)

        # Compute evolution through circuit one
        evolution_one = [input_val]
        state = input_val
        for active, c1, c2 in circuit_one:
            bit_c1 = (state >> c1) & 1
            bit_c2 = (state >> c2) & 1
            if bit_c1 == 1 and bit_c2 == 0:
                state ^= 1 << active
            evolution_one.append(state)

        # Compute evolution through circuit two
        evolution_two = [input_val]
        state = input_val
        for active, c1, c2 in circuit_two:
            bit_c1 = (state >> c1) & 1
            bit_c2 = (state >> c2) & 1
            if bit_c1 == 1 and bit_c2 == 0:
                state ^= 1 << active
            evolution_two.append(state)

        # Compute overlaps
        for xi, i1 in enumerate(x_range):
            for yi, i2 in enumerate(y_range):
                diff = evolution_one[i1] ^ evolution_two[i2]
                hamming = bin(diff).count("1")
                normalized = hamming / num_wires
                heatmap[xi][yi] += normalized / num_inputs

    # Compute statistics
    all_values = [v for row in heatmap for v in row]
    mean_val = sum(all_values) / len(all_values) if all_values else 0
    variance = (
        sum((v - mean_val) ** 2 for v in all_values) / len(all_values)
        if all_values
        else 0
    )
    std_dev = variance**0.5

    return {
        "heatmap_data": heatmap,
        "x_size": len(x_range),
        "y_size": len(y_range),
        "mean_overlap": mean_val,
        "std_dev": std_dev,
        "circuit_one_size": c1_len,
        "circuit_two_size": c2_len,
    }
