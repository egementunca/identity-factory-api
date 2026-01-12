"""
Advanced API endpoints for obfuscated-circuits.
Includes: Rainbow Table Compression, Pre-computed Identities, Optimal Synthesis.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..database import CircuitDatabase
from .endpoints import get_database
from .models import (
    CompressRequest,
    CompressResponse,
    IdentitiesResponse,
    IdentityCircuitInfo,
    SynthesizeRequest,
    SynthesizeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Paths
GO_PROJ_DIR = Path(__file__).parent.parent.parent / "go-proj"
IDENTITY_ANALYSIS_DIR = (
    Path(__file__).parent.parent.parent / "identity_circuits_analysis"
)

# Cache for pre-computed identities
_identity_cache: Dict[str, List[Dict]] = {}


def _load_identity_file(width: int, gate_count: int) -> List[Dict]:
    """Load pre-computed identity circuits from file."""
    cache_key = f"{width}_{gate_count}"

    if cache_key in _identity_cache:
        return _identity_cache[cache_key]

    filename = IDENTITY_ANALYSIS_DIR / f"identity_circuits_{width}w_{gate_count}l.txt"

    if not filename.exists():
        return []

    circuits = []
    try:
        with open(filename, "r") as f:
            content = f.read()

        # Parse the file format - each circuit block
        current_circuit = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("=== Circuit"):
                if current_circuit:
                    circuits.append(current_circuit)
                current_circuit = {"gates": [], "gate_count": 0}
            elif line.startswith("Gates:") and current_circuit:
                # Parse gates like: Gates: [('X', 0), ('X', 0)]
                try:
                    gates_str = line.split(":", 1)[1].strip()
                    # Simple parsing - extract gate tuples
                    current_circuit["gates_raw"] = gates_str
                except:
                    pass
            elif line.startswith("Gate count:") and current_circuit:
                try:
                    current_circuit["gate_count"] = int(line.split(":")[1].strip())
                except:
                    pass

        if current_circuit:
            circuits.append(current_circuit)

        _identity_cache[cache_key] = circuits

    except Exception as e:
        logger.warning(f"Failed to load identity file {filename}: {e}")

    return circuits


@router.post("/circuits/{circuit_id}/compress", response_model=CompressResponse)
async def compress_circuit(
    circuit_id: int,
    request: CompressRequest = None,
    database: CircuitDatabase = Depends(get_database),
) -> CompressResponse:
    """
    Compress a circuit using Go rainbow table lookup.

    Uses pre-computed circuit databases to find smaller equivalent subcircuits.

    Args:
        circuit_id: Circuit ID to compress
        request: Compression parameters (trials)

    Returns:
        Compression result with potentially smaller circuit
    """
    try:
        # Get the circuit
        circuit_record = database.get_circuit(circuit_id)
        if not circuit_record:
            raise HTTPException(status_code=404, detail="Circuit not found")

        trials = 1000
        if request:
            trials = request.trials

        original_gates = circuit_record.gates
        original_count = len(original_gates)

        # For now, we'll use a Python-based compression approach
        # since calling Go directly requires more setup
        # The Go rainbow table can be exposed via subprocess if go_compress binary exists

        go_compress_binary = GO_PROJ_DIR / "go_compress"

        if go_compress_binary.exists():
            # Use Go compression - need to convert gates to Go format
            try:
                # Convert gates to Go string format: "012;120;201;" (active, ctrl1, ctrl2)
                gate_strings = []
                for gate in original_gates:
                    if isinstance(gate, (list, tuple)) and len(gate) == 3:
                        if all(isinstance(x, int) for x in gate):
                            # ECA57 format: [ctrl1, ctrl2, target] -> Go: "target;ctrl1;ctrl2"
                            # Actually Go uses: active, ctrl1, ctrl2 (same as target, ctrl1, ctrl2)
                            gate_strings.append(f"{gate[2]}{gate[0]}{gate[1]}")

                circuit_str = ";".join(gate_strings) + ";" if gate_strings else ""

                logger.info(f"Calling go_compress with circuit: {circuit_str[:50]}...")

                result = subprocess.run(
                    [
                        str(go_compress_binary),
                        circuit_str,
                        str(circuit_record.width),
                        str(trials),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(GO_PROJ_DIR),  # Run from go-proj so it can find db/
                )

                if result.returncode == 0 and result.stdout.strip():
                    compressed_data = json.loads(result.stdout)
                    if compressed_data.get("success"):
                        return CompressResponse(
                            success=True,
                            circuit_id=circuit_id,
                            original_gate_count=compressed_data.get(
                                "original_gate_count", original_count
                            ),
                            compressed_gate_count=compressed_data.get(
                                "compressed_gate_count", original_count
                            ),
                            compression_ratio=compressed_data.get(
                                "compression_ratio", 1.0
                            ),
                            improved=compressed_data.get("improved", False),
                            compressed_gates=compressed_data.get("gates", []),
                            trials_used=compressed_data.get("trials_used", trials),
                        )
            except Exception as e:
                logger.warning(f"Go compression failed: {e}")

        # Fallback: basic Python compression using swap and cancel
        from circuit.circuit import Circuit

        circuit = Circuit(circuit_record.width)
        for gate in original_gates:
            if isinstance(gate, (list, tuple)):
                if len(gate) >= 2 and isinstance(gate[0], str):
                    gate_type = gate[0]
                    if gate_type == "X":
                        circuit = circuit.x(gate[1])
                    elif gate_type == "CX":
                        circuit = circuit.cx(gate[1], gate[2])
                    elif gate_type == "CCX":
                        circuit = circuit.mcx([gate[1], gate[2]], gate[3])
                elif len(gate) == 3 and all(isinstance(x, int) for x in gate):
                    ctrl1, ctrl2, target = gate
                    circuit = circuit.mcx([ctrl1, ctrl2], target)

        # Try reduction
        reduced_circuit, metrics = circuit.reduce_by_swaps_and_cancellation()
        reduced_gates = reduced_circuit.gates()
        reduced_count = len(reduced_gates)

        compression_ratio = (
            reduced_count / original_count if original_count > 0 else 1.0
        )

        return CompressResponse(
            success=True,
            circuit_id=circuit_id,
            original_gate_count=original_count,
            compressed_gate_count=reduced_count,
            compression_ratio=compression_ratio,
            improved=reduced_count < original_count,
            compressed_gates=reduced_gates,
            trials_used=trials,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return CompressResponse(
            success=False,
            circuit_id=circuit_id,
            original_gate_count=len(circuit_record.gates) if circuit_record else 0,
            compressed_gate_count=0,
            error=str(e),
        )


@router.get("/identities/{width}/{gate_count}", response_model=IdentitiesResponse)
async def get_identity_circuits(
    width: int,
    gate_count: int,
    limit: int = Query(50, ge=1, le=500, description="Maximum circuits to return"),
) -> IdentitiesResponse:
    """
    Get pre-computed identity circuits for given dimensions.

    Returns identity circuits from the pre-computed analysis.
    Available: (2,2), (2,4), (3,2), (3,4), (4,2), (4,4)

    Args:
        width: Number of qubits (2-4)
        gate_count: Number of gates (2 or 4)
        limit: Maximum circuits to return

    Returns:
        List of identity circuits
    """
    if width < 2 or width > 4:
        raise HTTPException(status_code=400, detail="Width must be 2-4")
    if gate_count not in [2, 4]:
        raise HTTPException(
            status_code=400,
            detail="Gate count must be 2 or 4 (odd counts have no identities)",
        )

    circuits = _load_identity_file(width, gate_count)

    # Convert to response format
    circuit_infos = []
    for i, circ in enumerate(circuits[:limit]):
        circuit_infos.append(
            IdentityCircuitInfo(
                gates=circ.get("gates", []),
                gate_count=circ.get("gate_count", gate_count),
                gate_composition=None,  # Could parse from file if needed
            )
        )

    return IdentitiesResponse(
        width=width,
        gate_count=gate_count,
        total_circuits=len(circuits),
        circuits=circuit_infos,
    )


@router.get("/identities/stats")
async def get_identity_stats() -> Dict[str, Any]:
    """
    Get statistics about available pre-computed identity circuits.

    Returns:
        Summary of available identity configurations
    """
    stats = {
        "configurations": [
            {"width": 2, "gate_count": 2, "total": 4},
            {"width": 2, "gate_count": 4, "total": 34},
            {"width": 3, "gate_count": 2, "total": 12},
            {"width": 3, "gate_count": 4, "total": 336},
            {"width": 4, "gate_count": 2, "total": 28},
            {"width": 4, "gate_count": 4, "total": 1900},
        ],
        "total_identities": 2314,
        "note": "Odd gate counts have 0 identities (requires even pairing)",
    }
    return stats


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_circuit(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize optimal ECA57 circuit for a given permutation.

    Uses SAT-based optimal synthesis to find minimum-gate circuit.

    Args:
        request: Synthesis parameters including target permutation

    Returns:
        Synthesized circuit(s) with minimum gates
    """
    try:
        permutation = request.permutation
        max_gates = request.max_gates
        find_all = request.find_all
        max_solutions = request.max_solutions

        # Validate permutation
        n = len(permutation)
        if n & (n - 1) != 0:  # Not power of 2
            raise HTTPException(
                status_code=400, detail="Permutation length must be power of 2"
            )

        # Calculate width from permutation length
        width = n.bit_length() - 1

        if width > 5:
            raise HTTPException(
                status_code=400, detail="Maximum 5 wires (32-element permutation)"
            )

        # Check it's a valid permutation
        if sorted(permutation) != list(range(n)):
            raise HTTPException(
                status_code=400,
                detail="Invalid permutation - must be bijection on [0, n)",
            )

        logger.info(
            f"Synthesizing circuit for {width}-wire permutation with max_gates={max_gates}"
        )

        # Import ECA57 synthesizer
        from eca57.optimal_synthesizer import ECA57OptimalSynthesizer
        from eca57.truth_table import ECA57TruthTable
        from sat.solver import Solver

        # Create truth table from permutation (use constructor directly)
        tt = ECA57TruthTable(width, values=permutation)

        # Create solver
        solver = Solver("minisat-gh")

        # Create optimal synthesizer
        synth = ECA57OptimalSynthesizer(
            output=tt, lower_gc=1, upper_gc=max_gates, solver=solver, anti_trivial=True
        )

        circuits = []

        if find_all:
            # Find all minimum-gate solutions
            found = synth.solve_all(max_solutions=max_solutions)
            for circuit in found:
                circuits.append(circuit.gates_as_tuples())
        else:
            # Find single minimum-gate solution
            circuit = synth.solve()
            if circuit:
                circuits.append(circuit.gates_as_tuples())

        gate_count = circuits[0][0] if circuits and circuits[0] else 0
        # Get actual gate count from first circuit length
        if circuits:
            gate_count = len(circuits[0])

        return SynthesizeResponse(
            success=len(circuits) > 0,
            permutation=permutation,
            gate_count=gate_count,
            circuits=circuits,
            total_solutions=len(circuits),
            attempts=synth.get_attempts(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return SynthesizeResponse(
            success=False, permutation=request.permutation, error=str(e)
        )
