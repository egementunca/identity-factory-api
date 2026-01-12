"""
API endpoints for Local Mixing operations.

Provides endpoints to:
- Load and parse circuits
- Canonicalize circuits
- Preview inflation (kneading) step
- Generate heatmaps
- Get example circuits
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from identity_factory.api.local_mixing_models import (
    CanonicalizeRequest,
    CanonicalizeResponse,
    ExampleCircuit,
    ExamplesResponse,
    HeatmapRequest,
    HeatmapResponse,
    InflatePreviewRequest,
    InflatePreviewResponse,
    InflateSample,
    LoadCircuitRequest,
    LoadCircuitResponse,
    RandomCircuitRequest,
    RandomCircuitResponse,
)
from identity_factory.local_mixing_utils import (
    canonicalize_simple,
    compute_heatmap_python,
    compute_permutation,
    gates_to_string,
    generate_random_circuit,
    generate_random_identity,
    parse_circuit_string,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/load-circuit", response_model=LoadCircuitResponse)
async def load_circuit(request: LoadCircuitRequest):
    """
    Load and parse a circuit string.

    Parses the circuit and computes its permutation.
    """
    try:
        gates = parse_circuit_string(request.circuit_string)

        if not gates:
            raise HTTPException(
                status_code=400,
                detail="No valid gates found in circuit string. Expected format: [0,1,2] [1,2,0] ...",
            )

        # Validate wire indices
        max_wire = max(max(g) for g in gates)
        if max_wire >= request.num_wires:
            raise HTTPException(
                status_code=400,
                detail=f"Gate references wire {max_wire} but num_wires is {request.num_wires}",
            )

        # Compute permutation (only for small circuits to avoid memory issues)
        if request.num_wires <= 20:
            perm = compute_permutation(gates, request.num_wires)
        else:
            # For large circuits, just return a placeholder
            perm = list(range(min(1 << request.num_wires, 1000)))

        return LoadCircuitResponse(
            gates=gates,
            gate_count=len(gates),
            num_wires=request.num_wires,
            permutation=perm[:1000],  # Limit response size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading circuit")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canonicalize", response_model=CanonicalizeResponse)
async def canonicalize(request: CanonicalizeRequest):
    """
    Canonicalize a circuit by reordering non-colliding gates.

    This brings identical gates adjacent so they can be removed (g ⊙ g = identity).
    """
    try:
        original_gates = parse_circuit_string(request.circuit_string)

        if not original_gates:
            raise HTTPException(
                status_code=400, detail="No valid gates found in circuit string."
            )

        # Validate wire indices
        max_wire = max(max(g) for g in original_gates)
        if max_wire >= request.num_wires:
            raise HTTPException(
                status_code=400,
                detail=f"Gate references wire {max_wire} but num_wires is {request.num_wires}",
            )

        canonical_gates, removals = canonicalize_simple(original_gates)

        return CanonicalizeResponse(
            original_gates=original_gates,
            canonical_gates=canonical_gates,
            original_count=len(original_gates),
            canonical_count=len(canonical_gates),
            gates_removed=removals,
            trace=None,  # Could add detailed trace if needed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error canonicalizing circuit")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inflate-preview", response_model=InflatePreviewResponse)
async def inflate_preview(request: InflatePreviewRequest):
    """
    Preview the inflation (kneading) step without full computation.

    The inflation step wraps each gate g as R⁻¹ g R where R is a random identity.
    This previews what that expansion looks like for a sample of gates.
    """
    try:
        gates = parse_circuit_string(request.circuit_string)

        if not gates:
            raise HTTPException(
                status_code=400, detail="No valid gates found in circuit string."
            )

        # Sample gates to preview
        import random

        sample_indices = random.sample(
            range(len(gates)), min(request.sample_count, len(gates))
        )

        samples = []
        total_estimated = 0
        r_min, r_max = request.random_id_size_range

        for idx in sorted(sample_indices):
            gate = gates[idx]
            # Random identity size for this gate
            r_size = random.randint(r_min, r_max)

            # Inflated size = r_size (R⁻¹) + 1 (gate) + r_size (R)
            inflated_size = 2 * r_size + 1

            # Estimate compression (typically 30-60% of inflated)
            compressed_size = int(inflated_size * random.uniform(0.3, 0.6))

            samples.append(
                InflateSample(
                    gate_index=idx,
                    original_gate=gate,
                    inflated_size=inflated_size,
                    r_size=r_size,
                    compressed_size=compressed_size,
                )
            )

            total_estimated += inflated_size

        # Extrapolate for full circuit
        avg_per_gate = total_estimated / len(samples) if samples else 0
        full_estimate = int(avg_per_gate * len(gates))

        # Estimate compression ratio
        avg_compression = (
            sum(s.compressed_size for s in samples)
            / sum(s.inflated_size for s in samples)
            if samples
            else 0.5
        )

        return InflatePreviewResponse(
            original_gate_count=len(gates),
            samples=samples,
            estimated_total_inflated_size=full_estimate,
            estimated_compression_ratio=avg_compression,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in inflate preview")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate a heatmap comparing two circuits.

    The heatmap shows the Hamming distance between circuit states at each position.
    For identity circuits, the diagonal should be dark (low distance).
    """
    try:
        gates_one = parse_circuit_string(request.circuit_one)
        gates_two = parse_circuit_string(request.circuit_two)

        if not gates_one or not gates_two:
            raise HTTPException(
                status_code=400, detail="Both circuits must have valid gates."
            )

        # Validate wire indices
        all_gates = gates_one + gates_two
        max_wire = max(max(g) for g in all_gates)
        if max_wire >= request.num_wires:
            raise HTTPException(
                status_code=400,
                detail=f"Gate references wire {max_wire} but num_wires is {request.num_wires}",
            )

        # Optionally canonicalize
        if request.canonicalize:
            gates_one, _ = canonicalize_simple(gates_one)
            gates_two, _ = canonicalize_simple(gates_two)

        # Limit circuit size for pure Python computation
        max_gates = 500
        if len(gates_one) > max_gates or len(gates_two) > max_gates:
            logger.warning(
                f"Circuits too large ({len(gates_one)}, {len(gates_two)}), using slices"
            )
            gates_one = gates_one[:max_gates]
            gates_two = gates_two[:max_gates]

        # Compute heatmap
        result = compute_heatmap_python(
            gates_one,
            gates_two,
            request.num_wires,
            request.num_inputs,
            request.x_slice,
            request.y_slice,
        )

        return HeatmapResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating heatmap")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/random-circuit", response_model=RandomCircuitResponse)
async def create_random_circuit(request: RandomCircuitRequest):
    """
    Generate a random circuit.
    """
    try:
        gates = generate_random_circuit(request.num_wires, request.num_gates)
        circuit_string = gates_to_string(gates)

        # Compute permutation for small circuits
        if request.num_wires <= 20:
            perm = compute_permutation(gates, request.num_wires)
        else:
            perm = list(range(min(1 << request.num_wires, 1000)))

        return RandomCircuitResponse(
            circuit_string=circuit_string, gates=gates, permutation=perm[:1000]
        )

    except Exception as e:
        logger.exception("Error generating random circuit")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples", response_model=ExamplesResponse)
async def get_examples():
    """
    Get example circuits for testing.
    """
    examples = [
        ExampleCircuit(
            name="Simple 3-wire",
            description="A simple circuit with 5 gates on 3 wires",
            circuit_string="[0,1,2] [1,2,0] [2,0,1] [0,2,1] [1,0,2]",
            num_wires=3,
        ),
        ExampleCircuit(
            name="Identity R⊙R⁻¹ (5 gates)",
            description="Random circuit followed by its inverse - should canonicalize to empty",
            circuit_string="[1,0,2] [0,2,1] [2,1,0] [2,1,0] [0,2,1] [1,0,2]",
            num_wires=3,
        ),
        ExampleCircuit(
            name="5-wire trace example",
            description="The example from COMPLETE_TRACE.md showing canonicalization",
            circuit_string="[1,0,3] [1,4,2] [0,4,1] [4,2,0] [4,3,0] [4,3,0] [4,2,0] [0,4,1] [1,4,2] [1,0,3]",
            num_wires=5,
        ),
        ExampleCircuit(
            name="Longer random (5 wires, 20 gates)",
            description="A larger random circuit for testing",
            circuit_string=gates_to_string(generate_random_circuit(5, 20)),
            num_wires=5,
        ),
    ]

    return ExamplesResponse(examples=examples)


@router.post("/random-identity", response_model=RandomCircuitResponse)
async def create_random_identity(request: RandomCircuitRequest):
    """
    Generate a random identity circuit (R followed by R⁻¹).

    The circuit should canonicalize to empty.
    """
    try:
        half_size = request.num_gates // 2
        gates = generate_random_identity(request.num_wires, half_size)
        circuit_string = gates_to_string(gates)

        # Compute permutation (should be identity!)
        if request.num_wires <= 20:
            perm = compute_permutation(gates, request.num_wires)
        else:
            perm = list(range(min(1 << request.num_wires, 1000)))

        return RandomCircuitResponse(
            circuit_string=circuit_string, gates=gates, permutation=perm[:1000]
        )

    except Exception as e:
        logger.exception("Error generating random identity")
        raise HTTPException(status_code=500, detail=str(e))
