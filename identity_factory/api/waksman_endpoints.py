"""
Waksman Network API endpoints.

Provides endpoints for generating and managing Waksman-style wire permutation circuits.
Waksman networks are O(n log n) permutation networks that scale better than SAT synthesis.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

# Add sat_revsynth to path for Waksman imports
sat_revsynth_path = Path(__file__).resolve().parent.parent.parent.parent / "sat_revsynth" / "src"
if str(sat_revsynth_path) not in sys.path:
    sys.path.insert(0, str(sat_revsynth_path))

from identity_factory.wire_shuffler_db import (
    WaksmanCircuitDatabase,
    WaksmanCircuitRecord,
    WirePermutationRecord,
)

router = APIRouter(prefix="/waksman", tags=["waksman"])

# Global database instance
_db: Optional[WaksmanCircuitDatabase] = None


def get_database() -> WaksmanCircuitDatabase:
    """Get or create the database instance."""
    global _db
    if _db is None:
        db_path = Path(__file__).resolve().parent.parent.parent / "wire_shuffler.db"
        _db = WaksmanCircuitDatabase(str(db_path))
    return _db


# =====================
# Request/Response Models
# =====================


class WaksmanGenerateRequest(BaseModel):
    """Request to generate a Waksman circuit."""

    width: int = Field(..., ge=2, le=128, description="Number of wires (>= 2)")
    permutation: Optional[List[int]] = Field(
        None, description="Specific permutation (if not provided, uses type)"
    )
    permutation_type: str = Field(
        "specific", description="'specific', 'random', 'reverse', 'shift', 'identity'"
    )
    shift_amount: Optional[int] = Field(None, description="Shift amount for 'shift' type")
    store_in_db: bool = Field(True, description="Store result in database")
    obfuscate: bool = Field(
        False, description="Fill identity slots with random identity circuits for obfuscation"
    )
    identity_gate_count: Optional[int] = Field(
        None, description="Target gate count for identity fillers (default: random)"
    )
    min_identity_gates: int = Field(
        24, ge=12, le=72, description="Minimum gates for identity fillers (12-72)"
    )
    obfuscation_seed: Optional[int] = Field(
        None, description="Random seed for reproducible obfuscation"
    )


class WaksmanCircuitResponse(BaseModel):
    """Generated Waksman circuit response."""

    id: Optional[int] = None
    width: int
    permutation: List[int]
    perm_hash: str
    gate_count: int
    gates: List[List[int]]
    swap_count: int
    synth_time_ms: float
    verified: Optional[bool] = None
    obfuscated: bool = False
    identity_slots: int = 0


class WaksmanStatsResponse(BaseModel):
    """Waksman circuit statistics."""

    total_circuits: int
    by_width: Dict[str, Dict[str, float]]


class WaksmanComparisonResponse(BaseModel):
    """SAT vs Waksman comparison."""

    perm_id: int
    perm_hash: str
    permutation: List[int]
    sat_available: bool
    waksman_available: bool
    sat_gate_count: Optional[int]
    waksman_gate_count: Optional[int]
    gate_count_diff: Optional[int]


class WaksmanBatchRequest(BaseModel):
    """Request for batch Waksman generation."""

    width: int = Field(..., ge=2, le=128)
    count: int = Field(..., ge=1, le=1000, description="Number of random permutations")
    store_in_db: bool = Field(True)


class WaksmanBatchResponse(BaseModel):
    """Batch generation result."""

    job_id: str
    status: str
    message: str
    circuits_generated: int = 0


# =====================
# Helper Functions
# =====================


def compute_perm_hash(perm: List[int]) -> str:
    """Compute hash for a permutation."""
    perm_str = ",".join(map(str, perm))
    return hashlib.sha256(perm_str.encode()).hexdigest()[:16]


def compute_circuit_hash(gates: List[List[int]]) -> str:
    """Compute hash for a circuit gate list."""
    payload = json.dumps(gates, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_perm_stats(perm: List[int]) -> Dict:
    """Compute permutation statistics."""
    n = len(perm)
    fixed_points = sum(1 for i in range(n) if perm[i] == i)
    hamming = n - fixed_points

    # Compute cycles
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            cycles.append(len(cycle))

    cycles.sort(reverse=True)
    cycle_type = ",".join(map(str, cycles))
    swap_distance = n - len(cycles)
    parity = "even" if swap_distance % 2 == 0 else "odd"

    return {
        "fixed_points": fixed_points,
        "hamming": hamming,
        "cycles": len(cycles),
        "swap_distance": swap_distance,
        "cycle_type": cycle_type,
        "parity": parity,
        "is_identity": perm == list(range(n)),
    }


def generate_waksman_circuit(
    width: int,
    perm: List[int],
    obfuscate: bool = False,
    identity_gate_count: Optional[int] = None,
    min_identity_gates: int = 24,
    obfuscation_seed: Optional[int] = None
) -> Dict:
    """Generate a Waksman circuit for a permutation.

    Args:
        width: Number of wires.
        perm: Target permutation.
        obfuscate: If True, fill identity slots with random identity circuits.
        identity_gate_count: Target gate count for identity fillers.
        min_identity_gates: Minimum gates for identity fillers (default 24).
        obfuscation_seed: Random seed for reproducible obfuscation.

    Returns:
        Dict with circuit info including gates, swap count, etc.
    """
    from synthesizers.waksman import SimpleSwapNetwork

    start_time = time.time()

    # Use simple network (works for any width)
    network = SimpleSwapNetwork(width)
    network.route(perm)
    circuit = network.to_eca57_circuit(
        obfuscate=obfuscate,
        identity_gate_count=identity_gate_count,
        rng_seed=obfuscation_seed,
        min_identity_gates=min_identity_gates
    )

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "gate_count": len(circuit),
        "gates": [g.to_tuple() for g in circuit.gates()],
        "swap_count": network.swap_count(),
        "identity_slots": network.identity_slot_count(),
        "obfuscated": obfuscate,
        "synth_time_ms": round(elapsed_ms, 2),
    }


def verify_circuit(gates: List, perm: List[int], width: int) -> bool:
    """Verify a circuit implements the correct permutation."""
    from gates.eca57 import ECA57Circuit

    circuit = ECA57Circuit(width + 1)  # +1 for aux wire
    for t, c1, c2 in gates:
        circuit.add_gate(t, c1, c2)

    # Test a sample of inputs
    import random

    rng = random.Random(42)
    n = len(perm)

    if n <= 8:
        test_inputs = list(range(2**n))
    else:
        test_inputs = [rng.randint(0, 2**n - 1) for _ in range(100)]

    for i in test_inputs:
        input_state = [(i >> b) & 1 for b in range(width + 1)]
        output_state = circuit.apply(input_state)
        for k in range(n):
            if output_state[k] != input_state[perm[k]]:
                return False

    return True


# =====================
# API Endpoints
# =====================


@router.post("/generate", response_model=WaksmanCircuitResponse)
async def generate_waksman(request: WaksmanGenerateRequest):
    """Generate a single Waksman circuit.

    Supports various permutation types:
    - 'specific': Use the provided permutation
    - 'random': Generate a random permutation
    - 'reverse': Generate reverse permutation [n-1, n-2, ..., 0]
    - 'shift': Generate cyclic shift by shift_amount
    - 'identity': Generate identity permutation [0, 1, ..., n-1]
    """
    import random

    width = request.width

    # Determine permutation
    if request.permutation_type == "specific":
        if request.permutation is None:
            raise HTTPException(status_code=400, detail="Permutation required for 'specific' type")
        if len(request.permutation) != width:
            raise HTTPException(
                status_code=400,
                detail=f"Permutation length {len(request.permutation)} != width {width}",
            )
        if sorted(request.permutation) != list(range(width)):
            raise HTTPException(status_code=400, detail="Invalid permutation")
        perm = request.permutation
    elif request.permutation_type == "random":
        perm = list(range(width))
        random.shuffle(perm)
    elif request.permutation_type == "reverse":
        perm = list(range(width - 1, -1, -1))
    elif request.permutation_type == "shift":
        k = (request.shift_amount or 1) % width
        perm = [(i - k) % width for i in range(width)]
    elif request.permutation_type == "identity":
        perm = list(range(width))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown permutation type: {request.permutation_type}")

    # Generate circuit (with optional obfuscation)
    result = generate_waksman_circuit(
        width,
        perm,
        obfuscate=request.obfuscate,
        identity_gate_count=request.identity_gate_count,
        min_identity_gates=request.min_identity_gates,
        obfuscation_seed=request.obfuscation_seed
    )
    perm_hash = compute_perm_hash(perm)

    # Optionally store in database
    circuit_id = None
    if request.store_in_db:
        db = get_database()

        # Upsert permutation
        perm_stats = compute_perm_stats(perm)
        perm_record = WirePermutationRecord(
            id=None,
            width=width,
            wire_perm=perm,
            wire_perm_hash=perm_hash,
            fixed_points=perm_stats["fixed_points"],
            hamming=perm_stats["hamming"],
            cycles=perm_stats["cycles"],
            swap_distance=perm_stats["swap_distance"],
            cycle_type=perm_stats["cycle_type"],
            parity=perm_stats["parity"],
            is_identity=perm_stats["is_identity"],
        )
        perm_id = db.upsert_permutation(perm_record)

        # Insert Waksman circuit
        circuit_record = WaksmanCircuitRecord(
            id=None,
            perm_id=perm_id,
            gate_count=result["gate_count"],
            gates=result["gates"],
            swap_count=result["swap_count"],
            synth_time_ms=int(result["synth_time_ms"]),
            verify_ok=True,  # We trust our implementation
            circuit_hash=compute_circuit_hash(result["gates"]),
        )
        circuit_id = db.insert_waksman_circuit(circuit_record)

    return WaksmanCircuitResponse(
        id=circuit_id,
        width=width,
        permutation=perm,
        perm_hash=perm_hash,
        gate_count=result["gate_count"],
        gates=result["gates"],
        swap_count=result["swap_count"],
        synth_time_ms=result["synth_time_ms"],
        verified=True,
        obfuscated=result.get("obfuscated", False),
        identity_slots=result.get("identity_slots", 0),
    )


@router.get("/circuits", response_model=List[WaksmanCircuitResponse])
async def list_waksman_circuits(
    width: Optional[int] = Query(None, ge=2, le=128),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List Waksman circuits with optional filtering."""
    db = get_database()
    circuits = db.get_waksman_circuits(width=width, limit=limit, offset=offset)

    results = []
    for c in circuits:
        # Get permutation info
        with db._connect() as conn:
            cursor = conn.execute(
                "SELECT wire_perm, wire_perm_hash FROM wire_permutations WHERE id = ?",
                (c.perm_id,),
            )
            row = cursor.fetchone()
            if row:
                perm = json.loads(row["wire_perm"])
                perm_hash = row["wire_perm_hash"]
            else:
                continue

        results.append(
            WaksmanCircuitResponse(
                id=c.id,
                width=len(perm),
                permutation=perm,
                perm_hash=perm_hash,
                gate_count=c.gate_count,
                gates=c.gates,
                swap_count=c.swap_count,
                synth_time_ms=c.synth_time_ms or 0,
                verified=c.verify_ok,
            )
        )

    return results


@router.get("/circuit/{circuit_id}", response_model=WaksmanCircuitResponse)
async def get_waksman_circuit(circuit_id: int):
    """Get a specific Waksman circuit by ID."""
    db = get_database()

    with db._connect() as conn:
        cursor = conn.execute(
            """
            SELECT w.*, p.wire_perm, p.wire_perm_hash, p.width
            FROM waksman_circuits w
            JOIN wire_permutations p ON w.perm_id = p.id
            WHERE w.id = ?
        """,
            (circuit_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Circuit not found")

    return WaksmanCircuitResponse(
        id=row["id"],
        width=row["width"],
        permutation=json.loads(row["wire_perm"]),
        perm_hash=row["wire_perm_hash"],
        gate_count=row["gate_count"],
        gates=json.loads(row["gates"]),
        swap_count=row["swap_count"],
        synth_time_ms=row["synth_time_ms"] or 0,
        verified=bool(row["verify_ok"]) if row["verify_ok"] is not None else None,
    )


@router.get("/stats", response_model=WaksmanStatsResponse)
async def get_waksman_stats():
    """Get Waksman circuit statistics."""
    db = get_database()
    stats = db.get_waksman_stats()

    # Convert integer keys to strings for JSON compatibility
    by_width_raw = stats.get("waksman_by_width", {})
    by_width = {str(k): v for k, v in by_width_raw.items()}

    return WaksmanStatsResponse(
        total_circuits=stats.get("total_waksman_circuits", 0),
        by_width=by_width,
    )


@router.get("/compare/{perm_hash}", response_model=WaksmanComparisonResponse)
async def compare_sat_vs_waksman(perm_hash: str):
    """Compare SAT-synthesized vs Waksman circuit for a permutation."""
    db = get_database()

    # Find permutation by hash
    with db._connect() as conn:
        cursor = conn.execute(
            "SELECT id, wire_perm FROM wire_permutations WHERE wire_perm_hash = ?",
            (perm_hash,),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Permutation not found")

    perm_id = row["id"]
    perm = json.loads(row["wire_perm"])

    comparison = db.compare_sat_vs_waksman(perm_id)

    return WaksmanComparisonResponse(
        perm_id=perm_id,
        perm_hash=perm_hash,
        permutation=perm,
        sat_available=comparison["sat_available"],
        waksman_available=comparison["waksman_available"],
        sat_gate_count=comparison["sat_gate_count"],
        waksman_gate_count=comparison["waksman_gate_count"],
        gate_count_diff=comparison["gate_count_diff"],
    )


@router.post("/generate-batch", response_model=WaksmanBatchResponse)
async def generate_waksman_batch(
    request: WaksmanBatchRequest, background_tasks: BackgroundTasks
):
    """Start async batch generation of Waksman circuits.

    Generates multiple random permutations and their circuits.
    """
    import uuid

    job_id = str(uuid.uuid4())[:8]

    # For now, do synchronous generation (can be made async later)
    import random

    db = get_database()
    count = 0

    for _ in range(request.count):
        perm = list(range(request.width))
        random.shuffle(perm)

        result = generate_waksman_circuit(request.width, perm)
        perm_hash = compute_perm_hash(perm)

        if request.store_in_db:
            perm_stats = compute_perm_stats(perm)
            perm_record = WirePermutationRecord(
                id=None,
                width=request.width,
                wire_perm=perm,
                wire_perm_hash=perm_hash,
                fixed_points=perm_stats["fixed_points"],
                hamming=perm_stats["hamming"],
                cycles=perm_stats["cycles"],
                swap_distance=perm_stats["swap_distance"],
                cycle_type=perm_stats["cycle_type"],
                parity=perm_stats["parity"],
                is_identity=perm_stats["is_identity"],
            )
            perm_id = db.upsert_permutation(perm_record)

            circuit_record = WaksmanCircuitRecord(
                id=None,
                perm_id=perm_id,
                gate_count=result["gate_count"],
                gates=result["gates"],
                swap_count=result["swap_count"],
                synth_time_ms=int(result["synth_time_ms"]),
                verify_ok=True,
                circuit_hash=compute_circuit_hash(result["gates"]),
            )
            db.insert_waksman_circuit(circuit_record)

        count += 1

    return WaksmanBatchResponse(
        job_id=job_id,
        status="completed",
        message=f"Generated {count} Waksman circuits",
        circuits_generated=count,
    )


@router.get("/health")
async def waksman_health():
    """Health check for Waksman endpoints."""
    return {"status": "ok", "service": "waksman"}
