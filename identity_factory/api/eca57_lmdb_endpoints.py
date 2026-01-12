"""
API endpoints for ECA57 LMDB Identity Database.

Provides access to the fresh enumeration results stored in sat_revsynth LMDB.
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Get sat_revsynth path from environment or default to sibling of identity-factory-api
# Path: identity-factory-api/identity_factory/api/eca57_lmdb_endpoints.py
# Need to go up 4 levels to reach research-group, then into sat_revsynth
SAT_REVSYNTH_PATH = Path(
    os.environ.get(
        "SAT_REVSYNTH_PATH", Path(__file__).parent.parent.parent.parent / "sat_revsynth"
    )
)
sys.path.insert(0, str(SAT_REVSYNTH_PATH / "src"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eca57-lmdb", tags=["ECA57 LMDB"])

# LMDB path - can be overridden via environment
LMDB_PATH = Path(
    os.environ.get(
        "ECA57_LMDB_PATH", SAT_REVSYNTH_PATH / "data" / "eca57_identities_lmdb"
    )
)


class CircuitResponse(BaseModel):
    """Response model for a single circuit."""

    id: int
    width: int
    gate_count: int
    gates: List[List[int]]  # [[target, ctrl1, ctrl2], ...]
    equivalence_class_size: int
    # Computed properties
    skeleton_edges: Optional[List[List[int]]] = None
    complexity_walk: Optional[List[int]] = None
    permutation: Optional[List[int]] = None  # Full permutation mapping
    cycle_notation: Optional[str] = None  # Cycle notation string


class ConfigStats(BaseModel):
    """Stats for a single configuration (width, gate_count)."""

    width: int
    gate_count: int
    num_representatives: int
    total_circuits: int


class DatabaseStats(BaseModel):
    """Overall database statistics."""

    configurations: List[ConfigStats]
    total_representatives: int
    total_circuits: int


def get_lmdb_env():
    """Get LMDB environment, creating if needed."""
    try:
        import lmdb

        if not LMDB_PATH.exists():
            raise HTTPException(
                status_code=404, detail=f"LMDB database not found at {LMDB_PATH}"
            )
        # Use lock=False to allow reading while another process might be writing
        return lmdb.open(str(LMDB_PATH), max_dbs=10, readonly=True, lock=False)
    except ImportError:
        raise HTTPException(status_code=500, detail="lmdb not installed")
    except Exception as e:
        logger.error(f"Failed to open LMDB: {e}")
        raise HTTPException(status_code=500, detail=f"LMDB error: {str(e)}")


def gates_collide(g1: List[int], g2: List[int]) -> bool:
    """
    Check if two gates collide (don't commute).
    Gates collide iff one's target is in the other's controls.
    Note: Same targets DO commute for ECA57 gates!
    """
    t1, c1_1, c2_1 = g1
    t2, c1_2, c2_2 = g2

    # Target in other's controls = collision
    if t1 in [c1_2, c2_2] or t2 in [c1_1, c2_1]:
        return True
    return False


def compute_skeleton_edges(gates: List[List[int]]) -> List[List[int]]:
    """
    Compute skeleton graph edges - ALL collision pairs.

    Two gates collide (can't be swapped) if:
    - They share the same target
    - One's target is in the other's controls

    This matches the existing SkeletonGraph.tsx implementation.
    """
    edges = []
    n = len(gates)

    for i in range(n):
        for j in range(i + 1, n):
            if gates_collide(gates[i], gates[j]):
                edges.append([i, j])

    return edges


def compute_complexity_walk(width: int, gates: List[List[int]]) -> List[int]:
    """
    Compute Hamming distance from identity after each gate.
    """
    N = 1 << width  # 2^width
    mapping = list(range(N))
    walk = []

    for t, c1, c2 in gates:
        # Apply ECA57 gate: target ^= (ctrl1 OR NOT ctrl2)
        new_mapping = mapping.copy()
        for state in range(N):
            ctrl1_set = bool(mapping[state] & (1 << c1))
            ctrl2_set = bool(mapping[state] & (1 << c2))
            condition = ctrl1_set or (not ctrl2_set)
            if condition:
                new_mapping[state] = mapping[state] ^ (1 << t)
            else:
                new_mapping[state] = mapping[state]
        mapping = new_mapping

        # Compute Hamming distance from identity
        hamming = sum(bin(i ^ mapping[i]).count("1") for i in range(N))
        walk.append(hamming)

    return walk


def compute_permutation(width: int, gates: List[List[int]]) -> List[int]:
    """
    Compute the final permutation mapping from applying all gates.
    Returns list where result[i] = j means input state i maps to output state j.
    """
    N = 1 << width
    mapping = list(range(N))

    for t, c1, c2 in gates:
        new_mapping = mapping.copy()
        for state in range(N):
            out = mapping[state]
            ctrl1_set = bool(out & (1 << c1))
            ctrl2_set = bool(out & (1 << c2))
            condition = ctrl1_set or (not ctrl2_set)
            if condition:
                new_mapping[state] = out ^ (1 << t)
            else:
                new_mapping[state] = out
        mapping = new_mapping

    return mapping


def compute_cycle_notation(permutation: List[int]) -> str:
    """
    Convert permutation to cycle notation string.
    For identity, returns "()".
    """
    n = len(permutation)
    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue

        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = permutation[i]

        # Skip 1-cycles (fixed points)
        if len(cycle) > 1:
            cycles.append(cycle)

    if not cycles:
        return "()"  # Identity

    # Format cycles
    result = ""
    for cycle in cycles:
        result += "(" + " ".join(str(x) for x in cycle) + ")"

    return result


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get overall database statistics."""
    env = get_lmdb_env()
    try:
        meta_db = env.open_db(b"metadata")
        circuits_db = env.open_db(b"circuits")

        configs = []
        total_reps = 0
        total_circuits = 0

        with env.begin() as txn:
            cursor = txn.cursor(db=meta_db)
            for key, value in cursor:
                config = key.decode()
                count = int(value.decode())

                # Parse width and gate_count from key like "w3g4"
                w = int(config[1])
                g = int(config[3:])

                # Sum up equivalence class sizes
                config_total = 0
                for i in range(count):
                    circuit_key = f"w{w}g{g}:{i:08d}".encode()
                    data = txn.get(circuit_key, db=circuits_db)
                    if data:
                        record = json.loads(data.decode())
                        config_total += record.get("equivalence_class_size", 1)

                configs.append(
                    ConfigStats(
                        width=w,
                        gate_count=g,
                        num_representatives=count,
                        total_circuits=config_total,
                    )
                )
                total_reps += count
                total_circuits += config_total

        return DatabaseStats(
            configurations=sorted(configs, key=lambda x: (x.width, x.gate_count)),
            total_representatives=total_reps,
            total_circuits=total_circuits,
        )
    finally:
        env.close()


@router.get("/configurations")
async def list_configurations():
    """List all available configurations."""
    env = get_lmdb_env()
    try:
        meta_db = env.open_db(b"metadata")

        configs = []
        with env.begin() as txn:
            cursor = txn.cursor(db=meta_db)
            for key, value in cursor:
                config = key.decode()
                count = int(value.decode())
                w = int(config[1])
                g = int(config[3:])
                configs.append(
                    {"width": w, "gate_count": g, "num_representatives": count}
                )

        return sorted(configs, key=lambda x: (x["width"], x["gate_count"]))
    finally:
        env.close()


@router.get("/circuits/{width}/{gate_count}", response_model=List[CircuitResponse])
async def get_circuits(
    width: int,
    gate_count: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    include_skeleton: bool = Query(True),
    include_complexity: bool = Query(True),
):
    """Get circuits for a specific configuration with pagination."""
    env = get_lmdb_env()
    try:
        circuits_db = env.open_db(b"circuits")
        meta_db = env.open_db(b"metadata")

        with env.begin() as txn:
            # Check if config exists
            config_key = f"w{width}g{gate_count}".encode()
            count_data = txn.get(config_key, db=meta_db)
            if not count_data:
                raise HTTPException(
                    status_code=404, detail=f"No data for w{width}g{gate_count}"
                )

            total = int(count_data.decode())

            circuits = []
            for i in range(offset, min(offset + limit, total)):
                key = f"w{width}g{gate_count}:{i:08d}".encode()
                data = txn.get(key, db=circuits_db)
                if data:
                    record = json.loads(data.decode())
                    gates = record["gates"]

                    response = CircuitResponse(
                        id=i,
                        width=record["width"],
                        gate_count=record["gate_count"],
                        gates=gates,
                        equivalence_class_size=record["equivalence_class_size"],
                    )

                    if include_skeleton:
                        response.skeleton_edges = compute_skeleton_edges(gates)

                    if include_complexity:
                        response.complexity_walk = compute_complexity_walk(width, gates)

                    circuits.append(response)

        return circuits
    finally:
        env.close()


@router.get(
    "/circuit/{width}/{gate_count}/{circuit_id}", response_model=CircuitResponse
)
async def get_circuit(width: int, gate_count: int, circuit_id: int):
    """Get a single circuit by ID."""
    env = get_lmdb_env()
    try:
        circuits_db = env.open_db(b"circuits")

        with env.begin() as txn:
            key = f"w{width}g{gate_count}:{circuit_id:08d}".encode()
            data = txn.get(key, db=circuits_db)
            if not data:
                raise HTTPException(status_code=404, detail="Circuit not found")

            record = json.loads(data.decode())
            gates = record["gates"]
            permutation = compute_permutation(width, gates)

            return CircuitResponse(
                id=circuit_id,
                width=record["width"],
                gate_count=record["gate_count"],
                gates=gates,
                equivalence_class_size=record["equivalence_class_size"],
                skeleton_edges=compute_skeleton_edges(gates),
                complexity_walk=compute_complexity_walk(width, gates),
                permutation=permutation,
                cycle_notation=compute_cycle_notation(permutation),
            )
    finally:
        env.close()


class EquivalentCircuit(BaseModel):
    """Response model for an equivalent circuit form."""

    index: int
    gates: List[List[int]]
    skeleton_edges: List[List[int]]


@router.get("/circuit/{width}/{gate_count}/{circuit_id}/equivalents")
async def get_equivalent_forms(
    width: int, gate_count: int, circuit_id: int, limit: int = Query(20, ge=1, le=100)
):
    """
    Get equivalent forms of a circuit via unroll.
    Returns up to `limit` equivalent circuits.
    """
    env = get_lmdb_env()
    try:
        circuits_db = env.open_db(b"circuits")

        with env.begin() as txn:
            key = f"w{width}g{gate_count}:{circuit_id:08d}".encode()
            data = txn.get(key, db=circuits_db)
            if not data:
                raise HTTPException(status_code=404, detail="Circuit not found")

            record = json.loads(data.decode())
    finally:
        env.close()

    # Import ECA57Circuit and rebuild
    try:
        from gates.eca57 import ECA57Circuit, ECA57Gate

        circuit = ECA57Circuit(width)
        for t, c1, c2 in record["gates"]:
            circuit.add_gate(t, c1, c2)

        # Compute equivalent forms via unroll
        equivalents = circuit.unroll()

        # Return limited set with skeleton edges
        results = []
        for i, eq in enumerate(equivalents[:limit]):
            gates = [(g.target, g.ctrl1, g.ctrl2) for g in eq.gates()]
            results.append(
                {
                    "index": i,
                    "gates": gates,
                    "skeleton_edges": compute_skeleton_edges(gates),
                }
            )

        return {
            "representative_id": circuit_id,
            "total_equivalents": len(equivalents),
            "returned": len(results),
            "equivalents": results,
        }
    except Exception as e:
        logger.error(f"Failed to compute equivalents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
