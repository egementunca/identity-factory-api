"""
API endpoints for Skeleton Chain Identity Database.

Provides access to the skeleton identity circuits stored in local_mixing/db LMDB,
indexed by GatePair taxonomy for use with local_mixing pair replacement.

These are "fully noncommuting" identity circuits where ALL adjacent gates collide.
"""

import logging
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import lmdb
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skeleton", tags=["Skeleton Database"])

# Path to local_mixing db - can be overridden via environment
LOCAL_MIXING_DB_PATH = Path(
    os.environ.get(
        "LOCAL_MIXING_DB_PATH",
        Path(__file__).parent.parent.parent.parent / "local_mixing" / "db"
    )
)

# Collision type enum (matches Rust bincode)
COLLISION_TYPES = {0: 'OnActive', 1: 'OnCtrl1', 2: 'OnCtrl2', 3: 'OnNew'}
COLLISION_TO_INT = {v: k for k, v in COLLISION_TYPES.items()}

# Database names for skeleton identities
DB_NAMES = ['ids_n3', 'ids_n4', 'ids_n5', 'ids_n6', 'ids_n7']


# Response models
class SkeletonCircuitResponse(BaseModel):
    """Response model for a single skeleton circuit."""
    id: str  # Format: "skeleton:{width}:{taxonomy}:{index}"
    width: int
    gate_count: int
    gates: List[List[int]]  # [[target, ctrl1, ctrl2], ...]
    gate_string: str  # Ready for handleLoadCircuit: "012;103;..."
    taxonomy: str  # GatePair taxonomy: "(OnCtrl1, OnActive, OnActive)"
    is_identity: bool = True
    is_fully_noncommuting: bool = True


class TaxonomyStats(BaseModel):
    """Stats for a single taxonomy."""
    taxonomy: str  # "(OnCtrl1, OnActive, OnActive)"
    circuit_count: int
    gate_sizes: Dict[int, int]  # {6: 24, 8: 48}


class WidthDetailedStats(BaseModel):
    """Detailed stats for a width including taxonomies."""
    width: int
    circuit_count: int
    taxonomies: List[TaxonomyStats]
    all_fully_noncommuting: bool


class SkeletonExplorerStats(BaseModel):
    """Full explorer statistics."""
    widths: List[WidthDetailedStats]
    total_circuits: int
    total_taxonomies: int


class SkeletonCircuitDetail(BaseModel):
    """Detailed circuit response with collision info."""
    id: str
    width: int
    gate_count: int
    gates: List[List[int]]
    taxonomy: str
    collision_edges: List[List[int]]  # [[0,1], [1,2], ...] for visualization
    is_identity: bool = True  # Skeleton DB only contains identity circuits
    is_fully_noncommuting: bool
    gate_string: str  # For Playground: "012;103;..."


class WidthStats(BaseModel):
    """Stats for a single width (backwards compatible)."""
    width: int
    circuit_count: int
    taxonomy_count: int
    gate_lengths: List[int]


class SkeletonDatabaseStats(BaseModel):
    """Overall skeleton database statistics (backwards compatible)."""
    widths: List[WidthStats]
    total_circuits: int
    total_taxonomies: int


# Helper functions
def decode_taxonomy(key_bytes: bytes) -> Tuple[str, str, str]:
    """Decode GatePair taxonomy from bincode key.

    Key format: 12 bytes (3 × u32 little-endian) matching GatePair::to_bytes()
    """
    if len(key_bytes) >= 12:
        # Unpack as 3 × u32 little-endian
        a, c1, c2 = struct.unpack("<III", key_bytes[:12])
        return (
            COLLISION_TYPES.get(a, f'Unknown({a})'),
            COLLISION_TYPES.get(c1, f'Unknown({c1})'),
            COLLISION_TYPES.get(c2, f'Unknown({c2})')
        )
    elif len(key_bytes) >= 3:
        # Fallback for old 3-byte format (backwards compatibility)
        return (
            COLLISION_TYPES.get(key_bytes[0], f'Unknown({key_bytes[0]})'),
            COLLISION_TYPES.get(key_bytes[1], f'Unknown({key_bytes[1]})'),
            COLLISION_TYPES.get(key_bytes[2], f'Unknown({key_bytes[2]})')
        )
    return ('?', '?', '?')


def encode_taxonomy(taxonomy_str: str) -> bytes:
    """Encode taxonomy string to bincode key.

    Accepts formats: "OnCtrl1,OnActive,OnActive" or "(OnCtrl1, OnActive, OnActive)"

    Key format: 12 bytes (3 × u32 little-endian) matching GatePair::to_bytes()
    """
    # Clean up the string
    clean = taxonomy_str.replace('(', '').replace(')', '').replace(' ', '')
    parts = clean.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid taxonomy format: {taxonomy_str}")

    # Pack as 3 × u32 little-endian (12 bytes total)
    return struct.pack("<III",
        COLLISION_TO_INT.get(parts[0], 3),
        COLLISION_TO_INT.get(parts[1], 3),
        COLLISION_TO_INT.get(parts[2], 3)
    )


def decode_circuits(val_bytes: bytes) -> List[List[Tuple[int, int, int]]]:
    """Decode bincode Vec<Vec<u8>> to list of circuits."""
    if len(val_bytes) < 8:
        return []

    num_circuits = struct.unpack('<Q', val_bytes[:8])[0]
    circuits = []
    offset = 8

    for _ in range(num_circuits):
        if offset + 8 > len(val_bytes):
            break
        blob_len = struct.unpack('<Q', val_bytes[offset:offset+8])[0]
        offset += 8
        if offset + blob_len > len(val_bytes):
            break

        blob = val_bytes[offset:offset+blob_len]
        gates = [(blob[i], blob[i+1], blob[i+2]) for i in range(0, len(blob), 3)]
        circuits.append(gates)
        offset += blob_len

    return circuits


def gates_collide(g1: Tuple[int, int, int], g2: Tuple[int, int, int]) -> bool:
    """Check if two gates collide (don't commute)."""
    t1, c1a, c1b = g1
    t2, c2a, c2b = g2
    # g1's target on g2's controls?
    if t1 == c2a or t1 == c2b:
        return True
    # g2's target on g1's controls?
    if t2 == c1a or t2 == c1b:
        return True
    return False


def all_adjacent_collide(circuit: List[Tuple[int, int, int]]) -> bool:
    """Check if all adjacent gate pairs collide."""
    for i in range(len(circuit) - 1):
        if not gates_collide(circuit[i], circuit[i+1]):
            return False
    return True


def get_collision_edges(circuit: List[Tuple[int, int, int]]) -> List[List[int]]:
    """Get list of edges between colliding adjacent gates."""
    edges = []
    for i in range(len(circuit) - 1):
        if gates_collide(circuit[i], circuit[i+1]):
            edges.append([i, i+1])
    return edges


def wire_to_char(wire: int) -> str:
    """Convert wire index to character for .gate format."""
    if wire < 10:
        return str(wire)
    if wire < 36:
        return chr(ord('a') + wire - 10)
    if wire < 62:
        return chr(ord('A') + wire - 36)
    special = "!@#$%^&*()-_=+[]{}?<>"
    if wire - 62 < len(special):
        return special[wire - 62]
    return '?'


def gates_to_gate_string(gates: List[Tuple[int, int, int]]) -> str:
    """Convert gates list to .gate format string."""
    tokens = []
    for t, c1, c2 in gates:
        tokens.append(f"{wire_to_char(t)}{wire_to_char(c1)}{wire_to_char(c2)}")
    return ";".join(tokens) + ";"


def format_taxonomy(taxonomy: Tuple[str, str, str]) -> str:
    """Format taxonomy tuple as string."""
    return f"({taxonomy[0]}, {taxonomy[1]}, {taxonomy[2]})"


def width_from_db_name(db_name: str) -> int:
    """Extract width from database name (e.g., ids_n5 -> 5)."""
    return int(db_name.replace('ids_n', ''))


# LMDB connection management
_lmdb_env = None


def get_lmdb_env():
    """Get or create LMDB environment."""
    global _lmdb_env
    if _lmdb_env is None:
        if not LOCAL_MIXING_DB_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Local mixing database not found at {LOCAL_MIXING_DB_PATH}"
            )
        _lmdb_env = lmdb.open(
            str(LOCAL_MIXING_DB_PATH),
            max_dbs=60,
            readonly=True,
            map_size=10 * 1024 * 1024 * 1024  # 10GB
        )
    return _lmdb_env


def get_available_dbs() -> List[str]:
    """Get list of available skeleton identity databases."""
    env = get_lmdb_env()
    available = []
    for db_name in DB_NAMES:
        try:
            db = env.open_db(db_name.encode())
            with env.begin(db=db) as txn:
                if txn.stat()['entries'] > 0:
                    available.append(db_name)
        except:
            pass
    return available


# ==================== Explorer Endpoints ====================

@router.get("/explorer/stats", response_model=SkeletonExplorerStats)
async def get_explorer_stats():
    """Get detailed stats for skeleton explorer UI."""
    env = get_lmdb_env()

    widths_list = []
    total_circuits = 0
    total_taxonomies = 0

    for db_name in get_available_dbs():
        width = width_from_db_name(db_name)

        try:
            db = env.open_db(db_name.encode())
        except:
            continue

        width_circuits = 0
        width_all_fnc = True  # fully noncommuting
        taxonomies = []

        with env.begin(db=db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                taxonomy = decode_taxonomy(key)
                circuits = decode_circuits(val)

                tax_stats = {
                    'taxonomy': format_taxonomy(taxonomy),
                    'circuit_count': len(circuits),
                    'gate_sizes': defaultdict(int)
                }

                for circuit in circuits:
                    size = len(circuit)
                    tax_stats['gate_sizes'][size] += 1
                    if not all_adjacent_collide(circuit):
                        width_all_fnc = False

                width_circuits += len(circuits)
                taxonomies.append(TaxonomyStats(
                    taxonomy=tax_stats['taxonomy'],
                    circuit_count=tax_stats['circuit_count'],
                    gate_sizes=dict(tax_stats['gate_sizes'])
                ))

        total_circuits += width_circuits
        total_taxonomies += len(taxonomies)

        widths_list.append(WidthDetailedStats(
            width=width,
            circuit_count=width_circuits,
            taxonomies=taxonomies,
            all_fully_noncommuting=width_all_fnc
        ))

    return SkeletonExplorerStats(
        widths=widths_list,
        total_circuits=total_circuits,
        total_taxonomies=total_taxonomies
    )


@router.get("/explorer/taxonomies/{width}")
async def get_taxonomies_for_width(width: int):
    """Get all taxonomies available for a width."""
    db_name = f"ids_n{width}"
    env = get_lmdb_env()

    try:
        db = env.open_db(db_name.encode())
    except:
        raise HTTPException(status_code=404, detail=f"Database {db_name} not found")

    taxonomies = []
    with env.begin(db=db) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            taxonomy = decode_taxonomy(key)
            circuits = decode_circuits(val)

            gate_sizes = defaultdict(int)
            for circuit in circuits:
                gate_sizes[len(circuit)] += 1

            taxonomies.append({
                'taxonomy': format_taxonomy(taxonomy),
                'taxonomy_key': f"{taxonomy[0]},{taxonomy[1]},{taxonomy[2]}",
                'circuit_count': len(circuits),
                'gate_sizes': dict(gate_sizes)
            })

    return taxonomies


@router.get("/explorer/circuits/{width}")
async def get_circuits_by_width_explorer(
    width: int,
    taxonomy: Optional[str] = Query(None, description="Filter by taxonomy: OnCtrl1,OnActive,OnActive"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Get circuits for a specific width with optional taxonomy filter."""
    db_name = f"ids_n{width}"
    env = get_lmdb_env()

    try:
        db = env.open_db(db_name.encode())
    except:
        raise HTTPException(status_code=404, detail=f"Database {db_name} not found")

    results = []
    skipped = 0
    global_index = 0

    with env.begin(db=db) as txn:
        if taxonomy:
            # Lookup specific taxonomy
            try:
                key = encode_taxonomy(taxonomy)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            val = txn.get(key, db=db)
            if val:
                tax_tuple = decode_taxonomy(key)
                circuits = decode_circuits(val)

                for i, circuit in enumerate(circuits):
                    if skipped < offset:
                        skipped += 1
                        continue
                    if len(results) >= limit:
                        break

                    gates_list = [[g[0], g[1], g[2]] for g in circuit]
                    results.append(SkeletonCircuitResponse(
                        id=f"skeleton:{width}:{taxonomy}:{i}",
                        width=width,
                        gate_count=len(circuit),
                        gates=gates_list,
                        gate_string=gates_to_gate_string(circuit),
                        taxonomy=format_taxonomy(tax_tuple),
                        is_identity=True,
                        is_fully_noncommuting=all_adjacent_collide(circuit)
                    ))
        else:
            # Iterate all taxonomies
            cursor = txn.cursor()
            for key, val in cursor:
                if len(results) >= limit:
                    break

                tax_tuple = decode_taxonomy(key)
                circuits = decode_circuits(val)

                for i, circuit in enumerate(circuits):
                    if skipped < offset:
                        skipped += 1
                        global_index += 1
                        continue
                    if len(results) >= limit:
                        break

                    tax_str = f"{tax_tuple[0]},{tax_tuple[1]},{tax_tuple[2]}"
                    gates_list = [[g[0], g[1], g[2]] for g in circuit]
                    results.append(SkeletonCircuitResponse(
                        id=f"skeleton:{width}:{tax_str}:{i}",
                        width=width,
                        gate_count=len(circuit),
                        gates=gates_list,
                        gate_string=gates_to_gate_string(circuit),
                        taxonomy=format_taxonomy(tax_tuple),
                        is_identity=True,
                        is_fully_noncommuting=all_adjacent_collide(circuit)
                    ))
                    global_index += 1

    return results


@router.get("/explorer/circuit/{width}/{taxonomy}/{index}", response_model=SkeletonCircuitDetail)
async def get_circuit_detail(width: int, taxonomy: str, index: int):
    """Get detailed circuit with collision edges for visualization."""
    db_name = f"ids_n{width}"
    env = get_lmdb_env()

    try:
        db = env.open_db(db_name.encode())
    except:
        raise HTTPException(status_code=404, detail=f"Database {db_name} not found")

    try:
        key = encode_taxonomy(taxonomy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with env.begin(db=db) as txn:
        val = txn.get(key, db=db)
        if not val:
            raise HTTPException(status_code=404, detail=f"Taxonomy {taxonomy} not found")

        circuits = decode_circuits(val)
        if index < 0 or index >= len(circuits):
            raise HTTPException(
                status_code=404,
                detail=f"Circuit index {index} out of range (0-{len(circuits)-1})"
            )

        circuit = circuits[index]
        gates_list = [[g[0], g[1], g[2]] for g in circuit]
        tax_tuple = decode_taxonomy(key)

        return SkeletonCircuitDetail(
            id=f"skeleton:{width}:{taxonomy}:{index}",
            width=width,
            gate_count=len(circuit),
            gates=gates_list,
            taxonomy=format_taxonomy(tax_tuple),
            collision_edges=get_collision_edges(circuit),
            is_fully_noncommuting=all_adjacent_collide(circuit),
            gate_string=gates_to_gate_string(circuit)
        )


# ==================== Original Endpoints (backwards compatible) ====================

@router.get("/stats", response_model=SkeletonDatabaseStats)
async def get_database_stats():
    """Get overall skeleton database statistics."""
    env = get_lmdb_env()

    widths_list = []
    total_circuits = 0
    total_taxonomies = 0

    for db_name in get_available_dbs():
        width = width_from_db_name(db_name)

        try:
            db = env.open_db(db_name.encode())
        except:
            continue

        circuit_count = 0
        taxonomy_count = 0
        gate_lengths = set()

        with env.begin(db=db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                taxonomy_count += 1
                circuits = decode_circuits(val)
                circuit_count += len(circuits)
                for circuit in circuits:
                    gate_lengths.add(len(circuit))

        widths_list.append(WidthStats(
            width=width,
            circuit_count=circuit_count,
            taxonomy_count=taxonomy_count,
            gate_lengths=sorted(gate_lengths)
        ))
        total_circuits += circuit_count
        total_taxonomies += taxonomy_count

    return SkeletonDatabaseStats(
        widths=widths_list,
        total_circuits=total_circuits,
        total_taxonomies=total_taxonomies
    )


@router.get("/circuits", response_model=List[SkeletonCircuitResponse])
async def search_circuits(
    width: Optional[int] = Query(None, ge=3, le=12, description="Filter by width"),
    gate_count: Optional[int] = Query(None, ge=4, le=30, description="Filter by gate count"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Search skeleton circuits with optional filters."""
    env = get_lmdb_env()

    results = []
    skipped = 0

    # Determine widths to search
    if width is not None:
        db_names_to_search = [f"ids_n{width}"]
    else:
        db_names_to_search = get_available_dbs()

    for db_name in db_names_to_search:
        if len(results) >= limit:
            break

        w = width_from_db_name(db_name)

        try:
            db = env.open_db(db_name.encode())
        except:
            continue

        with env.begin(db=db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                if len(results) >= limit:
                    break

                tax_tuple = decode_taxonomy(key)
                circuits = decode_circuits(val)

                for i, circuit in enumerate(circuits):
                    # Apply gate_count filter
                    if gate_count is not None and len(circuit) != gate_count:
                        continue

                    # Apply offset
                    if skipped < offset:
                        skipped += 1
                        continue

                    if len(results) >= limit:
                        break

                    tax_str = f"{tax_tuple[0]},{tax_tuple[1]},{tax_tuple[2]}"
                    gates_list = [[g[0], g[1], g[2]] for g in circuit]

                    results.append(SkeletonCircuitResponse(
                        id=f"skeleton:{w}:{tax_str}:{i}",
                        width=w,
                        gate_count=len(circuit),
                        gates=gates_list,
                        gate_string=gates_to_gate_string(circuit),
                        taxonomy=format_taxonomy(tax_tuple),
                        is_identity=True,
                        is_fully_noncommuting=all_adjacent_collide(circuit)
                    ))

    return results


@router.get("/circuits/{width}", response_model=List[SkeletonCircuitResponse])
async def get_circuits_by_width(
    width: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Get skeleton circuits for a specific width."""
    return await search_circuits(width=width, offset=offset, limit=limit)


@router.get("/circuit/{circuit_id}", response_model=SkeletonCircuitResponse)
async def get_circuit(circuit_id: str):
    """Get a single skeleton circuit by ID.

    ID format: "skeleton:{width}:{taxonomy}:{index}"
    """
    try:
        parts = circuit_id.split(":")
        if len(parts) != 4 or parts[0] != "skeleton":
            raise HTTPException(status_code=400, detail="Invalid circuit ID format. Expected: skeleton:{width}:{taxonomy}:{index}")

        width = int(parts[1])
        taxonomy = parts[2]
        index = int(parts[3])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid circuit ID format")

    detail = await get_circuit_detail(width, taxonomy, index)

    return SkeletonCircuitResponse(
        id=circuit_id,
        width=detail.width,
        gate_count=detail.gate_count,
        gates=detail.gates,
        gate_string=detail.gate_string,
        taxonomy=detail.taxonomy,
        is_identity=True,
        is_fully_noncommuting=detail.is_fully_noncommuting
    )


@router.get("/random/{width}")
async def get_random_circuit(width: int):
    """Get a random skeleton circuit for a given width."""
    import random

    db_name = f"ids_n{width}"
    env = get_lmdb_env()

    try:
        db = env.open_db(db_name.encode())
    except:
        raise HTTPException(status_code=404, detail=f"No circuits found for width {width}")

    # Collect all circuits for this width
    all_circuits = []
    with env.begin(db=db) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            tax_tuple = decode_taxonomy(key)
            circuits = decode_circuits(val)
            for i, circuit in enumerate(circuits):
                all_circuits.append((circuit, tax_tuple, i))

    if not all_circuits:
        raise HTTPException(status_code=404, detail=f"No circuits found for width {width}")

    circuit, tax_tuple, idx = random.choice(all_circuits)
    tax_str = f"{tax_tuple[0]},{tax_tuple[1]},{tax_tuple[2]}"
    gates_list = [[g[0], g[1], g[2]] for g in circuit]

    return SkeletonCircuitResponse(
        id=f"skeleton:{width}:{tax_str}:{idx}",
        width=width,
        gate_count=len(circuit),
        gates=gates_list,
        gate_string=gates_to_gate_string(circuit),
        taxonomy=format_taxonomy(tax_tuple),
        is_identity=True,
        is_fully_noncommuting=all_adjacent_collide(circuit)
    )
