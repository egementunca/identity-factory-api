"""
Unified search endpoint for querying multiple circuit databases.

Provides a single search interface across:
- SQLite circuits database
- ECA57 LMDB enumeration database
- Skeleton chain identity database
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Get sat_revsynth path
SAT_REVSYNTH_PATH = Path(
    os.environ.get(
        "SAT_REVSYNTH_PATH", Path(__file__).parent.parent.parent.parent / "sat_revsynth"
    )
)
sys.path.insert(0, str(SAT_REVSYNTH_PATH / "src"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Unified Search"])


# Response models
DatabaseSource = Literal["sqlite", "eca57-lmdb", "skeleton"]


class UnifiedCircuitResult(BaseModel):
    """Unified circuit result format."""
    id: str  # Unique: "{source}:{width}:{index}"
    source: DatabaseSource
    width: int
    gateCount: int
    gates: List[List[int]]  # [[target, ctrl1, ctrl2], ...]
    gateString: str  # Ready for handleLoadCircuit
    isIdentity: bool
    isRepresentative: Optional[bool] = None
    equivalenceClassSize: Optional[int] = None
    taxonomy: Optional[str] = None


class UnifiedSearchResponse(BaseModel):
    """Response for unified search."""
    results: List[UnifiedCircuitResult]
    total: int
    sourcesQueried: List[str]
    sourcesStats: Dict[str, int]


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


def gates_to_gate_string(gates: List[List[int]]) -> str:
    """Convert gates list to .gate format string."""
    tokens = []
    for gate in gates:
        target, ctrl1, ctrl2 = gate[0], gate[1], gate[2]
        tokens.append(f"{wire_to_char(target)}{wire_to_char(ctrl1)}{wire_to_char(ctrl2)}")
    return ";".join(tokens) + ";"


async def search_skeleton_db(
    width: Optional[int],
    gate_count: Optional[int],
    limit: int
) -> List[UnifiedCircuitResult]:
    """Search skeleton database."""
    results = []

    try:
        from database.skeleton_db import (
            SkeletonDBBuilder,
            SkeletonDBConfig,
            enumerate_collision_taxonomies
        )

        skeleton_db_path = Path(
            os.environ.get(
                "SKELETON_DB_PATH",
                SAT_REVSYNTH_PATH / "skeleton_identity_db"
            )
        )

        if not skeleton_db_path.exists():
            logger.warning(f"Skeleton DB not found at {skeleton_db_path}")
            return []

        config = SkeletonDBConfig(map_size=20 * 1024 * 1024 * 1024)
        with SkeletonDBBuilder(skeleton_db_path, config) as builder:
            stats = builder.get_stats()

            # Determine widths to search
            if width is not None:
                widths_to_search = [width] if width in stats.get("circuits_per_width", {}) else []
            else:
                widths_to_search = sorted(stats.get("circuits_per_width", {}).keys())

            circuit_index = 0
            for w in widths_to_search:
                if len(results) >= limit:
                    break

                for taxonomy in enumerate_collision_taxonomies():
                    if len(results) >= limit:
                        break

                    circuits = builder.lookup(w, taxonomy)
                    for blob in circuits:
                        if len(results) >= limit:
                            break

                        blob_gate_count = len(blob) // 3

                        # Apply gate_count filter
                        if gate_count is not None and blob_gate_count != gate_count:
                            continue

                        gates = []
                        for i in range(0, len(blob), 3):
                            gates.append([blob[i], blob[i + 1], blob[i + 2]])

                        results.append(UnifiedCircuitResult(
                            id=f"skeleton:{w}:{circuit_index}",
                            source="skeleton",
                            width=w,
                            gateCount=blob_gate_count,
                            gates=gates,
                            gateString=gates_to_gate_string(gates),
                            isIdentity=True,
                            taxonomy=f"({taxonomy.a.name},{taxonomy.c1.name},{taxonomy.c2.name})"
                        ))
                        circuit_index += 1

    except Exception as e:
        logger.error(f"Failed to search skeleton DB: {e}")

    return results


async def search_eca57_lmdb(
    width: Optional[int],
    gate_count: Optional[int],
    limit: int
) -> List[UnifiedCircuitResult]:
    """Search ECA57 LMDB database."""
    results = []

    try:
        import lmdb

        lmdb_path = Path(
            os.environ.get(
                "ECA57_LMDB_PATH",
                SAT_REVSYNTH_PATH / "data" / "eca57_identities_lmdb"
            )
        )

        if not lmdb_path.exists():
            logger.warning(f"ECA57 LMDB not found at {lmdb_path}")
            return []

        env = lmdb.open(str(lmdb_path), max_dbs=10, readonly=True, lock=False)
        try:
            meta_db = env.open_db(b"metadata")
            circuits_db = env.open_db(b"circuits")

            with env.begin() as txn:
                # Get all configs
                configs = []
                cursor = txn.cursor(db=meta_db)
                for key, value in cursor:
                    config = key.decode()
                    count = int(value.decode())
                    w = int(config[1])
                    g = int(config[3:])

                    # Apply filters
                    if width is not None and w != width:
                        continue
                    if gate_count is not None and g != gate_count:
                        continue

                    configs.append((w, g, count))

                # Fetch circuits from matching configs
                for w, g, count in configs:
                    if len(results) >= limit:
                        break

                    for i in range(min(count, limit - len(results))):
                        key = f"w{w}g{g}:{i:08d}".encode()
                        data = txn.get(key, db=circuits_db)
                        if data:
                            record = json.loads(data.decode())
                            gates = record["gates"]

                            results.append(UnifiedCircuitResult(
                                id=f"eca57-lmdb:{w}:{g}:{i}",
                                source="eca57-lmdb",
                                width=w,
                                gateCount=g,
                                gates=gates,
                                gateString=gates_to_gate_string(gates),
                                isIdentity=True,
                                isRepresentative=True,
                                equivalenceClassSize=record.get("equivalence_class_size", 1)
                            ))
        finally:
            env.close()

    except Exception as e:
        logger.error(f"Failed to search ECA57 LMDB: {e}")

    return results


async def search_sqlite(
    width: Optional[int],
    gate_count: Optional[int],
    limit: int
) -> List[UnifiedCircuitResult]:
    """Search SQLite circuits database."""
    results = []

    try:
        # Import from parent package
        from identity_factory.database import Database

        db_path = Path(
            os.environ.get(
                "IDENTITY_FACTORY_DB_PATH",
                Path(__file__).parent.parent.parent / "identity_circuits.db"
            )
        )

        if not db_path.exists():
            logger.warning(f"SQLite DB not found at {db_path}")
            return []

        db = Database(str(db_path))

        # Build query
        query = "SELECT * FROM circuits WHERE 1=1"
        params = []

        if width is not None:
            query += " AND width = ?"
            params.append(width)

        if gate_count is not None:
            query += " AND gate_count = ?"
            params.append(gate_count)

        query += f" LIMIT {limit}"

        cursor = db.conn.execute(query, params)
        for row in cursor:
            circuit_id = row[0]
            w = row[1]
            g = row[2]
            gates_json = row[3]
            perm_json = row[4]

            gates = json.loads(gates_json) if isinstance(gates_json, str) else gates_json

            # Convert gate format if needed (may be tuples with gate type)
            normalized_gates = []
            for gate in gates:
                if isinstance(gate, (list, tuple)):
                    if len(gate) == 3 and all(isinstance(x, int) for x in gate):
                        normalized_gates.append(list(gate))
                    elif len(gate) >= 3:
                        # Format: ('CCX', target, ctrl1, ctrl2) or similar
                        normalized_gates.append([gate[-3], gate[-2], gate[-1]])

            if not normalized_gates:
                continue

            # Check if identity (permutation is identity)
            perm = json.loads(perm_json) if isinstance(perm_json, str) else perm_json
            is_identity = perm == list(range(len(perm))) if perm else False

            results.append(UnifiedCircuitResult(
                id=f"sqlite:{circuit_id}",
                source="sqlite",
                width=w,
                gateCount=g,
                gates=normalized_gates,
                gateString=gates_to_gate_string(normalized_gates),
                isIdentity=is_identity,
                isRepresentative=row[7] is None if len(row) > 7 else None  # representative_id
            ))

    except Exception as e:
        logger.error(f"Failed to search SQLite: {e}")

    return results


@router.get("/circuits", response_model=UnifiedSearchResponse)
async def search_circuits(
    width: Optional[int] = Query(None, ge=1, le=64, description="Filter by width"),
    gate_count: Optional[int] = Query(None, ge=1, le=1000, description="Filter by gate count"),
    sources: List[str] = Query(
        ["skeleton", "eca57-lmdb", "sqlite"],
        description="Database sources to query"
    ),
    limit: int = Query(30, ge=1, le=100, description="Maximum results to return"),
    is_identity_only: bool = Query(True, description="Only return identity circuits"),
):
    """
    Search circuits across multiple databases.

    Supports filtering by width, gate count, and database sources.
    Returns results in a unified format with gate_string ready for canvas loading.
    """
    valid_sources = {"skeleton", "eca57-lmdb", "sqlite"}
    sources = [s for s in sources if s in valid_sources]

    if not sources:
        sources = ["skeleton", "eca57-lmdb", "sqlite"]

    # Search each source
    all_results = []
    sources_stats = {}

    # Limit per source to balance results
    per_source_limit = max(10, limit // len(sources))

    tasks = []
    if "skeleton" in sources:
        tasks.append(("skeleton", search_skeleton_db(width, gate_count, per_source_limit)))
    if "eca57-lmdb" in sources:
        tasks.append(("eca57-lmdb", search_eca57_lmdb(width, gate_count, per_source_limit)))
    if "sqlite" in sources:
        tasks.append(("sqlite", search_sqlite(width, gate_count, per_source_limit)))

    # Run searches concurrently
    for source, task in tasks:
        try:
            results = await task
            sources_stats[source] = len(results)

            # Filter for identity if requested
            if is_identity_only:
                results = [r for r in results if r.isIdentity]

            all_results.extend(results)
        except Exception as e:
            logger.error(f"Search failed for {source}: {e}")
            sources_stats[source] = 0

    # Sort by gate count, then width
    all_results.sort(key=lambda r: (r.gateCount, r.width))

    # Apply overall limit
    all_results = all_results[:limit]

    return UnifiedSearchResponse(
        results=all_results,
        total=len(all_results),
        sourcesQueried=sources,
        sourcesStats=sources_stats
    )
