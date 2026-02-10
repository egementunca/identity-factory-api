"""
API endpoints for browsing imported identity circuits.

These endpoints provide access to identity circuits imported from external
sources like big_identities.txt, with metrics, filtering, and export capabilities.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Database path for imported identities
IMPORTED_DB_PATH = Path.home() / ".identity_factory" / "imported_identities.db"

router = APIRouter(prefix="/imported-identities", tags=["Imported Identities"])


# Response models
class WireStats(BaseModel):
    wires: int
    count: int
    min_gates: int
    max_gates: int
    avg_gates: float


class GateCountStats(BaseModel):
    gate_count: int
    count: int


class SourceStats(BaseModel):
    source_table: str
    count: int


class ImportedIdentityStats(BaseModel):
    total_circuits: int
    by_wires: List[WireStats]
    by_source: List[SourceStats]
    top_gate_counts: List[GateCountStats]


class ImportedCircuit(BaseModel):
    id: int
    source_table: str
    wires: int
    gate_count: int
    gates: List[List[int]]
    circuit_str: str
    is_verified: bool


class PaginatedImportedCircuits(BaseModel):
    circuits: List[ImportedCircuit]
    total: int
    page: int
    page_size: int
    total_pages: int


def get_imported_db() -> sqlite3.Connection:
    """Get connection to imported identities database."""
    if not IMPORTED_DB_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Imported identities database not found. Run the import script first."
        )
    return sqlite3.connect(IMPORTED_DB_PATH)


@router.get("/stats", response_model=ImportedIdentityStats)
async def get_imported_stats() -> ImportedIdentityStats:
    """
    Get comprehensive statistics on imported identity circuits.

    Returns counts by wire count, source table, and gate count distribution.
    """
    try:
        conn = get_imported_db()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM imported_identities")
        total = cursor.fetchone()[0]

        # By wires
        cursor = conn.execute("""
            SELECT wires, COUNT(*) as count,
                   MIN(gate_count) as min_gates,
                   MAX(gate_count) as max_gates,
                   AVG(gate_count) as avg_gates
            FROM imported_identities
            GROUP BY wires
            ORDER BY wires
        """)
        by_wires = [
            WireStats(
                wires=row[0],
                count=row[1],
                min_gates=row[2],
                max_gates=row[3],
                avg_gates=round(row[4], 2) if row[4] else 0,
            )
            for row in cursor.fetchall()
        ]

        # By source
        cursor = conn.execute("""
            SELECT source_table, COUNT(*)
            FROM imported_identities
            GROUP BY source_table
            ORDER BY COUNT(*) DESC
        """)
        by_source = [
            SourceStats(source_table=row[0], count=row[1])
            for row in cursor.fetchall()
        ]

        # Top gate counts
        cursor = conn.execute("""
            SELECT gate_count, COUNT(*) as count
            FROM imported_identities
            GROUP BY gate_count
            ORDER BY count DESC
            LIMIT 20
        """)
        top_gate_counts = [
            GateCountStats(gate_count=row[0], count=row[1])
            for row in cursor.fetchall()
        ]

        conn.close()

        return ImportedIdentityStats(
            total_circuits=total,
            by_wires=by_wires,
            by_source=by_source,
            top_gate_counts=top_gate_counts,
        )

    except Exception as e:
        logger.error(f"Failed to get imported stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits", response_model=PaginatedImportedCircuits)
async def list_imported_circuits(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    wires: Optional[int] = Query(None, description="Filter by wire count"),
    min_gates: Optional[int] = Query(None, description="Minimum gate count"),
    max_gates: Optional[int] = Query(None, description="Maximum gate count"),
    source_table: Optional[str] = Query(None, description="Filter by source table"),
) -> PaginatedImportedCircuits:
    """
    List imported identity circuits with filtering and pagination.
    """
    try:
        conn = get_imported_db()

        # Build WHERE clause
        conditions = []
        params = []

        if wires is not None:
            conditions.append("wires = ?")
            params.append(wires)
        if min_gates is not None:
            conditions.append("gate_count >= ?")
            params.append(min_gates)
        if max_gates is not None:
            conditions.append("gate_count <= ?")
            params.append(max_gates)
        if source_table:
            conditions.append("source_table = ?")
            params.append(source_table)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM imported_identities WHERE {where_clause}",
            params
        )
        total = cursor.fetchone()[0]

        # Get page of results
        offset = (page - 1) * page_size
        cursor = conn.execute(
            f"""
            SELECT id, source_table, wires, gate_count, gates, circuit_str, is_verified
            FROM imported_identities
            WHERE {where_clause}
            ORDER BY wires, gate_count, id
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset]
        )

        circuits = []
        for row in cursor.fetchall():
            circuits.append(ImportedCircuit(
                id=row[0],
                source_table=row[1],
                wires=row[2],
                gate_count=row[3],
                gates=json.loads(row[4]),
                circuit_str=row[5],
                is_verified=bool(row[6]),
            ))

        conn.close()

        total_pages = (total + page_size - 1) // page_size

        return PaginatedImportedCircuits(
            circuits=circuits,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Failed to list imported circuits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits/{circuit_id}", response_model=ImportedCircuit)
async def get_imported_circuit(circuit_id: int) -> ImportedCircuit:
    """Get a specific imported circuit by ID."""
    try:
        conn = get_imported_db()
        cursor = conn.execute(
            """
            SELECT id, source_table, wires, gate_count, gates, circuit_str, is_verified
            FROM imported_identities
            WHERE id = ?
            """,
            (circuit_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Circuit not found")

        return ImportedCircuit(
            id=row[0],
            source_table=row[1],
            wires=row[2],
            gate_count=row[3],
            gates=json.loads(row[4]),
            circuit_str=row[5],
            is_verified=bool(row[6]),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wire-distribution")
async def get_wire_distribution() -> Dict[str, Any]:
    """
    Get detailed distribution of circuits by wire count.

    Useful for understanding the composition of imported identities.
    """
    try:
        conn = get_imported_db()

        cursor = conn.execute("""
            SELECT
                wires,
                COUNT(*) as count,
                MIN(gate_count) as min_gates,
                MAX(gate_count) as max_gates,
                AVG(gate_count) as avg_gates,
                SUM(gate_count) as total_gates
            FROM imported_identities
            GROUP BY wires
            ORDER BY wires
        """)

        distribution = []
        for row in cursor.fetchall():
            distribution.append({
                "wires": row[0],
                "circuit_count": row[1],
                "min_gates": row[2],
                "max_gates": row[3],
                "avg_gates": round(row[4], 2) if row[4] else 0,
                "total_gates": row[5],
                "density": round(row[1] / (row[3] - row[2] + 1), 2) if row[3] > row[2] else row[1],
            })

        conn.close()

        return {
            "distribution": distribution,
            "summary": {
                "wire_counts": [d["wires"] for d in distribution],
                "total_circuits": sum(d["circuit_count"] for d in distribution),
            }
        }

    except Exception as e:
        logger.error(f"Failed to get wire distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gate-count-histogram")
async def get_gate_count_histogram(
    wires: Optional[int] = Query(None, description="Filter by wire count"),
) -> Dict[str, Any]:
    """
    Get histogram of gate counts for imported identities.
    """
    try:
        conn = get_imported_db()

        if wires:
            cursor = conn.execute("""
                SELECT gate_count, COUNT(*) as count
                FROM imported_identities
                WHERE wires = ?
                GROUP BY gate_count
                ORDER BY gate_count
            """, (wires,))
        else:
            cursor = conn.execute("""
                SELECT gate_count, COUNT(*) as count
                FROM imported_identities
                GROUP BY gate_count
                ORDER BY gate_count
            """)

        histogram = [
            {"gate_count": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "histogram": histogram,
            "wires_filter": wires,
            "total_distinct_gate_counts": len(histogram),
        }

    except Exception as e:
        logger.error(f"Failed to get gate count histogram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random-sample")
async def get_random_sample(
    count: int = Query(10, ge=1, le=100, description="Number of circuits to sample"),
    wires: Optional[int] = Query(None, description="Filter by wire count"),
) -> List[ImportedCircuit]:
    """
    Get a random sample of imported circuits.

    Useful for testing and exploration.
    """
    try:
        conn = get_imported_db()

        if wires:
            cursor = conn.execute("""
                SELECT id, source_table, wires, gate_count, gates, circuit_str, is_verified
                FROM imported_identities
                WHERE wires = ?
                ORDER BY RANDOM()
                LIMIT ?
            """, (wires, count))
        else:
            cursor = conn.execute("""
                SELECT id, source_table, wires, gate_count, gates, circuit_str, is_verified
                FROM imported_identities
                ORDER BY RANDOM()
                LIMIT ?
            """, (count,))

        circuits = []
        for row in cursor.fetchall():
            circuits.append(ImportedCircuit(
                id=row[0],
                source_table=row[1],
                wires=row[2],
                gate_count=row[3],
                gates=json.loads(row[4]),
                circuit_str=row[5],
                is_verified=bool(row[6]),
            ))

        conn.close()
        return circuits

    except Exception as e:
        logger.error(f"Failed to get random sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))
