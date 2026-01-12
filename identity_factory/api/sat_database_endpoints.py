"""
SAT Database API endpoints.
Provides access to SAT-generated circuit databases.
"""

import ast
import logging
"""
WARNING: This module depends on `sat_circuits.db` which is currently missing.
Only `identity_circuits.db` is present.
These endpoints will not function until `sat_circuits.db` is generated or restored.
"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sat-database", tags=["sat-database"])

# Path to SAT database
SAT_DB_PATH = Path(__file__).parent.parent.parent / "sat_circuits.db"


class SATCircuit(BaseModel):
    """Circuit from SAT database."""

    id: int
    width: int
    gate_count: int
    gates: List[Any]  # Can be list of tuples (controls, target) or list of lists
    permutation: List[int]
    perm_cycle: List[List[int]]
    order_val: Optional[int] = None
    is_canonical: bool = False


class SATDatabaseStats(BaseModel):
    """Statistics for SAT database."""

    total_circuits: int
    by_dimension: Dict[str, int]
    max_width: Optional[int] = None
    max_gates: Optional[int] = None
    status: str = "unknown"


class CircuitListResponse(BaseModel):
    """Paginated circuit list."""

    circuits: List[SATCircuit]
    total: int
    offset: int
    limit: int


def get_db_connection():
    """Get database connection."""
    if not SAT_DB_PATH.exists():
        return None
    return sqlite3.connect(str(SAT_DB_PATH))


@router.get("/stats", response_model=SATDatabaseStats)
async def get_sat_database_stats():
    """Get SAT database statistics."""
    conn = get_db_connection()
    if not conn:
        return SATDatabaseStats(total_circuits=0, by_dimension={}, status="not_found")

    try:
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM circuits")
        total = cursor.fetchone()[0]

        # By dimension
        cursor.execute(
            "SELECT width, gate_count, COUNT(*) FROM circuits GROUP BY width, gate_count"
        )
        by_dim = {f"{w}w_{gc}g": count for w, gc, count in cursor.fetchall()}

        # Metadata
        cursor.execute(
            "SELECT key, value FROM metadata WHERE key IN ('max_width', 'max_gates', 'completed_at')"
        )
        metadata = dict(cursor.fetchall())

        status = "complete" if metadata.get("completed_at") else "generating"

        return SATDatabaseStats(
            total_circuits=total,
            by_dimension=by_dim,
            max_width=(
                int(metadata.get("max_width", 0)) if metadata.get("max_width") else None
            ),
            max_gates=(
                int(metadata.get("max_gates", 0)) if metadata.get("max_gates") else None
            ),
            status=status,
        )
    finally:
        conn.close()


@router.get("/circuits", response_model=CircuitListResponse)
async def list_sat_circuits(
    width: Optional[int] = Query(None, description="Filter by width"),
    gate_count: Optional[int] = Query(None, description="Filter by gate count"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List circuits from SAT database."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="SAT database not found")

    try:
        cursor = conn.cursor()

        # Build query
        where_clauses = []
        params = []

        if width is not None:
            where_clauses.append("width = ?")
            params.append(width)
        if gate_count is not None:
            where_clauses.append("gate_count = ?")
            params.append(gate_count)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Count total
        cursor.execute(f"SELECT COUNT(*) FROM circuits WHERE {where_sql}", params)
        total = cursor.fetchone()[0]

        # Get circuits
        cursor.execute(
            f"""
            SELECT id, width, gate_count, gates_str, permutation, perm_cycle, 
                   order_val, is_canonical
            FROM circuits 
            WHERE {where_sql}
            ORDER BY width, gate_count, id
            LIMIT ? OFFSET ?
        """,
            params + [limit, offset],
        )

        circuits = []
        for row in cursor.fetchall():
            try:
                gates = ast.literal_eval(row[3]) if row[3] else []
                perm = ast.literal_eval(row[4]) if row[4] else []
                perm_cycle = ast.literal_eval(row[5]) if row[5] else []

                circuits.append(
                    SATCircuit(
                        id=row[0],
                        width=row[1],
                        gate_count=row[2],
                        gates=gates,
                        permutation=perm,
                        perm_cycle=perm_cycle,
                        order_val=row[6],
                        is_canonical=bool(row[7]),
                    )
                )
            except Exception as e:
                logger.warning(f"Error parsing circuit {row[0]}: {e}")
                continue

        return CircuitListResponse(
            circuits=circuits, total=total, offset=offset, limit=limit
        )
    finally:
        conn.close()


@router.get("/circuit/{circuit_id}")
async def get_sat_circuit(circuit_id: int):
    """Get details of a specific circuit."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="SAT database not found")

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, width, gate_count, gates_str, permutation, perm_cycle,
                   order_val, is_canonical, canonical_hash, created_at
            FROM circuits WHERE id = ?
        """,
            (circuit_id,),
        )

        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail=f"Circuit {circuit_id} not found"
            )

        gates = ast.literal_eval(row[3]) if row[3] else []
        perm = ast.literal_eval(row[4]) if row[4] else []
        perm_cycle = ast.literal_eval(row[5]) if row[5] else []

        # Generate ASCII diagram
        diagram = generate_circuit_diagram(row[1], gates)

        return {
            "id": row[0],
            "width": row[1],
            "gate_count": row[2],
            "gates": gates,
            "permutation": perm,
            "perm_cycle": perm_cycle,
            "order_val": row[6],
            "is_canonical": bool(row[7]),
            "canonical_hash": row[8],
            "created_at": row[9],
            "diagram": diagram,
        }
    finally:
        conn.close()


def generate_circuit_diagram(width: int, gates: List[List[int]]) -> str:
    """Generate ASCII circuit diagram."""
    if not gates:
        return ""

    lines = []
    for w in range(width):
        line = f"{w} ─"
        for gate in gates:
            if len(gate) >= 3:
                target, ctrl1, ctrl2 = gate[0], gate[1], gate[2]
                if w == target:
                    line += "( )─"
                elif w == ctrl1:
                    line += "─●──"
                elif w == ctrl2:
                    line += "─○──"
                elif min(target, ctrl1, ctrl2) < w < max(target, ctrl1, ctrl2):
                    line += "─│──"
                else:
                    line += "────"
            else:
                line += "────"
        lines.append(line)

    return "\n".join(lines)
