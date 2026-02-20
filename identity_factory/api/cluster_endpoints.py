"""
Cluster Database API endpoints.
Provides access to SAT-generated ECA57/MCT identity circuits from cluster computation.
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster-database", tags=["cluster-database"])

# Path to cluster database (override via env)
CLUSTER_DB_PATH = Path(
    os.environ.get(
        "CLUSTER_DB_PATH",
        Path(__file__).parent.parent.parent / "cluster_circuits.db",
    )
).expanduser()


class ClusterCircuit(BaseModel):
    """Circuit from cluster database."""

    id: int
    width: int
    gate_count: int
    gates: str  # Text format: "target:ctrl1,ctrl2;..."
    permutation: str
    gate_set: str
    source: str


class ClusterDatabaseStats(BaseModel):
    """Statistics for cluster database."""

    total_circuits: int
    by_dimension: Dict[str, int]
    by_gate_set: Dict[str, int]
    max_width: Optional[int] = None
    max_gates: Optional[int] = None
    status: str = "ready"


class ClusterCircuitListResponse(BaseModel):
    """Paginated circuit list."""

    circuits: List[ClusterCircuit]
    total: int
    offset: int
    limit: int


def get_db_connection():
    """Get database connection."""
    if not CLUSTER_DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(CLUSTER_DB_PATH))
    conn.create_function("has_diverse_targets", 1, has_diverse_targets)
    conn.create_function("has_reducible_pairs", 1, has_reducible_pairs)
    return conn


@router.get("/stats", response_model=ClusterDatabaseStats)
async def get_cluster_database_stats():
    """Get cluster database statistics."""
    conn = get_db_connection()
    if not conn:
        return ClusterDatabaseStats(
            total_circuits=0, by_dimension={}, by_gate_set={}, status="not_found"
        )

    try:
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM circuits")
        total = cursor.fetchone()[0]

        # By dimension
        cursor.execute(
            """
            SELECT width, gate_count, COUNT(*) 
            FROM circuits 
            GROUP BY width, gate_count
        """
        )
        by_dim = {f"{w}w_{gc}g": count for w, gc, count in cursor.fetchall()}

        # By gate set
        cursor.execute("SELECT gate_set, COUNT(*) FROM circuits GROUP BY gate_set")
        by_gate_set = {gs: count for gs, count in cursor.fetchall()}

        # Max values
        cursor.execute("SELECT MAX(width), MAX(gate_count) FROM circuits")
        max_w, max_g = cursor.fetchone()

        return ClusterDatabaseStats(
            total_circuits=total,
            by_dimension=by_dim,
            by_gate_set=by_gate_set,
            max_width=max_w,
            max_gates=max_g,
            status="ready",
        )
    finally:
        conn.close()


@router.get("/circuits", response_model=ClusterCircuitListResponse)
async def list_cluster_circuits(
    width: Optional[int] = Query(None, description="Filter by width"),
    gate_count: Optional[int] = Query(None, description="Filter by gate count"),
    gate_set: Optional[str] = Query(
        None, description="Filter by gate set (eca57, mct)"
    ),
    diverse_targets: bool = Query(
        False, description="Only show circuits with >1 unique target"
    ),
    exclude_reducible: bool = Query(
        False, description="Exclude circuits with consecutive identical gates"
    ),
    only_reducible: bool = Query(
        False, description="Only show circuits with consecutive identical gates"
    ),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List circuits from cluster database."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Cluster database not found")

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
        if gate_set is not None:
            where_clauses.append("gate_set = ?")
            params.append(gate_set)
        if diverse_targets:
            where_clauses.append("has_diverse_targets(gates) = 1")
        if exclude_reducible:
            where_clauses.append("has_reducible_pairs(gates) = 0")
        if only_reducible:
            where_clauses.append("has_reducible_pairs(gates) = 1")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Count total
        cursor.execute(f"SELECT COUNT(*) FROM circuits WHERE {where_sql}", params)
        total = cursor.fetchone()[0]

        # Get circuits
        cursor.execute(
            f"""
            SELECT id, width, gate_count, gates, permutation, gate_set, source
            FROM circuits 
            WHERE {where_sql}
            ORDER BY width, gate_count, id
            LIMIT ? OFFSET ?
        """,
            params + [limit, offset],
        )

        circuits = []
        for row in cursor.fetchall():
            circuits.append(
                ClusterCircuit(
                    id=row[0],
                    width=row[1],
                    gate_count=row[2],
                    gates=row[3],
                    permutation=row[4],
                    gate_set=row[5] or "eca57",
                    source=row[6] or "cluster",
                )
            )

        return ClusterCircuitListResponse(
            circuits=circuits, total=total, offset=offset, limit=limit
        )
    finally:
        conn.close()


@router.get("/circuit/{circuit_id}")
async def get_cluster_circuit(circuit_id: int):
    """Get details of a specific circuit."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Cluster database not found")

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, width, gate_count, gates, permutation, gate_set, source,
                   circuit_hash, dim_group_id
            FROM circuits WHERE id = ?
        """,
            (circuit_id,),
        )

        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail=f"Circuit {circuit_id} not found"
            )

        # Parse gates to list format
        gates_list = parse_gates_text(row[3], row[5] or "eca57")
        diagram = generate_circuit_diagram(row[1], gates_list, row[5] or "eca57")

        return {
            "id": row[0],
            "width": row[1],
            "gate_count": row[2],
            "gates": row[3],
            "gates_parsed": gates_list,
            "permutation": row[4],
            "gate_set": row[5] or "eca57",
            "source": row[6] or "cluster",
            "circuit_hash": row[7],
            "dim_group_id": row[8],
            "diagram": diagram,
        }
    finally:
        conn.close()


@router.get("/dim-groups")
async def list_dim_groups():
    """List all dimension groups (width x gate_count combinations)."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Cluster database not found")

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, width, gate_count, circuit_count, is_processed
            FROM dim_groups
            ORDER BY width, gate_count
        """
        )

        groups = []
        for row in cursor.fetchall():
            groups.append(
                {
                    "id": row[0],
                    "width": row[1],
                    "gate_count": row[2],
                    "circuit_count": row[3],
                    "is_processed": bool(row[4]),
                }
            )

        return {"dim_groups": groups, "count": len(groups)}
    finally:
        conn.close()


def parse_gates_text(gates_text: str, gate_set: str) -> List[Dict[str, Any]]:
    """Parse gates text format to structured list."""
    if not gates_text:
        return []

    gates = []
    for gate_str in gates_text.split(";"):
        if not gate_str:
            continue
        try:
            parts = gate_str.split(":")
            target = int(parts[0])
            controls = (
                [int(c) for c in parts[1].split(",") if c] if len(parts) > 1 else []
            )
            gates.append({"target": target, "controls": controls, "gate_set": gate_set})
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing gate '{gate_str}': {e}")
            continue

    return gates


def generate_circuit_diagram(
    width: int, gates: List[Dict[str, Any]], gate_set: str
) -> str:
    """Generate ASCII circuit diagram."""
    if not gates:
        return ""

    lines = []
    for w in range(width):
        line = f"{w} ─"
        for gate in gates:
            target = gate.get("target")
            controls = gate.get("controls", [])

            if w == target:
                line += "( )─"
            elif w in controls:
                if gate_set == "eca57" and len(controls) >= 2:
                    # ECA57: first control is positive, second is negative
                    if controls.index(w) == 0:
                        line += "─●──"  # positive control
                    else:
                        line += "─○──"  # negative control
                else:
                    line += "─●──"
            elif target is not None and controls:
                all_wires = [target] + controls
                if min(all_wires) < w < max(all_wires):
                    line += "─│──"
                else:
                    line += "────"
            else:
                line += "────"
        lines.append(line)

    return "\n".join(lines)


def has_diverse_targets(gates_text: str) -> int:
    """Check if circuit has more than one unique target wire."""
    if not gates_text:
        return 0

    unique_targets = set()
    for gate_str in gates_text.split(";"):
        if not gate_str:
            continue
        try:
            target = int(gate_str.split(":")[0])
            unique_targets.add(target)
            if len(unique_targets) > 1:
                return 1
        except (ValueError, IndexError):
            continue

    return 0


def has_reducible_pairs(gates_text: str) -> int:
    """Check if circuit has consecutive identical gates (same target and controls).

    Such circuits are "reducible" because G·G = I for these reversible gates.
    Returns 1 if reducible pairs exist, 0 otherwise.
    """
    if not gates_text:
        return 0

    gates = gates_text.split(";")
    prev_gate = None

    for gate_str in gates:
        if not gate_str:
            continue

        # Normalize gate representation for comparison
        # Format: "target:ctrl1,ctrl2" - normalize by sorting controls
        try:
            parts = gate_str.split(":")
            target = parts[0]
            controls = sorted(parts[1].split(",")) if len(parts) > 1 and parts[1] else []
            normalized = f"{target}:{','.join(controls)}"

            if prev_gate is not None and normalized == prev_gate:
                return 1  # Found consecutive identical gates

            prev_gate = normalized
        except (ValueError, IndexError):
            continue

    return 0
