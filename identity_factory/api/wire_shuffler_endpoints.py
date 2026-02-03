"""
Wire Shuffler Database API endpoints.
Read-only access to SAT-synthesized wire shuffle circuits.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from identity_factory.wire_shuffler_db import WireShufflerDatabase

router = APIRouter(prefix="/wire-shuffler", tags=["wire-shuffler"])


class WireShufflerStats(BaseModel):
    total_permutations: int
    total_circuits: int
    by_width: Dict[str, int]
    by_gate_count: Dict[str, int]
    by_hamming: Dict[str, int]
    by_swap_distance: Dict[str, int]
    by_cycle_type: Dict[str, int]
    by_parity: Dict[str, int]


class WirePermutationEntry(BaseModel):
    width: int
    wire_perm: List[int]
    wire_perm_hash: str
    fixed_points: int
    hamming: int
    cycles: int
    swap_distance: int
    cycle_type: str
    parity: str
    is_identity: bool
    best_gate_count: Optional[int] = None


class WirePermutationList(BaseModel):
    total: int
    offset: int
    limit: int
    entries: List[WirePermutationEntry]


class WireCircuitEntry(BaseModel):
    id: int
    width: int
    wire_perm_hash: str
    wire_perm: List[int]
    gate_count: Optional[int]
    gates: Optional[List[List[int]]]
    found: bool
    is_best: bool
    full_perm: Optional[List[int]] = None
    metrics: Optional[Dict[str, float]] = None


class WireCircuitList(BaseModel):
    total: int
    offset: int
    limit: int
    entries: List[WireCircuitEntry]


class WireShufflerMetricsSummary(BaseModel):
    count: int
    metrics: Dict[str, Dict[str, float]]


class WireShufflerGateCountRow(BaseModel):
    gate_count: int
    circuit_count: int


class WireShufflerPermutationSummary(BaseModel):
    wire_perm_hash: str
    wire_perm: List[int]
    cycle_type: str
    hamming: int
    swap_distance: int
    total_circuits: int
    found_circuits: int
    min_gate_count: Optional[int]
    max_gate_count: Optional[int]
    gate_counts: Dict[str, int]


class WireShufflerCycleSummary(BaseModel):
    cycle_type: str
    perm_count: int
    circuit_count: int
    min_gate_count: Optional[int]
    max_gate_count: Optional[int]
    gate_counts: Dict[str, int]


class WireShufflerSummary(BaseModel):
    width: int
    total_permutations: int
    total_circuits: int
    gate_count_rows: List[WireShufflerGateCountRow]
    permutation_rows: List[WireShufflerPermutationSummary]
    cycle_rows: List[WireShufflerCycleSummary]


def get_db() -> WireShufflerDatabase:
    return WireShufflerDatabase()


@router.get("/stats", response_model=WireShufflerStats)
async def stats():
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM wire_permutations")
        total_perms = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM wire_shuffler_circuits")
        total_circuits = cur.fetchone()[0]

        cur.execute("SELECT width, COUNT(*) FROM wire_permutations GROUP BY width")
        by_width = {str(w): c for w, c in cur.fetchall()}

        cur.execute(
            "SELECT gate_count, COUNT(*) FROM wire_shuffler_circuits WHERE found = 1 GROUP BY gate_count"
        )
        by_gate = {str(gc): c for gc, c in cur.fetchall() if gc is not None}

        cur.execute(
            "SELECT hamming, COUNT(*) FROM wire_permutations GROUP BY hamming"
        )
        by_hamming = {str(h): c for h, c in cur.fetchall()}

        cur.execute(
            "SELECT swap_distance, COUNT(*) FROM wire_permutations GROUP BY swap_distance"
        )
        by_swap = {str(s): c for s, c in cur.fetchall()}

        cur.execute(
            "SELECT cycle_type, COUNT(*) FROM wire_permutations GROUP BY cycle_type"
        )
        by_cycle_type = {str(ct): c for ct, c in cur.fetchall() if ct is not None}

        cur.execute("SELECT parity, COUNT(*) FROM wire_permutations GROUP BY parity")
        by_parity = {str(p): c for p, c in cur.fetchall() if p is not None}

        return WireShufflerStats(
            total_permutations=total_perms,
            total_circuits=total_circuits,
            by_width=by_width,
            by_gate_count=by_gate,
            by_hamming=by_hamming,
            by_swap_distance=by_swap,
            by_cycle_type=by_cycle_type,
            by_parity=by_parity,
        )
    finally:
        conn.close()


@router.get("/permutations", response_model=WirePermutationList)
async def list_permutations(
    width: Optional[int] = Query(None),
    hamming: Optional[int] = Query(None),
    swap_distance: Optional[int] = Query(None),
    cycle_type: Optional[str] = Query(None),
    parity: Optional[str] = Query(None),
    is_identity: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        where = []
        params = []
        if width is not None:
            where.append("p.width = ?")
            params.append(width)
        if hamming is not None:
            where.append("p.hamming = ?")
            params.append(hamming)
        if swap_distance is not None:
            where.append("p.swap_distance = ?")
            params.append(swap_distance)
        if cycle_type is not None:
            where.append("p.cycle_type = ?")
            params.append(cycle_type)
        if parity is not None:
            where.append("p.parity = ?")
            params.append(parity)
        if is_identity is not None:
            where.append("p.is_identity = ?")
            params.append(1 if is_identity else 0)

        where_sql = " AND ".join(where) if where else "1=1"

        cur.execute(f"SELECT COUNT(*) FROM wire_permutations p WHERE {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT p.*, (
                SELECT MIN(c.gate_count) FROM wire_shuffler_circuits c
                WHERE c.perm_id = p.id AND c.found = 1
            ) AS best_gate_count
            FROM wire_permutations p
            WHERE {where_sql}
            ORDER BY p.width, p.hamming DESC, p.swap_distance DESC
            LIMIT ? OFFSET ?
        """,
            params + [limit, offset],
        )

        entries = []
        for row in cur.fetchall():
            entries.append(
                WirePermutationEntry(
                    width=row[1],
                    wire_perm=json.loads(row[2]),
                    wire_perm_hash=row[3],
                    fixed_points=row[4],
                    hamming=row[5],
                    cycles=row[6],
                    swap_distance=row[7],
                    cycle_type=row[8],
                    parity=row[9],
                    is_identity=bool(row[10]),
                    best_gate_count=row[12],
                )
            )

        return WirePermutationList(total=total, offset=offset, limit=limit, entries=entries)
    finally:
        conn.close()


@router.get("/circuits", response_model=WireCircuitList)
async def list_circuits(
    width: Optional[int] = Query(None),
    gate_count: Optional[int] = Query(None),
    found: Optional[bool] = Query(None),
    perm_hash: Optional[str] = Query(None),
    is_best: Optional[bool] = Query(None),
    min_wire_coverage: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_wire_coverage: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_collision_density: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_collision_density: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_adjacent_collisions: Optional[int] = Query(None, ge=0),
    max_adjacent_collisions: Optional[int] = Query(None, ge=0),
    min_total_collisions: Optional[int] = Query(None, ge=0),
    max_total_collisions: Optional[int] = Query(None, ge=0),
    min_max_wire_degree: Optional[int] = Query(None, ge=0),
    max_max_wire_degree: Optional[int] = Query(None, ge=0),
    min_avg_wire_degree: Optional[float] = Query(None, ge=0.0),
    max_avg_wire_degree: Optional[float] = Query(None, ge=0.0),
    min_wires_used: Optional[int] = Query(None, ge=0),
    max_wires_used: Optional[int] = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        where = []
        params = []
        if width is not None:
            where.append("p.width = ?")
            params.append(width)
        if gate_count is not None:
            where.append("c.gate_count = ?")
            params.append(gate_count)
        if found is not None:
            where.append("c.found = ?")
            params.append(1 if found else 0)
        if perm_hash is not None:
            where.append("p.wire_perm_hash = ?")
            params.append(perm_hash)
        if is_best is not None:
            where.append("c.is_best = ?")
            params.append(1 if is_best else 0)
        if min_wire_coverage is not None:
            where.append("m.wire_coverage >= ?")
            params.append(min_wire_coverage)
        if max_wire_coverage is not None:
            where.append("m.wire_coverage <= ?")
            params.append(max_wire_coverage)
        if min_collision_density is not None:
            where.append("m.collision_density >= ?")
            params.append(min_collision_density)
        if max_collision_density is not None:
            where.append("m.collision_density <= ?")
            params.append(max_collision_density)
        if min_adjacent_collisions is not None:
            where.append("m.adjacent_collisions >= ?")
            params.append(min_adjacent_collisions)
        if max_adjacent_collisions is not None:
            where.append("m.adjacent_collisions <= ?")
            params.append(max_adjacent_collisions)
        if min_total_collisions is not None:
            where.append("m.total_collisions >= ?")
            params.append(min_total_collisions)
        if max_total_collisions is not None:
            where.append("m.total_collisions <= ?")
            params.append(max_total_collisions)
        if min_max_wire_degree is not None:
            where.append("m.max_wire_degree >= ?")
            params.append(min_max_wire_degree)
        if max_max_wire_degree is not None:
            where.append("m.max_wire_degree <= ?")
            params.append(max_max_wire_degree)
        if min_avg_wire_degree is not None:
            where.append("m.avg_wire_degree >= ?")
            params.append(min_avg_wire_degree)
        if max_avg_wire_degree is not None:
            where.append("m.avg_wire_degree <= ?")
            params.append(max_avg_wire_degree)
        if min_wires_used is not None:
            where.append("m.wires_used >= ?")
            params.append(min_wires_used)
        if max_wires_used is not None:
            where.append("m.wires_used <= ?")
            params.append(max_wires_used)

        where_sql = " AND ".join(where) if where else "1=1"

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM wire_shuffler_circuits c
            JOIN wire_permutations p ON p.id = c.perm_id
            LEFT JOIN wire_shuffler_metrics m ON m.circuit_id = c.id
            WHERE {where_sql}
        """,
            params,
        )
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT c.id, p.width, p.wire_perm_hash, p.wire_perm, c.gate_count, c.gates,
                   c.found, c.is_best, c.full_perm,
                   m.wires_used, m.wire_coverage, m.max_wire_degree, m.avg_wire_degree,
                   m.adjacent_collisions, m.adjacent_commutes, m.total_collisions, m.collision_density
            FROM wire_shuffler_circuits c
            JOIN wire_permutations p ON p.id = c.perm_id
            LEFT JOIN wire_shuffler_metrics m ON m.circuit_id = c.id
            WHERE {where_sql}
            ORDER BY p.width, c.gate_count, c.id
            LIMIT ? OFFSET ?
        """,
            params + [limit, offset],
        )

        entries = []
        for row in cur.fetchall():
            metrics = None
            if row[9] is not None:
                metrics = {
                    "wires_used": row[9],
                    "wire_coverage": row[10],
                    "max_wire_degree": row[11],
                    "avg_wire_degree": row[12],
                    "adjacent_collisions": row[13],
                    "adjacent_commutes": row[14],
                    "total_collisions": row[15],
                    "collision_density": row[16],
                }
            entries.append(
                WireCircuitEntry(
                    id=row[0],
                    width=row[1],
                    wire_perm_hash=row[2],
                    wire_perm=json.loads(row[3]),
                    gate_count=row[4],
                    gates=json.loads(row[5]) if row[5] else None,
                    found=bool(row[6]),
                    is_best=bool(row[7]),
                    full_perm=json.loads(row[8]) if row[8] else None,
                    metrics=metrics,
                )
            )

        return WireCircuitList(total=total, offset=offset, limit=limit, entries=entries)
    finally:
        conn.close()


@router.get("/permutation/{perm_hash}", response_model=WirePermutationEntry)
async def get_permutation(perm_hash: str):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.*, (
                SELECT MIN(c.gate_count) FROM wire_shuffler_circuits c
                WHERE c.perm_id = p.id AND c.found = 1
            ) AS best_gate_count
            FROM wire_permutations p
            WHERE p.wire_perm_hash = ?
        """,
            (perm_hash,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Permutation not found")
        return WirePermutationEntry(
            width=row[1],
            wire_perm=json.loads(row[2]),
            wire_perm_hash=row[3],
            fixed_points=row[4],
            hamming=row[5],
            cycles=row[6],
            swap_distance=row[7],
            cycle_type=row[8],
            parity=row[9],
            is_identity=bool(row[10]),
            best_gate_count=row[12],
        )
    finally:
        conn.close()


@router.get("/circuit/{circuit_id}", response_model=WireCircuitEntry)
async def get_circuit(circuit_id: int):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.id, p.width, p.wire_perm_hash, p.wire_perm, c.gate_count, c.gates,
                   c.found, c.is_best, c.full_perm,
                   m.wires_used, m.wire_coverage, m.max_wire_degree, m.avg_wire_degree,
                   m.adjacent_collisions, m.adjacent_commutes, m.total_collisions, m.collision_density
            FROM wire_shuffler_circuits c
            JOIN wire_permutations p ON p.id = c.perm_id
            LEFT JOIN wire_shuffler_metrics m ON m.circuit_id = c.id
            WHERE c.id = ?
        """,
            (circuit_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Circuit not found")
        metrics = None
        if row[9] is not None:
            metrics = {
                "wires_used": row[9],
                "wire_coverage": row[10],
                "max_wire_degree": row[11],
                "avg_wire_degree": row[12],
                "adjacent_collisions": row[13],
                "adjacent_commutes": row[14],
                "total_collisions": row[15],
                "collision_density": row[16],
            }
        return WireCircuitEntry(
            id=row[0],
            width=row[1],
            wire_perm_hash=row[2],
            wire_perm=json.loads(row[3]),
            gate_count=row[4],
            gates=json.loads(row[5]) if row[5] else None,
            found=bool(row[6]),
            is_best=bool(row[7]),
            full_perm=json.loads(row[8]) if row[8] else None,
            metrics=metrics,
        )
    finally:
        conn.close()


@router.get("/metrics-summary", response_model=WireShufflerMetricsSummary)
async def metrics_summary(
    width: Optional[int] = Query(None),
    best_only: bool = Query(True),
):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        where = ["c.found = 1"]
        params: List[object] = []
        if width is not None:
            where.append("p.width = ?")
            params.append(width)
        if best_only:
            where.append("c.is_best = 1")

        where_sql = " AND ".join(where) if where else "1=1"

        cur.execute(
            f"""
            SELECT
                COUNT(*) AS count,
                MIN(m.wire_coverage), AVG(m.wire_coverage), MAX(m.wire_coverage),
                MIN(m.collision_density), AVG(m.collision_density), MAX(m.collision_density),
                MIN(m.adjacent_collisions), AVG(m.adjacent_collisions), MAX(m.adjacent_collisions),
                MIN(m.total_collisions), AVG(m.total_collisions), MAX(m.total_collisions),
                MIN(m.max_wire_degree), AVG(m.max_wire_degree), MAX(m.max_wire_degree),
                MIN(m.avg_wire_degree), AVG(m.avg_wire_degree), MAX(m.avg_wire_degree),
                MIN(m.wires_used), AVG(m.wires_used), MAX(m.wires_used)
            FROM wire_shuffler_metrics m
            JOIN wire_shuffler_circuits c ON c.id = m.circuit_id
            JOIN wire_permutations p ON p.id = c.perm_id
            WHERE {where_sql}
        """,
            params,
        )
        row = cur.fetchone()
        count = int(row[0] or 0)

        def metric_range(min_v, avg_v, max_v):
            return {
                "min": float(min_v or 0),
                "avg": float(avg_v or 0),
                "max": float(max_v or 0),
            }

        metrics = {
            "wire_coverage": metric_range(row[1], row[2], row[3]),
            "collision_density": metric_range(row[4], row[5], row[6]),
            "adjacent_collisions": metric_range(row[7], row[8], row[9]),
            "total_collisions": metric_range(row[10], row[11], row[12]),
            "max_wire_degree": metric_range(row[13], row[14], row[15]),
            "avg_wire_degree": metric_range(row[16], row[17], row[18]),
            "wires_used": metric_range(row[19], row[20], row[21]),
        }

        return WireShufflerMetricsSummary(count=count, metrics=metrics)
    finally:
        conn.close()


@router.get("/summary", response_model=WireShufflerSummary)
async def summary(
    width: int = Query(..., ge=1),
    include_unsat: bool = Query(False),
):
    db = get_db()
    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id, p.wire_perm_hash, p.wire_perm, p.cycle_type, p.hamming, p.swap_distance,
                   c.gate_count, c.found
            FROM wire_permutations p
            LEFT JOIN wire_shuffler_circuits c ON c.perm_id = p.id
            WHERE p.width = ?
            ORDER BY p.id
        """,
            (width,),
        )

        perm_rows: Dict[int, Dict[str, object]] = {}
        gate_count_rows: Dict[int, int] = {}
        total_circuits = 0

        for (
            perm_id,
            perm_hash,
            wire_perm_json,
            cycle_type,
            hamming,
            swap_distance,
            gate_count,
            found,
        ) in cur.fetchall():
            perm = perm_rows.get(perm_id)
            if perm is None:
                perm = {
                    "wire_perm_hash": perm_hash,
                    "wire_perm": json.loads(wire_perm_json),
                    "cycle_type": cycle_type,
                    "hamming": hamming,
                    "swap_distance": swap_distance,
                    "total_circuits": 0,
                    "found_circuits": 0,
                    "min_gate_count": None,
                    "max_gate_count": None,
                    "gate_counts": {},
                }
                perm_rows[perm_id] = perm

            if gate_count is None and found is None:
                continue

            total_circuits += 1
            perm["total_circuits"] = int(perm["total_circuits"]) + 1

            is_found = bool(found)
            if is_found:
                perm["found_circuits"] = int(perm["found_circuits"]) + 1
                gc = int(gate_count) if gate_count is not None else None
                if gc is not None:
                    gate_counts = perm["gate_counts"]
                    gate_counts[str(gc)] = gate_counts.get(str(gc), 0) + 1
                    gate_count_rows[gc] = gate_count_rows.get(gc, 0) + 1
                    min_gc = perm["min_gate_count"]
                    max_gc = perm["max_gate_count"]
                    perm["min_gate_count"] = gc if min_gc is None else min(min_gc, gc)
                    perm["max_gate_count"] = gc if max_gc is None else max(max_gc, gc)
            elif include_unsat:
                gc = int(gate_count) if gate_count is not None else None
                if gc is not None:
                    gate_counts = perm["gate_counts"]
                    gate_counts[str(gc)] = gate_counts.get(str(gc), 0) + 1
                    gate_count_rows[gc] = gate_count_rows.get(gc, 0) + 1
                    min_gc = perm["min_gate_count"]
                    max_gc = perm["max_gate_count"]
                    perm["min_gate_count"] = gc if min_gc is None else min(min_gc, gc)
                    perm["max_gate_count"] = gc if max_gc is None else max(max_gc, gc)

        cycle_rows: Dict[str, Dict[str, object]] = {}
        for perm in perm_rows.values():
            ctype = str(perm["cycle_type"])
            cycle = cycle_rows.get(ctype)
            if cycle is None:
                cycle = {
                    "cycle_type": ctype,
                    "perm_count": 0,
                    "circuit_count": 0,
                    "min_gate_count": None,
                    "max_gate_count": None,
                    "gate_counts": {},
                }
                cycle_rows[ctype] = cycle

            cycle["perm_count"] = int(cycle["perm_count"]) + 1
            cycle["circuit_count"] = int(cycle["circuit_count"]) + int(
                perm["found_circuits"]
            )

            min_gc = perm["min_gate_count"]
            max_gc = perm["max_gate_count"]
            if min_gc is not None:
                cycle["min_gate_count"] = (
                    min_gc
                    if cycle["min_gate_count"] is None
                    else min(int(cycle["min_gate_count"]), int(min_gc))
                )
            if max_gc is not None:
                cycle["max_gate_count"] = (
                    max_gc
                    if cycle["max_gate_count"] is None
                    else max(int(cycle["max_gate_count"]), int(max_gc))
                )

            for gc, count in perm["gate_counts"].items():
                cycle["gate_counts"][gc] = cycle["gate_counts"].get(gc, 0) + int(count)

        perm_list = [
            WireShufflerPermutationSummary(**perm) for perm in perm_rows.values()
        ]
        perm_list.sort(key=lambda p: (p.hamming, p.swap_distance, p.wire_perm_hash))

        cycle_list = [
            WireShufflerCycleSummary(**cycle) for cycle in cycle_rows.values()
        ]
        cycle_list.sort(key=lambda c: (c.cycle_type))

        gate_count_list = [
            WireShufflerGateCountRow(gate_count=gc, circuit_count=count)
            for gc, count in sorted(gate_count_rows.items())
        ]

        return WireShufflerSummary(
            width=width,
            total_permutations=len(perm_rows),
            total_circuits=total_circuits,
            gate_count_rows=gate_count_list,
            permutation_rows=perm_list,
            cycle_rows=cycle_list,
        )
    finally:
        conn.close()
