#!/usr/bin/env python3
"""
Backfill wire_shuffler_metrics for existing circuits.

Computes metrics from gate lists and inserts missing rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Dict, List


def gates_collide(g1: List[int], g2: List[int]) -> bool:
    t1, c1_1, c2_1 = g1
    t2, c1_2, c2_2 = g2
    return t1 in (c1_2, c2_2) or t2 in (c1_1, c2_1)


def compute_metrics(width: int, gates: List[List[int]]) -> Dict[str, float]:
    gate_count = len(gates)
    if gate_count == 0:
        return {
            "wires_used": 0,
            "wire_coverage": 0.0,
            "max_wire_degree": 0,
            "avg_wire_degree": 0.0,
            "adjacent_collisions": 0,
            "adjacent_commutes": 0,
            "total_collisions": 0,
            "collision_density": 0.0,
        }

    degrees = [0] * width
    for t, c1, c2 in gates:
        for w in (t, c1, c2):
            degrees[w] += 1
    wires_used = sum(1 for d in degrees if d > 0)
    wire_coverage = wires_used / width if width > 0 else 0.0
    max_wire_degree = max(degrees)
    avg_wire_degree = sum(degrees) / width if width > 0 else 0.0

    adjacent_collisions = 0
    for i in range(gate_count - 1):
        if gates_collide(gates[i], gates[i + 1]):
            adjacent_collisions += 1
    adjacent_commutes = (gate_count - 1) - adjacent_collisions

    total_collisions = 0
    for i in range(gate_count):
        for j in range(i + 1, gate_count):
            if gates_collide(gates[i], gates[j]):
                total_collisions += 1
    total_pairs = gate_count * (gate_count - 1) / 2
    collision_density = total_collisions / total_pairs if total_pairs > 0 else 0.0

    return {
        "wires_used": wires_used,
        "wire_coverage": wire_coverage,
        "max_wire_degree": max_wire_degree,
        "avg_wire_degree": avg_wire_degree,
        "adjacent_collisions": adjacent_collisions,
        "adjacent_commutes": adjacent_commutes,
        "total_collisions": total_collisions,
        "collision_density": collision_density,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill wire shuffler metrics")
    parser.add_argument(
        "--db-path",
        default="identity-factory-api/wire_shuffler.db",
        help="SQLite DB path",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute metrics even if rows exist",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()

    if args.recompute:
        cur.execute(
            """
            SELECT c.id, c.gates, p.width
            FROM wire_shuffler_circuits c
            JOIN wire_permutations p ON p.id = c.perm_id
            WHERE c.found = 1 AND c.gates IS NOT NULL
        """
        )
    else:
        cur.execute(
            """
            SELECT c.id, c.gates, p.width
            FROM wire_shuffler_circuits c
            JOIN wire_permutations p ON p.id = c.perm_id
            LEFT JOIN wire_shuffler_metrics m ON m.circuit_id = c.id
            WHERE c.found = 1 AND c.gates IS NOT NULL AND m.circuit_id IS NULL
        """
        )

    rows = cur.fetchall()
    updated = 0
    for circuit_id, gates_json, width in rows:
        gates = json.loads(gates_json)
        metrics = compute_metrics(int(width), gates)
        cur.execute(
            """
            INSERT OR REPLACE INTO wire_shuffler_metrics
            (circuit_id, width, gate_count, wires_used, wire_coverage,
             max_wire_degree, avg_wire_degree, adjacent_collisions,
             adjacent_commutes, total_collisions, collision_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                circuit_id,
                int(width),
                len(gates),
                int(metrics["wires_used"]),
                float(metrics["wire_coverage"]),
                int(metrics["max_wire_degree"]),
                float(metrics["avg_wire_degree"]),
                int(metrics["adjacent_collisions"]),
                int(metrics["adjacent_commutes"]),
                int(metrics["total_collisions"]),
                float(metrics["collision_density"]),
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Backfilled {updated} circuit metrics.")


if __name__ == "__main__":
    main()
