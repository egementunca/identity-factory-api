#!/usr/bin/env python3
"""
Import wire shuffler JSON results into SQLite + optional LMDB mirror.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb

# Ensure identity_factory is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_factory.wire_shuffler_db import (
    WirePermutationRecord,
    WireShufflerCircuit,
    WireShufflerMetrics,
    WireShufflerDatabase,
    WireShufflerRun,
)


def compute_cycles(perm: List[int]) -> List[List[int]]:
    n = len(perm)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle = []
        while not visited[j]:
            visited[j] = True
            cycle.append(j)
            j = perm[j]
        cycles.append(cycle)
    return cycles


def cycle_type(perm: List[int]) -> List[int]:
    lengths = [len(c) for c in compute_cycles(perm)]
    lengths.sort(reverse=True)
    return lengths


def perm_stats(perm: List[int]) -> Dict[str, int]:
    n = len(perm)
    fixed = sum(1 for i, p in enumerate(perm) if p == i)
    cycles = len(compute_cycles(perm))
    hamming = n - fixed
    swap_distance = n - cycles
    return {
        "fixed_points": fixed,
        "hamming": hamming,
        "cycles": cycles,
        "swap_distance": swap_distance,
    }


def parity_from_swap_distance(swap_distance: int) -> str:
    return "even" if (swap_distance % 2 == 0) else "odd"


def full_perm_from_wire_perm(perm: List[int]) -> List[int]:
    n = len(perm)
    size = 1 << n
    values = [0] * size
    for x in range(size):
        y = 0
        for out_idx, src_idx in enumerate(perm):
            bit = (x >> src_idx) & 1
            y |= bit << out_idx
        values[x] = y
    return values


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

    # Wire usage / degree
    degrees = [0] * width
    for t, c1, c2 in gates:
        for w in (t, c1, c2):
            degrees[w] += 1
    wires_used = sum(1 for d in degrees if d > 0)
    wire_coverage = wires_used / width if width > 0 else 0.0
    max_wire_degree = max(degrees)
    avg_wire_degree = sum(degrees) / width if width > 0 else 0.0

    # Collisions
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
    collision_density = (
        total_collisions / total_pairs if total_pairs > 0 else 0.0
    )

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


def open_lmdb(path: Path):
    env = lmdb.open(
        str(path),
        map_size=1 << 30,
        max_dbs=10,
    )
    dbs = {
        "meta": env.open_db(b"ws_meta", create=True),
        "perm": env.open_db(b"ws_perm", create=True),
        "perm_by_width": env.open_db(b"ws_perm_by_width", create=True),
        "perm_by_hamming": env.open_db(b"ws_perm_by_hamming", create=True),
        "perm_by_swap": env.open_db(b"ws_perm_by_swap", create=True),
        "circ": env.open_db(b"ws_circ", create=True),
        "circ_by_perm": env.open_db(b"ws_circ_by_perm", create=True),
    }
    return env, dbs


def lmdb_put(txn, db, key: str, value: Optional[dict] = None):
    k = key.encode("ascii")
    if value is None:
        txn.put(k, b"", db=db)
    else:
        txn.put(k, json.dumps(value).encode("utf-8"), db=db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import wire shuffler JSON into DBs")
    parser.add_argument("--json", required=True, help="Input JSON from synth_wire_shuffle.py")
    parser.add_argument("--db-path", help="SQLite DB path (default: wire_shuffler.db)")
    parser.add_argument("--lmdb-path", help="LMDB path (default: wire_shuffler.lmdb)")
    parser.add_argument("--no-lmdb", action="store_true", help="Disable LMDB mirror")
    parser.add_argument("--width", type=int, help="Override width")
    parser.add_argument("--min-gates", type=int, default=0)
    parser.add_argument("--max-gates", type=int, default=0)
    parser.add_argument("--solver", type=str, default="cadical153")
    parser.add_argument("--order", type=str, default="lex")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--require-all-wires", action="store_true")
    parser.add_argument("--notes", type=str)
    parser.add_argument("--git-sha", type=str)
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())
    results = data.get("results", [])
    if not results:
        raise ValueError("No results in JSON")

    width = args.width or data.get("width") or len(results[0]["perm"])

    db = WireShufflerDatabase(args.db_path)

    run = WireShufflerRun(
        id=None,
        width=width,
        min_gates=args.min_gates,
        max_gates=args.max_gates,
        solver=args.solver,
        require_all_wires=args.require_all_wires,
        order=args.order,
        seed=args.seed,
        status="running",
        notes=args.notes,
        git_sha=args.git_sha,
        started_at=datetime.utcnow().isoformat(),
    )
    run_id = db.start_run(run)

    lmdb_env = None
    lmdb_dbs = None
    if not args.no_lmdb:
        lmdb_path = Path(args.lmdb_path) if args.lmdb_path else (Path(args.db_path or db.db_path).parent / "wire_shuffler.lmdb")
        lmdb_env, lmdb_dbs = open_lmdb(lmdb_path)

    for entry in results:
        perm = entry["perm"]
        stats = entry.get("stats")
        if not stats or any(k not in stats for k in ["fixed_points", "hamming", "cycles", "swap_distance"]):
            stats = perm_stats(perm)
        ctype = cycle_type(perm)
        parity = parity_from_swap_distance(stats["swap_distance"])
        perm_hash = db.compute_perm_hash(perm)

        perm_rec = WirePermutationRecord(
            id=None,
            width=width,
            wire_perm=perm,
            wire_perm_hash=perm_hash,
            fixed_points=stats["fixed_points"],
            hamming=stats["hamming"],
            cycles=stats["cycles"],
            swap_distance=stats["swap_distance"],
            cycle_type="-".join(map(str, ctype)),
            parity=parity,
            is_identity=stats["hamming"] == 0,
        )
        perm_id = db.upsert_permutation(perm_rec)

        found = bool(entry.get("found"))
        gate_count = entry.get("gate_count") if found else None
        gates = entry.get("gates") if found else None
        circuit_hash = (
            db.compute_circuit_hash(gates, perm_hash) if gates is not None else None
        )

        full_perm = None
        if width <= 6:
            full_perm = full_perm_from_wire_perm(perm)

        circ_rec = WireShufflerCircuit(
            id=None,
            run_id=run_id,
            perm_id=perm_id,
            found=found,
            gate_count=gate_count,
            gates=gates,
            circuit_hash=circuit_hash,
            full_perm=full_perm,
            verify_ok=entry.get("verify_ok", None),
            synth_time_ms=entry.get("synth_time_ms", None),
            is_best=False,
        )
        circuit_id = db.insert_circuit(circ_rec)
        db.update_best_for_perm(perm_id)

        if found and gates is not None:
            metrics = compute_metrics(width, gates)
            db.insert_metrics(
                WireShufflerMetrics(
                    circuit_id=circuit_id,
                    width=width,
                    gate_count=gate_count or 0,
                    wires_used=int(metrics["wires_used"]),
                    wire_coverage=float(metrics["wire_coverage"]),
                    max_wire_degree=int(metrics["max_wire_degree"]),
                    avg_wire_degree=float(metrics["avg_wire_degree"]),
                    adjacent_collisions=int(metrics["adjacent_collisions"]),
                    adjacent_commutes=int(metrics["adjacent_commutes"]),
                    total_collisions=int(metrics["total_collisions"]),
                    collision_density=float(metrics["collision_density"]),
                )
            )

        if lmdb_env and lmdb_dbs:
            with lmdb_env.begin(write=True) as txn:
                # Perm record
                lmdb_put(
                    txn,
                    lmdb_dbs["perm"],
                    f"p:{width}:{perm_hash}",
                    {
                        "wire_perm": perm,
                        "fixed_points": stats["fixed_points"],
                        "hamming": stats["hamming"],
                        "cycles": stats["cycles"],
                        "swap_distance": stats["swap_distance"],
                        "cycle_type": "-".join(map(str, ctype)),
                        "parity": parity,
                        "is_identity": stats["hamming"] == 0,
                    },
                )
                lmdb_put(txn, lmdb_dbs["perm_by_width"], f"pw:{width}:{perm_hash}")
                lmdb_put(
                    txn,
                    lmdb_dbs["perm_by_hamming"],
                    f"ph:{width}:{stats['hamming']}:{perm_hash}",
                )
                lmdb_put(
                    txn,
                    lmdb_dbs["perm_by_swap"],
                    f"ps:{width}:{stats['swap_distance']}:{perm_hash}",
                )

                # Circuit record
                circ_key = f"c:{perm_hash}:{run_id}:{gate_count if gate_count is not None else -1}:{circuit_hash or 'unsat'}"
                lmdb_put(
                    txn,
                    lmdb_dbs["circ"],
                    circ_key,
                    {
                        "perm_hash": perm_hash,
                        "run_id": run_id,
                        "found": found,
                        "gate_count": gate_count,
                        "gates": gates,
                        "full_perm": full_perm,
                        "verify_ok": entry.get("verify_ok", None),
                        "synth_time_ms": entry.get("synth_time_ms", None),
                    },
                )
                lmdb_put(txn, lmdb_dbs["circ_by_perm"], f"cp:{perm_hash}:{run_id}:{gate_count if gate_count is not None else -1}:{circuit_hash or 'unsat'}")

                lmdb_put(
                    txn,
                    lmdb_dbs["meta"],
                    "meta:version",
                    {"version": 1},
                )
                lmdb_put(
                    txn,
                    lmdb_dbs["meta"],
                    "meta:last_run_id",
                    {"run_id": run_id},
                )

    db.finish_run(run_id, status="complete")

    if lmdb_env:
        lmdb_env.close()


if __name__ == "__main__":
    main()
