#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def cycle_type(perm: List[int]) -> str:
    lengths = [len(c) for c in compute_cycles(perm)]
    lengths.sort(reverse=True)
    return "-".join(map(str, lengths))


def perm_stats(perm: List[int]) -> Dict[str, Any]:
    n = len(perm)
    fixed_points = sum(1 for i, p in enumerate(perm) if p == i)
    cycles = len(compute_cycles(perm))
    hamming = n - fixed_points
    swap_distance = n - cycles
    parity = "even" if (swap_distance % 2 == 0) else "odd"
    return {
        "fixed_points": fixed_points,
        "hamming": hamming,
        "cycles": cycles,
        "swap_distance": swap_distance,
        "cycle_type": cycle_type(perm),
        "parity": parity,
        "is_identity": hamming == 0,
    }


def compute_perm_hash(perm: List[int]) -> str:
    perm_str = ",".join(map(str, perm))
    return hashlib.sha256(perm_str.encode()).hexdigest()[:16]


def compute_circuit_hash(gates: List[List[int]]) -> str:
    payload = json.dumps(gates, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wire_permutations (
            id INTEGER PRIMARY KEY,
            width INTEGER NOT NULL,
            wire_perm TEXT NOT NULL,
            wire_perm_hash TEXT NOT NULL,
            fixed_points INTEGER,
            hamming INTEGER,
            cycles INTEGER,
            swap_distance INTEGER,
            cycle_type TEXT,
            parity TEXT,
            is_identity BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_perm_hash_width ON wire_permutations(width, wire_perm_hash)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS waksman_circuits (
            id INTEGER PRIMARY KEY,
            perm_id INTEGER NOT NULL,
            gate_count INTEGER NOT NULL,
            gates TEXT NOT NULL,
            swap_count INTEGER NOT NULL,
            synth_time_ms INTEGER,
            verify_ok BOOLEAN,
            circuit_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (perm_id) REFERENCES wire_permutations(id)
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_waksman_perm ON waksman_circuits(perm_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_waksman_gatecount ON waksman_circuits(gate_count)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_waksman_swapcount ON waksman_circuits(swap_count)"
    )
    conn.commit()


def upsert_permutation(conn: sqlite3.Connection, width: int, perm: List[int]) -> int:
    phash = compute_perm_hash(perm)
    stats = perm_stats(perm)

    conn.execute(
        """
        INSERT OR IGNORE INTO wire_permutations
            (width, wire_perm, wire_perm_hash, fixed_points, hamming, cycles, swap_distance, cycle_type, parity, is_identity)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            width,
            json.dumps(perm, separators=(",", ":")),
            phash,
            stats["fixed_points"],
            stats["hamming"],
            stats["cycles"],
            stats["swap_distance"],
            stats["cycle_type"],
            stats["parity"],
            int(stats["is_identity"]),
        ),
    )
    row = conn.execute(
        "SELECT id FROM wire_permutations WHERE width = ? AND wire_perm_hash = ?",
        (width, phash),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to upsert/select permutation")
    return int(row[0])


def waksman_exists(conn: sqlite3.Connection, perm_id: int, circuit_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM waksman_circuits WHERE perm_id = ? AND circuit_hash = ? LIMIT 1",
        (perm_id, circuit_hash),
    ).fetchone()
    return row is not None


def insert_waksman(
    conn: sqlite3.Connection,
    perm_id: int,
    gates: List[List[int]],
    swap_count: int,
    synth_time_ms: Optional[int],
    verify_ok: Optional[bool],
    circuit_hash: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO waksman_circuits
            (perm_id, gate_count, gates, swap_count, synth_time_ms, verify_ok, circuit_hash)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            perm_id,
            len(gates),
            json.dumps(gates, separators=(",", ":")),
            swap_count,
            synth_time_ms,
            (1 if verify_ok else 0) if verify_ok is not None else None,
            circuit_hash,
        ),
    )
    return int(cursor.lastrowid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Waksman circuits export into wire_shuffler.db")
    parser.add_argument("--json", required=True, help="Export JSON (share/waksman_circuits_export_*.json)")
    parser.add_argument("--db-path", required=True, help="Target SQLite DB path (wire_shuffler.db)")
    parser.add_argument("--no-dedupe", action="store_true", help="Allow duplicate inserts")
    args = parser.parse_args()

    export_path = Path(args.json)
    payload = json.loads(export_path.read_text())
    circuits = payload.get("waksman_circuits", [])
    if not isinstance(circuits, list):
        raise ValueError("Invalid export JSON: expected waksman_circuits list")

    conn = sqlite3.connect(args.db_path)
    ensure_schema(conn)

    inserted = 0
    skipped = 0

    for entry in circuits:
        width = int(entry["width"])
        perm = list(entry["permutation"])
        perm_hash = entry.get("perm_hash")

        computed_perm_hash = compute_perm_hash(perm)
        if perm_hash and perm_hash != computed_perm_hash:
            raise ValueError(
                f"perm_hash mismatch: got {perm_hash}, computed {computed_perm_hash} for perm={perm}"
            )

        gates = entry["gates"]
        if not isinstance(gates, list):
            raise ValueError("Invalid gates")

        swap_count = int(entry.get("swap_count") or 0)
        synth_time_ms = entry.get("synth_time_ms")
        verify_ok = entry.get("verify_ok")

        circuit_hash = entry.get("circuit_hash") or compute_circuit_hash(gates)

        perm_id = upsert_permutation(conn, width, perm)

        if not args.no_dedupe and waksman_exists(conn, perm_id, circuit_hash):
            skipped += 1
            continue

        insert_waksman(
            conn,
            perm_id,
            gates=gates,
            swap_count=swap_count,
            synth_time_ms=int(synth_time_ms) if synth_time_ms is not None else None,
            verify_ok=bool(verify_ok) if verify_ok is not None else None,
            circuit_hash=circuit_hash,
        )
        inserted += 1

    conn.commit()
    conn.close()

    print(f"Imported {inserted} circuits (skipped {skipped} duplicates) into {args.db_path}")


if __name__ == "__main__":
    main()

