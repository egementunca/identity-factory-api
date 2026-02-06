"""
Wire shuffler database (SQLite).

Stores SAT-synthesized ECA57 circuits that implement wire permutations:
    y[i] = x[w[i]]
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WireShufflerRun:
    id: Optional[int]
    width: int
    min_gates: int
    max_gates: int
    solver: str
    require_all_wires: bool
    order: str
    seed: Optional[int]
    status: str
    notes: Optional[str] = None
    git_sha: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class WirePermutationRecord:
    id: Optional[int]
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
    created_at: Optional[str] = None


@dataclass
class WireShufflerCircuit:
    id: Optional[int]
    run_id: int
    perm_id: int
    found: bool
    gate_count: Optional[int]
    gates: Optional[List[Tuple[int, int, int]]]
    circuit_hash: Optional[str]
    full_perm: Optional[List[int]]
    verify_ok: Optional[bool]
    synth_time_ms: Optional[int]
    is_best: bool = False
    created_at: Optional[str] = None


@dataclass
class WireShufflerMetrics:
    circuit_id: int
    width: int
    gate_count: int
    wires_used: int
    wire_coverage: float
    max_wire_degree: int
    avg_wire_degree: float
    adjacent_collisions: int
    adjacent_commutes: int
    total_collisions: int
    collision_density: float


class WireShufflerDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent.parent / "wire_shuffler.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wire_shuffler_runs (
                    id INTEGER PRIMARY KEY,
                    width INTEGER NOT NULL,
                    min_gates INTEGER NOT NULL,
                    max_gates INTEGER NOT NULL,
                    solver TEXT NOT NULL,
                    require_all_wires BOOLEAN DEFAULT 0,
                    "order" TEXT,
                    seed INTEGER,
                    status TEXT,
                    notes TEXT,
                    git_sha TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """
            )
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
                """
                CREATE TABLE IF NOT EXISTS wire_shuffler_circuits (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    perm_id INTEGER NOT NULL,
                    found BOOLEAN NOT NULL,
                    gate_count INTEGER,
                    gates TEXT,
                    circuit_hash TEXT,
                    full_perm TEXT,
                    verify_ok BOOLEAN,
                    synth_time_ms INTEGER,
                    is_best BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES wire_shuffler_runs(id),
                    FOREIGN KEY (perm_id) REFERENCES wire_permutations(id)
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wire_shuffler_metrics (
                    circuit_id INTEGER PRIMARY KEY,
                    width INTEGER NOT NULL,
                    gate_count INTEGER NOT NULL,
                    wires_used INTEGER NOT NULL,
                    wire_coverage REAL NOT NULL,
                    max_wire_degree INTEGER NOT NULL,
                    avg_wire_degree REAL NOT NULL,
                    adjacent_collisions INTEGER NOT NULL,
                    adjacent_commutes INTEGER NOT NULL,
                    total_collisions INTEGER NOT NULL,
                    collision_density REAL NOT NULL,
                    FOREIGN KEY (circuit_id) REFERENCES wire_shuffler_circuits(id)
                )
            """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_perm_hash_width ON wire_permutations(width, wire_perm_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perm_width ON wire_permutations(width)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perm_hamming ON wire_permutations(width, hamming)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perm_swapdist ON wire_permutations(width, swap_distance)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perm_cycle_type ON wire_permutations(width, cycle_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circ_perm ON wire_shuffler_circuits(perm_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circ_run ON wire_shuffler_circuits(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circ_gatecount ON wire_shuffler_circuits(gate_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circ_found ON wire_shuffler_circuits(found)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_coverage ON wire_shuffler_metrics(wire_coverage)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_adj_collisions ON wire_shuffler_metrics(adjacent_collisions)"
            )
            conn.commit()

    @staticmethod
    def compute_perm_hash(perm: List[int]) -> str:
        perm_str = ",".join(map(str, perm))
        return hashlib.sha256(perm_str.encode()).hexdigest()[:16]

    @staticmethod
    def compute_circuit_hash(gates: List[Tuple[int, int, int]], perm_hash: str) -> str:
        gates_str = json.dumps(gates)
        combined = f"{gates_str}|{perm_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def start_run(self, run: WireShufflerRun) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO wire_shuffler_runs
                (width, min_gates, max_gates, solver, require_all_wires, "order", seed, status,
                 notes, git_sha, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.width,
                    run.min_gates,
                    run.max_gates,
                    run.solver,
                    int(run.require_all_wires),
                    run.order,
                    run.seed,
                    run.status,
                    run.notes,
                    run.git_sha,
                    run.started_at or datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def finish_run(self, run_id: int, status: str = "complete") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wire_shuffler_runs
                SET status = ?, completed_at = ?
                WHERE id = ?
            """,
                (status, datetime.utcnow().isoformat(), run_id),
            )
            conn.commit()

    def upsert_permutation(self, record: WirePermutationRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id FROM wire_permutations
                WHERE width = ? AND wire_perm_hash = ?
            """,
                (record.width, record.wire_perm_hash),
            )
            row = cursor.fetchone()
            if row:
                return int(row["id"])

            cursor = conn.execute(
                """
                INSERT INTO wire_permutations
                (width, wire_perm, wire_perm_hash, fixed_points, hamming, cycles, swap_distance,
                 cycle_type, parity, is_identity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.width,
                    json.dumps(record.wire_perm),
                    record.wire_perm_hash,
                    record.fixed_points,
                    record.hamming,
                    record.cycles,
                    record.swap_distance,
                    record.cycle_type,
                    record.parity,
                    int(record.is_identity),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def insert_circuit(self, record: WireShufflerCircuit) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO wire_shuffler_circuits
                (run_id, perm_id, found, gate_count, gates, circuit_hash, full_perm,
                 verify_ok, synth_time_ms, is_best)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.run_id,
                    record.perm_id,
                    int(record.found),
                    record.gate_count,
                    json.dumps(record.gates) if record.gates is not None else None,
                    record.circuit_hash,
                    json.dumps(record.full_perm) if record.full_perm is not None else None,
                    int(record.verify_ok) if record.verify_ok is not None else None,
                    record.synth_time_ms,
                    int(record.is_best),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def insert_metrics(self, metrics: WireShufflerMetrics) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO wire_shuffler_metrics
                (circuit_id, width, gate_count, wires_used, wire_coverage,
                 max_wire_degree, avg_wire_degree, adjacent_collisions,
                 adjacent_commutes, total_collisions, collision_density)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.circuit_id,
                    metrics.width,
                    metrics.gate_count,
                    metrics.wires_used,
                    metrics.wire_coverage,
                    metrics.max_wire_degree,
                    metrics.avg_wire_degree,
                    metrics.adjacent_collisions,
                    metrics.adjacent_commutes,
                    metrics.total_collisions,
                    metrics.collision_density,
                ),
            )
            conn.commit()

    def update_best_for_perm(self, perm_id: int) -> None:
        with self._connect() as conn:
            # Clear
            conn.execute(
                "UPDATE wire_shuffler_circuits SET is_best = 0 WHERE perm_id = ?",
                (perm_id,),
            )
            # Set best (min gate_count among found circuits)
            cursor = conn.execute(
                """
                SELECT id FROM wire_shuffler_circuits
                WHERE perm_id = ? AND found = 1 AND gate_count IS NOT NULL
                ORDER BY gate_count ASC, id ASC
                LIMIT 1
            """,
                (perm_id,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE wire_shuffler_circuits SET is_best = 1 WHERE id = ?",
                    (row["id"],),
                )
            conn.commit()

    # =====================
    # Query methods
    # =====================

    def get_permutation_by_hash(self, width: int, perm_hash: str) -> Optional[WirePermutationRecord]:
        """Get permutation by width and hash."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM wire_permutations
                WHERE width = ? AND wire_perm_hash = ?
            """,
                (width, perm_hash),
            )
            row = cursor.fetchone()
            if row:
                return WirePermutationRecord(
                    id=row["id"],
                    width=row["width"],
                    wire_perm=json.loads(row["wire_perm"]),
                    wire_perm_hash=row["wire_perm_hash"],
                    fixed_points=row["fixed_points"],
                    hamming=row["hamming"],
                    cycles=row["cycles"],
                    swap_distance=row["swap_distance"],
                    cycle_type=row["cycle_type"],
                    parity=row["parity"],
                    is_identity=bool(row["is_identity"]),
                    created_at=row["created_at"],
                )
            return None

    def get_circuits_for_perm(self, perm_id: int) -> List[WireShufflerCircuit]:
        """Get all circuits for a permutation."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM wire_shuffler_circuits
                WHERE perm_id = ?
                ORDER BY gate_count ASC
            """,
                (perm_id,),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    WireShufflerCircuit(
                        id=row["id"],
                        run_id=row["run_id"],
                        perm_id=row["perm_id"],
                        found=bool(row["found"]),
                        gate_count=row["gate_count"],
                        gates=json.loads(row["gates"]) if row["gates"] else None,
                        circuit_hash=row["circuit_hash"],
                        full_perm=json.loads(row["full_perm"]) if row["full_perm"] else None,
                        verify_ok=bool(row["verify_ok"]) if row["verify_ok"] is not None else None,
                        synth_time_ms=row["synth_time_ms"],
                        is_best=bool(row["is_best"]),
                        created_at=row["created_at"],
                    )
                )
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            stats = {}

            # Permutation stats
            cursor = conn.execute("SELECT COUNT(*) as count FROM wire_permutations")
            stats["total_permutations"] = cursor.fetchone()["count"]

            # Circuit stats
            cursor = conn.execute("SELECT COUNT(*) as count FROM wire_shuffler_circuits WHERE found = 1")
            stats["total_circuits"] = cursor.fetchone()["count"]

            # By width
            cursor = conn.execute(
                """
                SELECT width, COUNT(*) as count
                FROM wire_permutations
                GROUP BY width
                ORDER BY width
            """
            )
            stats["permutations_by_width"] = {row["width"]: row["count"] for row in cursor.fetchall()}

            # Circuits by width
            cursor = conn.execute(
                """
                SELECT p.width, COUNT(*) as count
                FROM wire_shuffler_circuits c
                JOIN wire_permutations p ON c.perm_id = p.id
                WHERE c.found = 1
                GROUP BY p.width
                ORDER BY p.width
            """
            )
            stats["circuits_by_width"] = {row["width"]: row["count"] for row in cursor.fetchall()}

            return stats


# =====================
# Waksman Circuit Support
# =====================


@dataclass
class WaksmanCircuitRecord:
    """Record for a Waksman-generated circuit."""

    id: Optional[int]
    perm_id: int
    gate_count: int
    gates: List[Tuple[int, int, int]]
    swap_count: int
    synth_time_ms: Optional[int] = None
    verify_ok: Optional[bool] = None
    circuit_hash: Optional[str] = None
    created_at: Optional[str] = None


class WaksmanCircuitDatabase(WireShufflerDatabase):
    """Extended database with Waksman circuit support."""

    def _init_schema(self) -> None:
        """Initialize schema including Waksman tables."""
        # First init parent schema
        super()._init_schema()

        # Add Waksman-specific table
        with self._connect() as conn:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_waksman_perm ON waksman_circuits(perm_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_waksman_gatecount ON waksman_circuits(gate_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_waksman_swapcount ON waksman_circuits(swap_count)"
            )
            conn.commit()

    def insert_waksman_circuit(self, record: WaksmanCircuitRecord) -> int:
        """Insert a Waksman-generated circuit."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO waksman_circuits
                (perm_id, gate_count, gates, swap_count, synth_time_ms, verify_ok, circuit_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.perm_id,
                    record.gate_count,
                    json.dumps(record.gates),
                    record.swap_count,
                    record.synth_time_ms,
                    int(record.verify_ok) if record.verify_ok is not None else None,
                    record.circuit_hash,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_waksman_for_perm(self, perm_id: int) -> Optional[WaksmanCircuitRecord]:
        """Get Waksman circuit for a permutation."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM waksman_circuits
                WHERE perm_id = ?
                ORDER BY gate_count ASC
                LIMIT 1
            """,
                (perm_id,),
            )
            row = cursor.fetchone()
            if row:
                return WaksmanCircuitRecord(
                    id=row["id"],
                    perm_id=row["perm_id"],
                    gate_count=row["gate_count"],
                    gates=json.loads(row["gates"]),
                    swap_count=row["swap_count"],
                    synth_time_ms=row["synth_time_ms"],
                    verify_ok=bool(row["verify_ok"]) if row["verify_ok"] is not None else None,
                    circuit_hash=row["circuit_hash"],
                    created_at=row["created_at"],
                )
            return None

    def get_waksman_circuits(
        self, width: Optional[int] = None, limit: int = 100, offset: int = 0
    ) -> List[WaksmanCircuitRecord]:
        """Get Waksman circuits with optional filtering."""
        with self._connect() as conn:
            if width is not None:
                cursor = conn.execute(
                    """
                    SELECT w.* FROM waksman_circuits w
                    JOIN wire_permutations p ON w.perm_id = p.id
                    WHERE p.width = ?
                    ORDER BY w.gate_count ASC
                    LIMIT ? OFFSET ?
                """,
                    (width, limit, offset),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM waksman_circuits
                    ORDER BY gate_count ASC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

            results = []
            for row in cursor.fetchall():
                results.append(
                    WaksmanCircuitRecord(
                        id=row["id"],
                        perm_id=row["perm_id"],
                        gate_count=row["gate_count"],
                        gates=json.loads(row["gates"]),
                        swap_count=row["swap_count"],
                        synth_time_ms=row["synth_time_ms"],
                        verify_ok=bool(row["verify_ok"]) if row["verify_ok"] is not None else None,
                        circuit_hash=row["circuit_hash"],
                        created_at=row["created_at"],
                    )
                )
            return results

    def get_waksman_stats(self) -> Dict[str, Any]:
        """Get Waksman circuit statistics."""
        with self._connect() as conn:
            stats = {}

            # Total Waksman circuits
            cursor = conn.execute("SELECT COUNT(*) as count FROM waksman_circuits")
            stats["total_waksman_circuits"] = cursor.fetchone()["count"]

            # By width
            cursor = conn.execute(
                """
                SELECT p.width, COUNT(*) as count, AVG(w.gate_count) as avg_gates, AVG(w.swap_count) as avg_swaps
                FROM waksman_circuits w
                JOIN wire_permutations p ON w.perm_id = p.id
                GROUP BY p.width
                ORDER BY p.width
            """
            )
            stats["waksman_by_width"] = {
                row["width"]: {
                    "count": row["count"],
                    "avg_gates": round(row["avg_gates"], 2) if row["avg_gates"] else 0,
                    "avg_swaps": round(row["avg_swaps"], 2) if row["avg_swaps"] else 0,
                }
                for row in cursor.fetchall()
            }

            return stats

    def compare_sat_vs_waksman(self, perm_id: int) -> Dict[str, Any]:
        """Compare SAT-synthesized vs Waksman circuit for a permutation."""
        sat_circuits = self.get_circuits_for_perm(perm_id)
        waksman_circuit = self.get_waksman_for_perm(perm_id)

        result = {
            "perm_id": perm_id,
            "sat_available": len(sat_circuits) > 0,
            "waksman_available": waksman_circuit is not None,
            "sat_gate_count": None,
            "waksman_gate_count": None,
            "gate_count_diff": None,
        }

        if sat_circuits:
            found_circuits = [c for c in sat_circuits if c.found and c.gate_count is not None]
            if found_circuits:
                best_sat = min(found_circuits, key=lambda c: c.gate_count)
                result["sat_gate_count"] = best_sat.gate_count

        if waksman_circuit:
            result["waksman_gate_count"] = waksman_circuit.gate_count

        if result["sat_gate_count"] and result["waksman_gate_count"]:
            result["gate_count_diff"] = result["waksman_gate_count"] - result["sat_gate_count"]

        return result
