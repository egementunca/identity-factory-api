#!/usr/bin/env python3
"""
Import script for big_identities.txt file.

Parses the ECA57 identity circuit format and imports into the identity-factory database.

Format: table=ids_n16g0, wires=16, gates=162, circuit=012;47d;c74;...

Each gate is 3 characters (active, ctrl1, ctrl2) using the wire map:
0-9 = wires 0-9
a-z = wires 10-35
A-Z = wires 36-61
etc.
"""

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wire character mapping (same as Go code)
WIRE_MAP_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"


def char_to_wire(c: str) -> int:
    """Convert a character to a wire index."""
    idx = WIRE_MAP_CHARS.find(c)
    if idx < 0:
        raise ValueError(f"Unknown wire character: {c}")
    return idx


def parse_circuit_string(circuit_str: str) -> List[Tuple[int, int, int]]:
    """
    Parse a circuit string into a list of gate tuples.

    Each gate is (active, ctrl1, ctrl2) representing an ECA57 gate.
    """
    gates = []
    for gate_str in circuit_str.split(";"):
        if not gate_str or len(gate_str) != 3:
            continue
        active = char_to_wire(gate_str[0])
        ctrl1 = char_to_wire(gate_str[1])
        ctrl2 = char_to_wire(gate_str[2])
        gates.append((active, ctrl1, ctrl2))
    return gates


def compute_eca57_permutation(gates: List[Tuple[int, int, int]], num_wires: int) -> List[int]:
    """
    Compute the permutation implemented by an ECA57 circuit.

    Each gate [a, c1, c2] applies: if c1=1 and c2=0, flip wire a.
    This is the NIMPLY function (Rule 57).
    """
    size = 1 << num_wires
    perm = list(range(size))

    for input_val in range(size):
        output_val = input_val
        for active, c1, c2 in gates:
            bit_c1 = (output_val >> c1) & 1
            bit_c2 = (output_val >> c2) & 1
            if bit_c1 == 1 and bit_c2 == 0:
                output_val ^= 1 << active
        perm[input_val] = output_val

    return perm


def is_identity_permutation(perm: List[int]) -> bool:
    """Check if a permutation is the identity."""
    return perm == list(range(len(perm)))


@dataclass
class ParsedCircuit:
    """A parsed circuit from the file."""
    table_name: str
    wires: int
    gate_count: int
    gates: List[Tuple[int, int, int]]
    circuit_str: str


def parse_line(line: str) -> Optional[ParsedCircuit]:
    """Parse a single line from the identities file."""
    line = line.strip()
    if not line:
        return None

    # Parse: table=ids_n16g0, wires=16, gates=162, circuit=...
    match = re.match(
        r"table=(\w+),\s*wires=(\d+),\s*gates=(\d+),\s*circuit=(.+)",
        line
    )
    if not match:
        logger.warning(f"Failed to parse line: {line[:100]}...")
        return None

    table_name = match.group(1)
    wires = int(match.group(2))
    gate_count = int(match.group(3))
    circuit_str = match.group(4)

    gates = parse_circuit_string(circuit_str)

    if len(gates) != gate_count:
        logger.warning(f"Gate count mismatch: expected {gate_count}, got {len(gates)}")

    return ParsedCircuit(
        table_name=table_name,
        wires=wires,
        gate_count=len(gates),
        gates=gates,
        circuit_str=circuit_str,
    )


def stream_circuits(file_path: Path) -> Generator[ParsedCircuit, None, None]:
    """Stream parsed circuits from a file."""
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                logger.info(f"Processing line {line_num}...")

            parsed = parse_line(line)
            if parsed:
                yield parsed


def compute_circuit_hash(gates: List[Tuple[int, int, int]], wires: int) -> str:
    """Compute a hash for the circuit."""
    gates_str = ";".join(f"{a}{c1}{c2}" for a, c1, c2 in gates)
    combined = f"w{wires}:{gates_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def gates_to_nct_format(gates: List[Tuple[int, int, int]]) -> List[List]:
    """
    Convert ECA57 gates to NCT format for database storage.

    ECA57 gate (a, c1, c2): if c1=1 and c2=0, flip a
    This is similar to a Toffoli but with NIMPLY control function.
    We store as ["ECA57", c1, c2, a] to distinguish from CCX.
    """
    return [["ECA57", c1, c2, a] for a, c1, c2 in gates]


class BigIdentitiesImporter:
    """Import big_identities.txt into the identity factory database."""

    def __init__(self, db_path: str, verify_identity: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.verify_identity = verify_identity
        self._init_database()
        self.stats = {
            "total_parsed": 0,
            "imported": 0,
            "duplicates": 0,
            "invalid": 0,
            "by_wires": {},
            "by_gate_count": {},
        }

    def _init_database(self):
        """Initialize database with additional tables for imported identities."""
        with sqlite3.connect(self.db_path) as conn:
            # Create table for imported identity circuits
            conn.execute("""
                CREATE TABLE IF NOT EXISTS imported_identities (
                    id INTEGER PRIMARY KEY,
                    source_table TEXT NOT NULL,
                    wires INTEGER NOT NULL,
                    gate_count INTEGER NOT NULL,
                    gates TEXT NOT NULL,
                    circuit_str TEXT NOT NULL,
                    circuit_hash TEXT UNIQUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_imported_wires ON imported_identities(wires)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_imported_gates ON imported_identities(gate_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_imported_hash ON imported_identities(circuit_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_imported_source ON imported_identities(source_table)"
            )

            # Create metrics view
            conn.execute("""
                CREATE VIEW IF NOT EXISTS imported_identity_stats AS
                SELECT
                    source_table,
                    wires,
                    COUNT(*) as circuit_count,
                    MIN(gate_count) as min_gates,
                    MAX(gate_count) as max_gates,
                    AVG(gate_count) as avg_gates
                FROM imported_identities
                GROUP BY source_table, wires
                ORDER BY wires, source_table
            """)

            conn.commit()

    def import_circuit(self, circuit: ParsedCircuit, conn: sqlite3.Connection) -> bool:
        """Import a single circuit."""
        circuit_hash = compute_circuit_hash(circuit.gates, circuit.wires)

        # Convert gates to JSON
        gates_json = json.dumps(circuit.gates)

        # Verify if requested (expensive for large circuits)
        is_verified = False
        if self.verify_identity and circuit.wires <= 16:
            try:
                perm = compute_eca57_permutation(circuit.gates, circuit.wires)
                is_verified = is_identity_permutation(perm)
                if not is_verified:
                    self.stats["invalid"] += 1
                    logger.warning(f"Circuit is not identity: {circuit.table_name}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to verify circuit: {e}")

        try:
            conn.execute("""
                INSERT INTO imported_identities
                (source_table, wires, gate_count, gates, circuit_str, circuit_hash, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                circuit.table_name,
                circuit.wires,
                circuit.gate_count,
                gates_json,
                circuit.circuit_str,
                circuit_hash,
                is_verified,
            ))

            self.stats["imported"] += 1

            # Track stats
            wire_key = str(circuit.wires)
            self.stats["by_wires"][wire_key] = self.stats["by_wires"].get(wire_key, 0) + 1

            gate_key = str(circuit.gate_count)
            self.stats["by_gate_count"][gate_key] = self.stats["by_gate_count"].get(gate_key, 0) + 1

            return True

        except sqlite3.IntegrityError:
            self.stats["duplicates"] += 1
            return False

    def import_file(self, file_path: Path, batch_size: int = 1000):
        """Import all circuits from a file."""
        logger.info(f"Starting import from {file_path}")
        start_time = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            batch = []

            for circuit in stream_circuits(file_path):
                self.stats["total_parsed"] += 1
                self.import_circuit(circuit, conn)

                if self.stats["total_parsed"] % batch_size == 0:
                    conn.commit()
                    logger.info(
                        f"Progress: {self.stats['total_parsed']} parsed, "
                        f"{self.stats['imported']} imported, "
                        f"{self.stats['duplicates']} duplicates"
                    )

            conn.commit()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Import completed in {elapsed:.2f} seconds")
        self._print_stats()

    def _print_stats(self):
        """Print import statistics."""
        print("\n" + "=" * 60)
        print("IMPORT STATISTICS")
        print("=" * 60)
        print(f"Total parsed:  {self.stats['total_parsed']}")
        print(f"Imported:      {self.stats['imported']}")
        print(f"Duplicates:    {self.stats['duplicates']}")
        print(f"Invalid:       {self.stats['invalid']}")
        print()
        print("By wire count:")
        for wires, count in sorted(self.stats["by_wires"].items(), key=lambda x: int(x[0])):
            print(f"  {wires} wires: {count} circuits")
        print()
        print("Top gate counts:")
        sorted_gates = sorted(
            self.stats["by_gate_count"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for gate_count, count in sorted_gates:
            print(f"  {gate_count} gates: {count} circuits")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM imported_identities")
            total = cursor.fetchone()[0]

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
                {
                    "wires": row[0],
                    "count": row[1],
                    "min_gates": row[2],
                    "max_gates": row[3],
                    "avg_gates": round(row[4], 2) if row[4] else 0,
                }
                for row in cursor.fetchall()
            ]

            cursor = conn.execute("""
                SELECT source_table, COUNT(*)
                FROM imported_identities
                GROUP BY source_table
            """)
            by_source = dict(cursor.fetchall())

            return {
                "total_circuits": total,
                "by_wires": by_wires,
                "by_source": by_source,
            }


def main():
    parser = argparse.ArgumentParser(description="Import big_identities.txt into database")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to big_identities.txt",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / ".identity_factory" / "imported_identities.db",
        help="Database path",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify each circuit computes identity (slow for large circuits)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for commits",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show database statistics, don't import",
    )

    args = parser.parse_args()

    importer = BigIdentitiesImporter(str(args.db_path), verify_identity=args.verify)

    if args.stats_only:
        stats = importer.get_database_stats()
        print("\n" + "=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total circuits: {stats['total_circuits']}")
        print()
        print("By wire count:")
        for entry in stats["by_wires"]:
            print(f"  {entry['wires']} wires: {entry['count']} circuits "
                  f"(gates: {entry['min_gates']}-{entry['max_gates']}, avg: {entry['avg_gates']})")
        print()
        print("By source table:")
        for source, count in stats["by_source"].items():
            print(f"  {source}: {count}")
        return

    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    importer.import_file(args.input_file, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
