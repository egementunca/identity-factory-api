#!/usr/bin/env python3
"""Import swap-flip gadgets from JSON into the circuits database.

This makes the gadgets visible in Playground Pro.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def compute_circuit_hash(gates: list, width: int, flip_name: str) -> str:
    """Compute unique hash for a circuit."""
    data = f"{width}:{flip_name}:{json.dumps(gates, sort_keys=True)}"
    return hashlib.sha256(data.encode()).hexdigest()[:32]


def import_gadgets(json_path: str, db_path: str, verbose: bool = False):
    """Import swap-flip gadgets into the circuits database."""
    with open(json_path) as f:
        data = json.load(f)

    circuits = data.get("circuits", [])
    width = data.get("width", 3)

    print(f"Loading {len(circuits)} circuits from {json_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    imported = 0
    skipped = 0

    for circuit in circuits:
        gates = circuit["gates"]
        gate_count = circuit["gate_count"]
        flip_name = circuit.get("flip_name", "unknown")
        flip_mask = circuit.get("flip_mask", [0, 0])

        # Encode permutation with flip info: [1, 0, 2] for swap, flip_mask as metadata
        # Using the flip_name in the permutation string for identification
        permutation = json.dumps([1, 0, 2])  # Standard swap permutation

        # Create a unique hash including flip info
        circuit_hash = compute_circuit_hash(gates, width, flip_name)

        # Store gates as JSON
        gates_json = json.dumps(gates)

        # Add source and metadata
        source = f"swap_flip_{flip_name}"

        try:
            cursor.execute("""
                INSERT INTO circuits (width, gate_count, gates, permutation, circuit_hash, source, gate_set)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (width, gate_count, gates_json, permutation, circuit_hash, source, "eca57"))
            imported += 1

            if verbose:
                print(f"  Imported: {flip_name} @ {gate_count}g")

        except sqlite3.IntegrityError:
            # Duplicate hash
            skipped += 1
            if verbose:
                print(f"  Skipped (duplicate): {flip_name} @ {gate_count}g")

    conn.commit()
    conn.close()

    print(f"\nImported: {imported}")
    print(f"Skipped (duplicates): {skipped}")

    return imported, skipped


def main():
    parser = argparse.ArgumentParser(description="Import swap-flip gadgets into database")
    parser.add_argument(
        "--json", "-j",
        default="../share/swap_flip_gadgets.json",
        help="Path to swap_flip_gadgets.json"
    )
    parser.add_argument(
        "--db", "-d",
        default="cluster_circuits.db",
        help="Database file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = Path(__file__).parent.parent / args.json

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / args.db

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1

    import_gadgets(str(json_path), str(db_path), args.verbose)
    return 0


if __name__ == "__main__":
    exit(main())
