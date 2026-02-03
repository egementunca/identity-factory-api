#!/usr/bin/env python3
"""
Scan cluster_circuits.db for reducible circuits (consecutive identical gates).

This script:
1. Scans all circuits in the database
2. Identifies those with consecutive identical gates
3. Reports statistics by width/gate_count
4. Optionally adds a 'has_reducible_pairs' column to flag them
"""

import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict


def has_reducible_pairs(gates_text: str) -> bool:
    """Check if circuit has consecutive identical gates (same target and controls).

    Such circuits are "reducible" because G·G = I for these reversible gates.
    """
    if not gates_text:
        return False

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
                return True  # Found consecutive identical gates

            prev_gate = normalized
        except (ValueError, IndexError):
            continue

    return False


def scan_database(db_path: Path, add_column: bool = False, verbose: bool = False):
    """Scan database and report on reducible circuits."""

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all circuits
    cursor.execute("SELECT id, width, gate_count, gates FROM circuits")

    total = 0
    reducible_count = 0
    reducible_by_dim = defaultdict(int)
    total_by_dim = defaultdict(int)
    reducible_examples = []

    for row in cursor.fetchall():
        circuit_id, width, gate_count, gates = row
        total += 1
        dim_key = f"{width}w_{gate_count}g"
        total_by_dim[dim_key] += 1

        if has_reducible_pairs(gates):
            reducible_count += 1
            reducible_by_dim[dim_key] += 1

            if verbose and len(reducible_examples) < 10:
                reducible_examples.append({
                    'id': circuit_id,
                    'width': width,
                    'gate_count': gate_count,
                    'gates': gates
                })

    # Print summary
    print("=" * 60)
    print("REDUCIBLE CIRCUITS SCAN REPORT")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")
    print(f"Total circuits: {total:,}")
    print(f"Reducible circuits: {reducible_count:,} ({100*reducible_count/total:.1f}%)")
    print(f"Clean circuits: {total - reducible_count:,} ({100*(total-reducible_count)/total:.1f}%)")

    print("\n" + "-" * 60)
    print("BREAKDOWN BY DIMENSION (width × gates)")
    print("-" * 60)
    print(f"{'Dimension':<15} {'Total':>10} {'Reducible':>12} {'%':>8}")
    print("-" * 60)

    for dim_key in sorted(total_by_dim.keys()):
        t = total_by_dim[dim_key]
        r = reducible_by_dim.get(dim_key, 0)
        pct = 100 * r / t if t > 0 else 0
        print(f"{dim_key:<15} {t:>10,} {r:>12,} {pct:>7.1f}%")

    if verbose and reducible_examples:
        print("\n" + "-" * 60)
        print("EXAMPLE REDUCIBLE CIRCUITS (first 10)")
        print("-" * 60)
        for ex in reducible_examples:
            print(f"\nID: {ex['id']} ({ex['width']}w × {ex['gate_count']}g)")
            print(f"Gates: {ex['gates']}")
            # Show which gates are duplicated
            gates = ex['gates'].split(";")
            for i in range(len(gates) - 1):
                if gates[i] and gates[i+1]:
                    if gates[i] == gates[i+1]:
                        print(f"  ^^ Gate {i} and {i+1} are identical: {gates[i]}")

    # Optionally add column to database
    if add_column:
        print("\n" + "-" * 60)
        print("ADDING has_reducible_pairs COLUMN")
        print("-" * 60)

        try:
            cursor.execute("ALTER TABLE circuits ADD COLUMN has_reducible_pairs INTEGER DEFAULT 0")
            print("Added column 'has_reducible_pairs' to circuits table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("Column 'has_reducible_pairs' already exists")
            else:
                print(f"Error adding column: {e}")
                conn.close()
                return

        # Update all circuits
        print("Updating reducible circuit flags...")
        cursor.execute("SELECT id, gates FROM circuits")

        updated = 0
        for row in cursor.fetchall():
            circuit_id, gates = row
            flag = 1 if has_reducible_pairs(gates) else 0
            cursor.execute(
                "UPDATE circuits SET has_reducible_pairs = ? WHERE id = ?",
                (flag, circuit_id)
            )
            updated += 1
            if updated % 1000 == 0:
                print(f"  Updated {updated:,} circuits...")

        conn.commit()
        print(f"Done! Updated {updated:,} circuits")

        # Verify
        cursor.execute("SELECT COUNT(*) FROM circuits WHERE has_reducible_pairs = 1")
        flagged = cursor.fetchone()[0]
        print(f"Verified: {flagged:,} circuits flagged as reducible")

    conn.close()
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Scan cluster_circuits.db for reducible circuits"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent / "cluster_circuits.db",
        help="Path to database"
    )
    parser.add_argument(
        "--add-column",
        action="store_true",
        help="Add has_reducible_pairs column and update flags"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show example reducible circuits"
    )

    args = parser.parse_args()
    scan_database(args.db, args.add_column, args.verbose)


if __name__ == "__main__":
    main()
