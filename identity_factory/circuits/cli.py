#!/usr/bin/env python3
"""
Circuit Index CLI - Search and manage circuit files across the codebase.

Usage:
    python -m identity_factory.circuits.cli index [--root PATH]
    python -m identity_factory.circuits.cli search --width 32 --gates 100-500
    python -m identity_factory.circuits.cli info PATH
    python -m identity_factory.circuits.cli convert PATH --format eca57
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .formats import (
    parse_gate_file,
    find_circuit_files,
    compute_circuit_hash,
    compute_circuit_hash_short,
    list_to_gate_string,
    gate_string_to_list,
    write_eca57_file,
    gates_to_json,
    parse_big_identities_line,
)

# Default paths
DEFAULT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # research-group
INDEX_DB_PATH = DEFAULT_ROOT / "circuit_index.db"


def init_index_db(db_path: Path) -> sqlite3.Connection:
    """Initialize or open the circuit index database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS circuit_index (
            id INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            width INTEGER NOT NULL,
            gate_count INTEGER NOT NULL,
            gate_string TEXT NOT NULL,
            source TEXT NOT NULL,
            file_path TEXT,
            is_identity BOOLEAN,
            permutation TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dims ON circuit_index(width, gate_count)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON circuit_index(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON circuit_index(hash)")
    conn.commit()
    return conn


def index_gate_files(conn: sqlite3.Connection, root: Path, verbose: bool = True) -> int:
    """Index all .gate and .eca57 files under root."""
    count = 0
    for path in find_circuit_files(root):
        try:
            width, gates = parse_gate_file(path)
            gate_string = list_to_gate_string(gates)
            hash_short = compute_circuit_hash_short(gates)

            rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            source = f"file:{rel_path}"

            conn.execute("""
                INSERT OR REPLACE INTO circuit_index
                (hash, width, gate_count, gate_string, source, file_path, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (hash_short, width, len(gates), gate_string, source, str(path), datetime.now()))

            count += 1
            if verbose and count % 100 == 0:
                print(f"  Indexed {count} files...")

        except Exception as e:
            if verbose:
                print(f"  Warning: {path}: {e}", file=sys.stderr)

    conn.commit()
    return count


def index_big_identities(conn: sqlite3.Connection, root: Path, verbose: bool = True) -> int:
    """Index big_identities.txt files."""
    count = 0
    for txt_path in root.rglob("*identities*.txt"):
        try:
            with open(txt_path) as f:
                for line_num, line in enumerate(f, 1):
                    result = parse_big_identities_line(line)
                    if result is None:
                        continue

                    hash_short = compute_circuit_hash_short(result["gates"])
                    source = f"big_identities:{txt_path.name}:{line_num}"

                    conn.execute("""
                        INSERT OR REPLACE INTO circuit_index
                        (hash, width, gate_count, gate_string, source, file_path, is_identity, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (hash_short, result["width"], result["gate_count"],
                          result["gate_string"], source, str(txt_path), True, datetime.now()))

                    count += 1
                    if verbose and count % 1000 == 0:
                        print(f"  Indexed {count} identities...")

        except Exception as e:
            if verbose:
                print(f"  Warning: {txt_path}: {e}", file=sys.stderr)

    conn.commit()
    return count


def index_go_json(conn: sqlite3.Connection, root: Path, verbose: bool = True) -> int:
    """Index Go-exported JSON circuit files."""
    count = 0
    go_db_path = root / "obfuscated-circuits" / "go-proj" / "db"

    if not go_db_path.exists():
        return 0

    for json_path in go_db_path.glob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)

            n = data.get("n", 0)
            m = data.get("m", 0)

            for entry_idx, entry in enumerate(data.get("entries", [])):
                perm = entry.get("perm", [])
                perm_str = json.dumps(perm)

                for circuit_idx, circuit in enumerate(entry.get("circuits", [])):
                    gates = [(g[0], g[1], g[2]) for g in circuit]
                    gate_string = list_to_gate_string(gates)
                    hash_short = compute_circuit_hash_short(gates)
                    source = f"go-json:{json_path.stem}:{entry_idx}:{circuit_idx}"

                    conn.execute("""
                        INSERT OR REPLACE INTO circuit_index
                        (hash, width, gate_count, gate_string, source, file_path, permutation, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (hash_short, n, len(gates), gate_string, source, str(json_path), perm_str, datetime.now()))

                    count += 1

        except Exception as e:
            if verbose:
                print(f"  Warning: {json_path}: {e}", file=sys.stderr)

    conn.commit()
    return count


def cmd_index(args):
    """Build or update the circuit index."""
    root = Path(args.root).resolve()
    db_path = Path(args.db) if args.db else INDEX_DB_PATH

    print(f"Indexing circuits under: {root}")
    print(f"Database: {db_path}")

    conn = init_index_db(db_path)

    if args.clear:
        conn.execute("DELETE FROM circuit_index")
        conn.commit()
        print("Cleared existing index.")

    total = 0

    print("\nIndexing .gate/.eca57 files...")
    count = index_gate_files(conn, root, verbose=not args.quiet)
    print(f"  -> {count} files indexed")
    total += count

    print("\nIndexing big_identities.txt...")
    count = index_big_identities(conn, root, verbose=not args.quiet)
    print(f"  -> {count} identities indexed")
    total += count

    print("\nIndexing Go JSON exports...")
    count = index_go_json(conn, root, verbose=not args.quiet)
    print(f"  -> {count} circuits indexed")
    total += count

    # Summary
    cursor = conn.execute("SELECT COUNT(*), COUNT(DISTINCT hash) FROM circuit_index")
    total_rows, unique_hashes = cursor.fetchone()

    cursor = conn.execute("SELECT width, COUNT(*) FROM circuit_index GROUP BY width ORDER BY width")
    by_width = cursor.fetchall()

    print(f"\n{'='*50}")
    print(f"Total indexed: {total_rows} ({unique_hashes} unique)")
    print(f"\nBy width:")
    for width, count in by_width:
        print(f"  {width}-wire: {count}")

    conn.close()


def cmd_search(args):
    """Search the circuit index."""
    db_path = Path(args.db) if args.db else INDEX_DB_PATH

    if not db_path.exists():
        print(f"Index not found: {db_path}", file=sys.stderr)
        print("Run 'index' command first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Build query
    conditions = []
    params = []

    if args.width:
        conditions.append("width = ?")
        params.append(args.width)

    if args.min_gates:
        conditions.append("gate_count >= ?")
        params.append(args.min_gates)

    if args.max_gates:
        conditions.append("gate_count <= ?")
        params.append(args.max_gates)

    if args.source:
        conditions.append("source LIKE ?")
        params.append(f"%{args.source}%")

    if args.hash:
        conditions.append("hash LIKE ?")
        params.append(f"{args.hash}%")

    if args.identity:
        conditions.append("is_identity = 1")

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT * FROM circuit_index
        WHERE {where}
        ORDER BY width, gate_count
        LIMIT ?
    """
    params.append(args.limit)

    cursor = conn.execute(query, params)
    results = cursor.fetchall()

    if args.json:
        output = []
        for row in results:
            output.append({
                "hash": row["hash"],
                "width": row["width"],
                "gate_count": row["gate_count"],
                "source": row["source"],
                "file_path": row["file_path"],
                "is_identity": bool(row["is_identity"]),
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"Found {len(results)} circuits:\n")
        for row in results:
            identity_mark = " [ID]" if row["is_identity"] else ""
            print(f"{row['hash'][:8]}  {row['width']:2}w  {row['gate_count']:4}g  {row['source']}{identity_mark}")
            if args.verbose and row["file_path"]:
                print(f"         -> {row['file_path']}")

    conn.close()


def cmd_info(args):
    """Show detailed info about a circuit file."""
    path = Path(args.path)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    width, gates = parse_gate_file(path)
    gate_string = list_to_gate_string(gates)
    hash_full = compute_circuit_hash(gates)

    print(f"File: {path}")
    print(f"Width: {width}")
    print(f"Gates: {len(gates)}")
    print(f"Hash: {hash_full}")
    print(f"\nGate string ({len(gate_string)} chars):")

    if len(gate_string) > 200:
        print(f"  {gate_string[:100]}...{gate_string[-100:]}")
    else:
        print(f"  {gate_string}")

    if args.full:
        print(f"\nGates (target, ctrl1, ctrl2):")
        for i, (t, c1, c2) in enumerate(gates):
            print(f"  {i:4}: ({t}, {c1}, {c2})")

    if args.json:
        print(f"\nJSON:")
        print(json.dumps(gates_to_json(width, gates, source=str(path)), indent=2))


def cmd_convert(args):
    """Convert a circuit file to another format."""
    path = Path(args.path)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    width, gates = parse_gate_file(path)

    if args.format == "eca57":
        out_path = path.with_suffix(".eca57")
        write_eca57_file(out_path, width, gates, source=str(path))
        print(f"Written: {out_path}")

    elif args.format == "json":
        out_path = path.with_suffix(".json")
        with open(out_path, "w") as f:
            json.dump(gates_to_json(width, gates, source=str(path)), f, indent=2)
        print(f"Written: {out_path}")

    elif args.format == "gate":
        out_path = path.with_suffix(".gate")
        out_path.write_text(list_to_gate_string(gates))
        print(f"Written: {out_path}")

    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)


def cmd_stats(args):
    """Show index statistics."""
    db_path = Path(args.db) if args.db else INDEX_DB_PATH

    if not db_path.exists():
        print(f"Index not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute("SELECT COUNT(*), COUNT(DISTINCT hash) FROM circuit_index")
    total, unique = cursor.fetchone()

    cursor = conn.execute("""
        SELECT width, gate_count, COUNT(*) as count
        FROM circuit_index
        GROUP BY width, gate_count
        ORDER BY width, gate_count
    """)
    dims = cursor.fetchall()

    cursor = conn.execute("""
        SELECT
            CASE
                WHEN source LIKE 'file:%' THEN 'file'
                WHEN source LIKE 'big_identities:%' THEN 'big_identities'
                WHEN source LIKE 'go-json:%' THEN 'go-json'
                ELSE 'other'
            END as src_type,
            COUNT(*)
        FROM circuit_index
        GROUP BY src_type
    """)
    sources = cursor.fetchall()

    print(f"Circuit Index Statistics")
    print(f"{'='*40}")
    print(f"Database: {db_path}")
    print(f"Total entries: {total}")
    print(f"Unique circuits: {unique}")
    print(f"Duplicates: {total - unique}")

    print(f"\nBy source:")
    for src, count in sources:
        print(f"  {src}: {count}")

    print(f"\nBy dimensions (width x gates):")
    current_width = None
    for width, gate_count, count in dims:
        if width != current_width:
            print(f"\n  {width}-wire:")
            current_width = width
        print(f"    {gate_count} gates: {count}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Circuit Index CLI - Search and manage circuit files"
    )
    parser.add_argument("--db", help="Path to index database")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    p_index = subparsers.add_parser("index", help="Build/update circuit index")
    p_index.add_argument("--root", default=str(DEFAULT_ROOT), help="Root directory to scan")
    p_index.add_argument("--clear", action="store_true", help="Clear existing index first")
    p_index.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_index.set_defaults(func=cmd_index)

    # search command
    p_search = subparsers.add_parser("search", help="Search indexed circuits")
    p_search.add_argument("--width", "-w", type=int, help="Filter by wire count")
    p_search.add_argument("--min-gates", type=int, help="Minimum gate count")
    p_search.add_argument("--max-gates", type=int, help="Maximum gate count")
    p_search.add_argument("--source", "-s", help="Filter by source (substring match)")
    p_search.add_argument("--hash", help="Filter by hash prefix")
    p_search.add_argument("--identity", "-i", action="store_true", help="Only identity circuits")
    p_search.add_argument("--limit", "-n", type=int, default=50, help="Max results")
    p_search.add_argument("--json", "-j", action="store_true", help="JSON output")
    p_search.add_argument("--verbose", "-v", action="store_true", help="Show file paths")
    p_search.set_defaults(func=cmd_search)

    # info command
    p_info = subparsers.add_parser("info", help="Show circuit file info")
    p_info.add_argument("path", help="Path to circuit file")
    p_info.add_argument("--full", "-f", action="store_true", help="Show all gates")
    p_info.add_argument("--json", "-j", action="store_true", help="Show JSON format")
    p_info.set_defaults(func=cmd_info)

    # convert command
    p_convert = subparsers.add_parser("convert", help="Convert circuit format")
    p_convert.add_argument("path", help="Path to circuit file")
    p_convert.add_argument("--format", "-f", required=True,
                          choices=["eca57", "json", "gate"], help="Output format")
    p_convert.set_defaults(func=cmd_convert)

    # stats command
    p_stats = subparsers.add_parser("stats", help="Show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
