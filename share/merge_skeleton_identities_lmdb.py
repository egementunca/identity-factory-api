#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lmdb


def deserialize_vec_vec_u8(data: bytes) -> List[bytes]:
    if len(data) < 8:
        return []
    count = struct.unpack("<Q", data[:8])[0]
    offset = 8
    out: List[bytes] = []
    for _ in range(count):
        if offset + 8 > len(data):
            break
        blob_len = struct.unpack("<Q", data[offset : offset + 8])[0]
        offset += 8
        blob = data[offset : offset + blob_len]
        offset += blob_len
        out.append(blob)
    return out


def serialize_vec_vec_u8(blobs: List[bytes]) -> bytes:
    buf = bytearray()
    buf.extend(struct.pack("<Q", len(blobs)))
    for blob in blobs:
        buf.extend(struct.pack("<Q", len(blob)))
        buf.extend(blob)
    return bytes(buf)


def merge_blobs(
    existing: List[bytes],
    incoming: List[bytes],
    max_circuits_per_key: Optional[int],
) -> Tuple[List[bytes], int]:
    if not incoming:
        return existing, 0

    existing_set = set(existing)
    added = 0
    merged = list(existing)
    for blob in incoming:
        if blob in existing_set:
            continue
        merged.append(blob)
        existing_set.add(blob)
        added += 1

        if max_circuits_per_key is not None and len(merged) >= max_circuits_per_key:
            break

    return merged, added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge skeleton identity LMDB(s) (ids_n*) into a target LMDB environment."
    )
    parser.add_argument("--src", required=True, help="Source LMDB directory (e.g., share/skeleton_ids_n4_n7.lmdb)")
    parser.add_argument("--dst", required=True, help="Target LMDB directory (e.g., local_mixing/db)")
    parser.add_argument(
        "--db-names",
        nargs="+",
        default=["ids_n4", "ids_n5", "ids_n6", "ids_n7"],
        help="Named DBs to merge",
    )
    parser.add_argument(
        "--max-circuits-per-key",
        type=int,
        default=None,
        help="Cap circuits per taxonomy key (default: no cap)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute counts but do not write")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source LMDB not found: {src_path}")
    if not dst_path.exists():
        raise FileNotFoundError(f"Target LMDB not found: {dst_path}")

    src_env = lmdb.open(str(src_path), readonly=True, max_dbs=200, lock=False)
    dst_env = lmdb.open(str(dst_path), map_size=1 << 40, max_dbs=200)  # map_size ignored for existing env

    totals: Dict[str, Dict[str, int]] = {}

    for db_name in args.db_names:
        totals[db_name] = {"keys": 0, "added": 0, "skipped": 0}
        src_db = src_env.open_db(db_name.encode())
        dst_db = dst_env.open_db(db_name.encode(), create=True)

        with src_env.begin(db=src_db) as src_txn, dst_env.begin(
            write=not args.dry_run, db=dst_db
        ) as dst_txn:
            cursor = src_txn.cursor()
            for key, val in cursor:
                totals[db_name]["keys"] += 1

                incoming = deserialize_vec_vec_u8(val)
                existing_raw = dst_txn.get(key)
                existing = deserialize_vec_vec_u8(existing_raw) if existing_raw else []

                merged, added = merge_blobs(
                    existing, incoming, max_circuits_per_key=args.max_circuits_per_key
                )
                skipped = max(0, len(incoming) - added)

                totals[db_name]["added"] += added
                totals[db_name]["skipped"] += skipped

                if not args.dry_run and added > 0:
                    dst_txn.put(key, serialize_vec_vec_u8(merged))

    if not args.dry_run:
        dst_env.sync()
    dst_env.close()
    src_env.close()

    for db_name, info in totals.items():
        print(
            f"{db_name}: keys={info['keys']} circuits_added={info['added']} circuits_skipped={info['skipped']}"
        )

    if args.dry_run:
        print("(dry-run: no changes written)")


if __name__ == "__main__":
    main()
