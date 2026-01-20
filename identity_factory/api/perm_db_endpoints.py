"""
Permutation Database API endpoints.
Provides read-only access to the local_mixing LMDB permutation tables.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/perm-database", tags=["perm-database"])

# Path to local_mixing/db
# Assumes the API is running from identity-factory-api/
DB_PATH = Path(__file__).parent.parent.parent.parent / "local_mixing" / "db"


class PermTableStats(BaseModel):
    name: str
    entries: int
    key_size: int
    n_wires: int


class PermDatabaseStats(BaseModel):
    tables: List[PermTableStats]
    total_entries: int
    path: str
    status: str


class PermEntry(BaseModel):
    permutation: List[int]
    gate_counts: List[int]
    hex_key: str  # For easier display/copying


class PermEntriesResponse(BaseModel):
    entries: List[PermEntry]
    limit: int
    offset: int
    has_more: bool


def get_lmdb_env():
    if not DB_PATH.exists():
        return None
    try:
        # Open in read-only mode, no lock to avoid interfering with backend
        return lmdb.open(str(DB_PATH), max_dbs=50, readonly=True, create=False, lock=False)
    except lmdb.Error as e:
        print(f"Error opening LMDB: {e}")
        return None


@router.get("/stats", response_model=PermDatabaseStats)
async def get_stats():
    """Get statistics for all permutation tables."""
    env = get_lmdb_env()
    if not env:
        return PermDatabaseStats(
            tables=[], total_entries=0, path=str(DB_PATH), status="not_found"
        )

    tables = []
    total_entries = 0

    try:
        with env.begin() as txn:
            # We check for known table names n3..n8
            for n in range(3, 9):
                name = f"perm_tables_n{n}"
                try:
                    db = env.open_db(name.encode(), txn=txn, create=False)
                    stat = txn.stat(db)
                    entries = stat["entries"]
                    tables.append(
                        PermTableStats(
                            name=name,
                            entries=entries,
                            key_size=1 << n,  # 2^n bytes
                            n_wires=n,
                        )
                    )
                    total_entries += entries
                except lmdb.Error:
                    continue  # Table doesn't exist
    except Exception as e:
        return PermDatabaseStats(
            tables=[], total_entries=0, path=str(DB_PATH), status=f"error: {str(e)}"
        )
    finally:
        env.close()

    return PermDatabaseStats(
        tables=tables,
        total_entries=total_entries,
        path=str(DB_PATH),
        status="active",
    )


@router.get("/{table_name}/entries", response_model=PermEntriesResponse)
async def list_entries(
    table_name: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List entries from a specific permutation table."""
    env = get_lmdb_env()
    if not env:
        raise HTTPException(status_code=404, detail="Database not found")

    entries = []
    has_more = False

    try:
        with env.begin() as txn:
            try:
                db = env.open_db(table_name.encode(), txn=txn, create=False)
            except lmdb.Error:
                raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

            cursor = txn.cursor(db)
            
            # Move to offset
            # LMDB doesn't support random access by index, so we must scan.
            # For large offsets, this is slow. 
            # Ideally we'd use a key to seek, but we don't have one here.
            # We'll limit offset in UI or accept it's slow.
            
            if not cursor.first():
                 return PermEntriesResponse(entries=[], limit=limit, offset=offset, has_more=False)

            # Skip 'offset' records
            # For header skip, we can iterate. 
            # Note: For very large offsets, this API will effectively timeout. 
            # We instruct user to assume linear scan.
            current_idx = 0
            while current_idx < offset:
                if not cursor.next():
                    return PermEntriesResponse(entries=[], limit=limit, offset=offset, has_more=False)
                current_idx += 1

            # Read 'limit' records
            while len(entries) < limit:
                k, v = cursor.item()
                
                # Parse Key (Permutation)
                # It's raw bytes (u8)
                perm = list(k)
                
                # Parse Value (Gate Counts)
                # Bincode: [u64 len] [u8...]
                gate_counts = []
                if len(v) >= 8:
                    try:
                        vec_len = struct.unpack('<Q', v[:8])[0]
                        if len(v) == 8 + vec_len:
                            gate_counts = list(v[8:])
                    except:
                        pass # Parsing failed
                
                entries.append(PermEntry(
                    permutation=perm,
                    gate_counts=gate_counts,
                    hex_key=k.hex()
                ))

                if not cursor.next():
                    break
            
            has_more = cursor.next() # Check if one more exists

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        env.close()

    return PermEntriesResponse(
        entries=entries,
        limit=limit,
        offset=offset,
        has_more=has_more
    )
