"""
Database status endpoint.

Provides a lightweight inventory of expected databases and their availability.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class DbRecord(BaseModel):
    path: str
    exists: bool
    type: str
    size_mb: Optional[float] = None
    tables: Optional[Dict[str, int]] = None
    table_count: Optional[int] = None
    files: Optional[List[str]] = None
    note: Optional[str] = None


class DbStatusResponse(BaseModel):
    generated_at: str
    databases: Dict[str, DbRecord]


def _file_size_mb(path: Path) -> Optional[float]:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return None


def _dir_size_mb(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                continue
    return total / (1024 * 1024)


def _sqlite_record(path: Path, tables_to_count: Optional[List[str]] = None, note: Optional[str] = None) -> DbRecord:
    record = DbRecord(path=str(path), exists=path.exists(), type="sqlite", note=note)
    if not path.exists():
        return record

    record.size_mb = _file_size_mb(path)

    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [row[0] for row in cursor.fetchall()]
        record.table_count = len(table_names)

        if tables_to_count:
            counts = {}
            for table in tables_to_count:
                if table in table_names:
                    try:
                        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        counts[table] = cur.fetchone()[0]
                    except Exception:
                        counts[table] = -1
            record.tables = counts
        conn.close()
    except Exception as exc:
        record.note = f"{note + '; ' if note else ''}error: {exc}"

    return record


def _lmdb_record(path: Path, note: Optional[str] = None) -> DbRecord:
    record = DbRecord(path=str(path), exists=path.exists(), type="lmdb", note=note)
    if not path.exists():
        return record

    record.size_mb = _dir_size_mb(path) if path.is_dir() else _file_size_mb(path)

    if path.is_dir():
        try:
            record.files = sorted([p.name for p in path.iterdir() if p.is_file()])
        except Exception:
            pass

    return record


def _gob_record(path: Path, note: Optional[str] = None) -> DbRecord:
    record = DbRecord(path=str(path), exists=path.exists(), type="gob", note=note)
    if not path.exists():
        return record

    record.size_mb = _dir_size_mb(path)
    files = []
    try:
        for p in sorted(path.glob("*.gob")):
            size = _file_size_mb(p)
            if size is None:
                files.append(p.name)
            else:
                files.append(f"{p.name} ({size:.1f} MB)")
    except Exception:
        pass
    record.files = files
    return record


def _default_identity_db_path() -> Path:
    env_path = os.environ.get("IDENTITY_FACTORY_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()
    cluster_db = Path(__file__).resolve().parent.parent.parent / "cluster_circuits.db"
    if cluster_db.exists():
        return cluster_db
    return Path.home() / ".identity_factory" / "circuits.db"


@router.get("/db-status", response_model=DbStatusResponse)
async def db_status() -> DbStatusResponse:
    """Return a compact status report of known databases."""
    api_root = Path(__file__).resolve().parent.parent.parent  # identity-factory-api
    repo_root = api_root.parent

    identity_db = _default_identity_db_path()
    imported_db = Path.home() / ".identity_factory" / "imported_identities.db"
    irreducible_db = Path.home() / ".identity_factory" / "irreducible.db"

    cluster_db = Path(
        os.environ.get("CLUSTER_DB_PATH", api_root / "cluster_circuits.db")
    ).expanduser()
    wire_shuffler_db = Path(
        os.environ.get("WIRE_SHUFFLER_DB_PATH", api_root / "wire_shuffler.db")
    ).expanduser()
    sat_db = Path(os.environ.get("SAT_DB_PATH", api_root / "sat_circuits.db")).expanduser()
    legacy_identity_db = api_root / "identity_circuits.db"

    local_mixing_sqlite = repo_root / "local_mixing" / "db" / "circuits.db"
    local_mixing_lmdb = Path(
        os.environ.get(
            "LOCAL_MIXING_PERM_DB_PATH",
            repo_root / "local_mixing" / "db",
        )
    )

    template_lmdb_primary = repo_root / "local_mixing" / "data" / "collection.lmdb"
    template_lmdb_sat = repo_root / "sat_revsynth" / "data" / "collection.lmdb"
    template_lmdb_sat_alt = repo_root / "sat_revsynth" / "collection.lmdb"
    template_lmdb_root = repo_root / "collection.lmdb"

    eca57_lmdb = Path(
        os.environ.get(
            "ECA57_LMDB_PATH",
            repo_root / "sat_revsynth" / "data" / "eca57_identities_lmdb",
        )
    )

    skeleton_lmdb = Path(
        os.environ.get(
            "LOCAL_MIXING_DB_PATH",
            repo_root / "local_mixing" / "db",
        )
    )

    go_db_dir_actual = repo_root / "obfuscated-circuits" / "go-proj" / "db"
    go_db_dir_expected = Path(
        os.environ.get("GO_DB_DIR", api_root / "go-proj" / "db")
    ).expanduser()

    databases: Dict[str, DbRecord] = {
        "identity_factory": _sqlite_record(
            identity_db, ["circuits", "dim_groups", "jobs"]
        ),
        "imported_identities": _sqlite_record(imported_db, ["imported_identities"]),
        "irreducible": _sqlite_record(
            irreducible_db, ["forward_circuits", "inverse_circuits", "identity_circuits"]
        ),
        "cluster": _sqlite_record(cluster_db, ["circuits", "dim_groups"]),
        "identity_factory_legacy": _sqlite_record(
            legacy_identity_db,
            ["circuits", "dim_groups", "jobs"],
            note="Legacy placeholder (not default)",
        ),
        "wire_shuffler": _sqlite_record(
            wire_shuffler_db,
            ["wire_permutations", "wire_shuffler_circuits", "wire_shuffler_metrics", "waksman_circuits"],
        ),
        "sat_db": _sqlite_record(sat_db, ["circuits", "metadata"]),
        "local_mixing_sqlite": _sqlite_record(
            local_mixing_sqlite,
            None,
            note="Large rainbow table DB; counts skipped",
        ),
        "local_mixing_perm_lmdb": _lmdb_record(local_mixing_lmdb, note="Permutation tables"),
        "template_lmdb_local": _lmdb_record(template_lmdb_primary),
        "template_lmdb_sat": _lmdb_record(template_lmdb_sat),
        "template_lmdb_sat_alt": _lmdb_record(template_lmdb_sat_alt),
        "template_lmdb_root": _lmdb_record(template_lmdb_root),
        "eca57_identities_lmdb": _lmdb_record(eca57_lmdb),
        "skeleton_lmdb": _lmdb_record(skeleton_lmdb),
        "go_gob_dir_actual": _gob_record(
            go_db_dir_actual,
            note="Actual location in this workspace",
        ),
        "go_gob_dir_expected": _gob_record(
            go_db_dir_expected,
            note="Expected by /api/v1/go-database/* endpoints",
        ),
    }

    return DbStatusResponse(
        generated_at=datetime.utcnow().isoformat() + "Z",
        databases=databases,
    )
