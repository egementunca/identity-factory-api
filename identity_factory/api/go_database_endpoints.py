"""
Go Database API endpoints.
Provides access to pre-generated Go circuit databases.
"""

"""
WARNING: This module depends on the `obfuscated-circuits/go-proj` directory which is currently missing from the API root.
These endpoints will fail until the Go binaries (`go_stats`, `go_explore`) and database files are restored.
"""
import logging
import os
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/go-database", tags=["go-database"])

# Path to Go database files (override via env)
GO_PROJ_DIR = Path(
    os.environ.get("GO_PROJ_DIR", Path(__file__).parent.parent.parent / "go-proj")
).expanduser()
GO_DB_DIR = Path(os.environ.get("GO_DB_DIR", GO_PROJ_DIR / "db")).expanduser()


class GoCircuitInfo(BaseModel):
    """Information about a circuit from Go database."""

    n_wires: int
    n_gates: int
    permutation: List[int]
    circuit_id: str


class GoDatabaseStats(BaseModel):
    """Statistics about Go database files."""

    n_wires: int
    n_gates: int
    file_name: str
    file_size_bytes: int
    available: bool


class GoDatabaseListResponse(BaseModel):
    """Response listing available Go databases."""

    databases: List[GoDatabaseStats]
    total_databases: int


def parse_gob_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a Go .gob file to extract circuit data.

    Go gob format is binary, we'll attempt basic parsing.
    For complex gob files, this may need a Go helper.
    """
    # For now, return file metadata since gob decoding in Python is complex
    # The actual circuit data would need a Go helper or gob library
    stats = file_path.stat()
    return {"file_name": file_path.name, "file_size": stats.st_size, "available": True}


@router.get("/", response_model=GoDatabaseListResponse)
async def list_go_databases():
    """List all available Go circuit databases."""
    databases = []

    if GO_DB_DIR.exists():
        for gob_file in sorted(GO_DB_DIR.glob("*.gob")):
            # Parse filename like n3m4.gob -> n=3, m=4
            name = gob_file.stem
            n_wires = 0
            n_gates = 0

            if "n" in name and "m" in name:
                try:
                    parts = name.split("m")
                    n_wires = int(parts[0][1:])  # n3 -> 3
                    n_gates = int(parts[1])  # 4 -> 4
                except (ValueError, IndexError):
                    pass

            stats = gob_file.stat()
            databases.append(
                GoDatabaseStats(
                    n_wires=n_wires,
                    n_gates=n_gates,
                    file_name=gob_file.name,
                    file_size_bytes=stats.st_size,
                    available=True,
                )
            )

    return GoDatabaseListResponse(databases=databases, total_databases=len(databases))


@router.get("/stats")
async def get_go_database_stats():
    """Get summary statistics for all Go databases, including circuit counts."""
    import json as json_lib
    import subprocess

    # Path to go_stats binary
    go_stats_binary = GO_PROJ_DIR / "go_stats"

    # Try to get detailed stats using go_stats binary
    detailed_stats = None
    if go_stats_binary.exists() and GO_DB_DIR.exists():
        try:
            result = subprocess.run(
                [str(go_stats_binary), str(GO_DB_DIR)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                detailed_stats = json_lib.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to run go_stats: {e}")

    # If go_stats worked, return its output enhanced
    if detailed_stats:
        return {
            "source": "go-project",
            "description": "Pre-generated exhaustive circuit enumeration from Go project",
            "total_databases": len(detailed_stats.get("databases", [])),
            "total_permutations": detailed_stats.get("totals", {}).get(
                "total_permutations", 0
            ),
            "total_circuits": detailed_stats.get("totals", {}).get("total_circuits", 0),
            "databases": detailed_stats.get("databases", []),
            "wire_count": (
                detailed_stats["databases"][0]["n_wires"]
                if detailed_stats.get("databases")
                else 0
            ),
        }

    # Fallback to file-based stats
    databases = []
    total_size = 0

    if GO_DB_DIR.exists():
        for gob_file in sorted(GO_DB_DIR.glob("*.gob")):
            name = gob_file.stem
            n_wires = 0
            n_gates = 0

            if "n" in name and "m" in name:
                try:
                    parts = name.split("m")
                    n_wires = int(parts[0][1:])
                    n_gates = int(parts[1])
                except (ValueError, IndexError):
                    pass

            stats = gob_file.stat()
            total_size += stats.st_size

            databases.append(
                {
                    "n_wires": n_wires,
                    "n_gates": n_gates,
                    "file_name": gob_file.name,
                    "file_size_bytes": stats.st_size,
                }
            )

    return {
        "source": "go-project",
        "description": "Pre-generated exhaustive circuit enumeration from Go project",
        "total_databases": len(databases),
        "total_size_bytes": total_size,
        "databases": databases,
        "largest_enumeration": (
            max([d["n_gates"] for d in databases]) if databases else 0
        ),
        "wire_count": databases[0]["n_wires"] if databases else 0,
        "note": "Compile go_stats binary for detailed circuit counts",
    }


@router.get("/{file_name}")
async def get_go_database_info(file_name: str):
    """Get information about a specific Go database file."""
    file_path = GO_DB_DIR / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Database file '{file_name}' not found"
        )

    if not file_path.suffix == ".gob":
        raise HTTPException(status_code=400, detail="Only .gob files are supported")

    stats = file_path.stat()
    name = file_path.stem
    n_wires = 0
    n_gates = 0

    if "n" in name and "m" in name:
        try:
            parts = name.split("m")
            n_wires = int(parts[0][1:])
            n_gates = int(parts[1])
        except (ValueError, IndexError):
            pass

    return {
        "file_name": file_name,
        "n_wires": n_wires,
        "n_gates": n_gates,
        "file_size_bytes": stats.st_size,
        "path": str(file_path),
        "note": "Circuit data extraction requires Go binary helper. File contains serialized circuit permutation data.",
    }


@router.get("/{file_name}/circuits")
async def list_go_database_circuits(file_name: str, limit: int = 50, offset: int = 0):
    """List circuits in a Go database file using go_explore binary."""
    import json as json_lib
    import subprocess

    file_path = GO_DB_DIR / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Database file '{file_name}' not found"
        )

    if not file_path.suffix == ".gob":
        raise HTTPException(status_code=400, detail="Only .gob files are supported")

    # Path to go_explore binary
    go_explore_binary = GO_PROJ_DIR / "go_explore"

    if not go_explore_binary.exists():
        raise HTTPException(
            status_code=503,
            detail="go_explore binary not found. Build it with: cd go-proj && go build -o go_explore cmd/go_explore/main.go",
        )

    try:
        result = subprocess.run(
            [
                str(go_explore_binary),
                str(file_path),
                "--limit",
                str(limit),
                "--offset",
                str(offset),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error(f"go_explore failed: {result.stderr}")
            raise HTTPException(
                status_code=500, detail=f"Failed to read database: {result.stderr}"
            )

        data = json_lib.loads(result.stdout)
        return data

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Database read timed out")
    except json_lib.JSONDecodeError as e:
        logger.error(f"Failed to parse go_explore output: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse circuit data")
    except Exception as e:
        logger.exception("Error reading Go database")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_name}/circuit/{perm_key}")
async def get_go_database_circuit(file_name: str, perm_key: str):
    """Get a specific circuit from Go database by permutation key."""
    import json as json_lib
    import subprocess

    file_path = GO_DB_DIR / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Database file '{file_name}' not found"
        )

    go_explore_binary = GO_PROJ_DIR / "go_explore"

    if not go_explore_binary.exists():
        raise HTTPException(status_code=503, detail="go_explore binary not found")

    try:
        result = subprocess.run(
            [str(go_explore_binary), str(file_path), "--perm", perm_key],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"Failed to read circuit: {result.stderr}"
            )

        data = json_lib.loads(result.stdout)

        if data.get("error"):
            raise HTTPException(status_code=404, detail=data["error"])

        if not data.get("circuits"):
            raise HTTPException(status_code=404, detail="Circuit not found")

        return data["circuits"][0]

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Read timed out")
    except json_lib.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse circuit data")
