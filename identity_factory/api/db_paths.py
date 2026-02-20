"""
Shared helpers for resolving database paths.
"""

import os
from pathlib import Path


def resolve_cluster_db_path() -> Path:
    api_root = Path(__file__).resolve().parent.parent.parent  # identity-factory-api/
    return api_root / "cluster_circuits.db"


def resolve_identity_db_path() -> Path:
    env_path = os.environ.get("IDENTITY_FACTORY_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()

    cluster_db = resolve_cluster_db_path()
    if cluster_db.exists():
        return cluster_db

    return Path.home() / ".identity_factory" / "circuits.db"

