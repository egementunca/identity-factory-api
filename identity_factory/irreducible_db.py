"""
Irreducible Circuit Database.

Stores forward circuits (touching all wires), their inverses, and resulting identities.
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ForwardCircuit:
    """Forward circuit that touches all wires."""

    id: Optional[int]
    width: int
    gate_count: int
    gates: List[Tuple[int, int, int]]  # [(c1, c2, target), ...]
    permutation: List[int]
    permutation_hash: str
    created_at: Optional[str] = None


@dataclass
class InverseCircuit:
    """Inverse of a forward circuit."""

    id: Optional[int]
    forward_id: int
    gate_count: int
    gates: List[Tuple[int, int, int]]
    synthesis_method: str  # 'reverse', 'sat_optimal', 'sat_bounded'
    created_at: Optional[str] = None


@dataclass
class IdentityCircuit:
    """Identity circuit formed by forward + inverse."""

    id: Optional[int]
    forward_id: int
    inverse_id: int
    width: int
    total_gates: int
    quality_score: Optional[float] = None
    created_at: Optional[str] = None


class IrreducibleDatabase:
    """Database for irreducible circuits."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".identity_factory" / "irreducible.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

        logger.info(f"IrreducibleDatabase initialized at {db_path}")

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Forward circuits table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_circuits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                width INTEGER NOT NULL,
                gate_count INTEGER NOT NULL,
                gates TEXT NOT NULL,
                permutation TEXT NOT NULL,
                permutation_hash TEXT NOT NULL,
                touches_all_wires BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_perm_hash 
            ON forward_circuits(permutation_hash)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_width_gates 
            ON forward_circuits(width, gate_count)
        """
        )

        # Inverse circuits table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inverse_circuits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forward_id INTEGER NOT NULL,
                gate_count INTEGER NOT NULL,
                gates TEXT NOT NULL,
                synthesis_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (forward_id) REFERENCES forward_circuits(id)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_id 
            ON inverse_circuits(forward_id)
        """
        )

        # Identity circuits table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_circuits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forward_id INTEGER NOT NULL,
                inverse_id INTEGER NOT NULL,
                width INTEGER NOT NULL,
                total_gates INTEGER NOT NULL,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (forward_id) REFERENCES forward_circuits(id),
                FOREIGN KEY (inverse_id) REFERENCES inverse_circuits(id)
            )
        """
        )

        self.conn.commit()
        logger.info("Database schema initialized")

    @staticmethod
    def compute_permutation_hash(permutation: List[int]) -> str:
        """Compute hash of permutation for fast lookup."""
        perm_str = ",".join(map(str, permutation))
        return hashlib.sha256(perm_str.encode()).hexdigest()[:16]

    def store_forward(self, circuit: ForwardCircuit) -> int:
        """Store forward circuit, return ID."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO forward_circuits 
            (width, gate_count, gates, permutation, permutation_hash)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                circuit.width,
                circuit.gate_count,
                json.dumps(circuit.gates),
                json.dumps(circuit.permutation),
                circuit.permutation_hash,
            ),
        )

        self.conn.commit()
        circuit_id = cursor.lastrowid
        logger.info(
            f"Stored forward circuit {circuit_id}: {circuit.width}w × {circuit.gate_count}g"
        )
        return circuit_id

    def store_inverse(self, inverse: InverseCircuit) -> int:
        """Store inverse circuit, return ID."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO inverse_circuits 
            (forward_id, gate_count, gates, synthesis_method)
            VALUES (?, ?, ?, ?)
        """,
            (
                inverse.forward_id,
                inverse.gate_count,
                json.dumps(inverse.gates),
                inverse.synthesis_method,
            ),
        )

        self.conn.commit()
        inverse_id = cursor.lastrowid
        logger.info(
            f"Stored inverse circuit {inverse_id} for forward {inverse.forward_id}"
        )
        return inverse_id

    def store_identity(self, identity: IdentityCircuit) -> int:
        """Store identity circuit, return ID."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO identity_circuits 
            (forward_id, inverse_id, width, total_gates, quality_score)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                identity.forward_id,
                identity.inverse_id,
                identity.width,
                identity.total_gates,
                identity.quality_score,
            ),
        )

        self.conn.commit()
        identity_id = cursor.lastrowid
        logger.info(
            f"Stored identity circuit {identity_id}: {identity.total_gates} gates"
        )
        return identity_id

    def get_forward(self, circuit_id: int) -> Optional[ForwardCircuit]:
        """Get forward circuit by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM forward_circuits WHERE id = ?", (circuit_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return ForwardCircuit(
            id=row["id"],
            width=row["width"],
            gate_count=row["gate_count"],
            gates=json.loads(row["gates"]),
            permutation=json.loads(row["permutation"]),
            permutation_hash=row["permutation_hash"],
            created_at=row["created_at"],
        )

    def get_inverse(self, inverse_id: int) -> Optional[InverseCircuit]:
        """Get inverse circuit by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM inverse_circuits WHERE id = ?", (inverse_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return InverseCircuit(
            id=row["id"],
            forward_id=row["forward_id"],
            gate_count=row["gate_count"],
            gates=json.loads(row["gates"]),
            synthesis_method=row["synthesis_method"],
            created_at=row["created_at"],
        )

    def get_identity(self, identity_id: int) -> Optional[IdentityCircuit]:
        """Get identity circuit by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM identity_circuits WHERE id = ?", (identity_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return IdentityCircuit(
            id=row["id"],
            forward_id=row["forward_id"],
            inverse_id=row["inverse_id"],
            width=row["width"],
            total_gates=row["total_gates"],
            quality_score=row["quality_score"],
            created_at=row["created_at"],
        )

    def list_forward_by_width(
        self, width: int, limit: int = 100
    ) -> List[ForwardCircuit]:
        """List forward circuits by width."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM forward_circuits 
            WHERE width = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """,
            (width, limit),
        )

        circuits = []
        for row in cursor.fetchall():
            circuits.append(
                ForwardCircuit(
                    id=row["id"],
                    width=row["width"],
                    gate_count=row["gate_count"],
                    gates=json.loads(row["gates"]),
                    permutation=json.loads(row["permutation"]),
                    permutation_hash=row["permutation_hash"],
                    created_at=row["created_at"],
                )
            )

        return circuits

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM forward_circuits")
        forward_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM inverse_circuits")
        inverse_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM identity_circuits")
        identity_count = cursor.fetchone()["count"]

        cursor.execute(
            """
            SELECT width, COUNT(*) as count 
            FROM forward_circuits 
            GROUP BY width
        """
        )
        by_width = {row["width"]: row["count"] for row in cursor.fetchall()}

        return {
            "forward_circuits": forward_count,
            "inverse_circuits": inverse_count,
            "identity_circuits": identity_count,
            "by_width": by_width,
        }

    def close(self):
        """Close database connection."""
        self.conn.close()
