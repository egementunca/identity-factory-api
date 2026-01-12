"""
Database management system for the Identity Circuit Factory.
Simplified structure focusing on circuits and dimension groups.
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
class CircuitRecord:
    """Represents a circuit stored in the database."""

    id: Optional[int]
    width: int
    gate_count: int  # This is the length in gates
    gates: List[Tuple]
    permutation: List[int]
    complexity_walk: Optional[List[int]] = None
    circuit_hash: Optional[str] = None
    dim_group_id: Optional[int] = None
    representative_id: Optional[int] = (
        None  # Points to representative circuit (self if representative)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "gate_count": self.gate_count,
            "gates": self.gates,
            "permutation": self.permutation,
            "complexity_walk": self.complexity_walk,
            "circuit_hash": self.circuit_hash,
            "dim_group_id": self.dim_group_id,
            "representative_id": self.representative_id,
        }

    def get_gate_composition(self) -> Tuple[int, int, int]:
        """Calculate gate composition (NOT, CNOT, CCNOT counts)."""
        not_count = sum(1 for gate in self.gates if gate[0] == "X")
        cnot_count = sum(1 for gate in self.gates if gate[0] == "CX")
        ccnot_count = sum(1 for gate in self.gates if gate[0] == "CCX")
        return (not_count, cnot_count, ccnot_count)


@dataclass
class DimGroupRecord:
    """Represents a dimension group - a collection of identity circuits with same (width, gate_count)."""

    id: Optional[int]
    width: int
    gate_count: int  # Number of gates in the circuits
    circuit_count: int = 0  # How many circuits are in this dim group
    is_processed: bool = False  # Whether unrolling has been done

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "gate_count": self.gate_count,
            "circuit_count": self.circuit_count,
            "is_processed": self.is_processed,
        }


@dataclass
class JobRecord:
    """Represents a job in the processing queue."""

    id: Optional[int]
    job_type: (
        str  # 'seed_generation', 'unrolling', 'post_processing', 'debris_analysis'
    )
    status: str  # 'pending', 'running', 'completed', 'failed'
    priority: int
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "priority": self.priority,
            "parameters": self.parameters,
            "result": self.result,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class CircuitDatabase:
    """Simplified database manager for identity circuit factory."""

    def __init__(self, db_path: str = "identity_circuits.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables with simplified schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Core circuit table - stores all identity circuits
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS circuits (
                    id INTEGER PRIMARY KEY,
                    width INTEGER NOT NULL,
                    gate_count INTEGER NOT NULL,
                    gates TEXT NOT NULL,
                    permutation TEXT NOT NULL,
                    complexity_walk TEXT,
                    circuit_hash TEXT UNIQUE,
                    dim_group_id INTEGER,
                    representative_id INTEGER,
                    FOREIGN KEY (representative_id) REFERENCES circuits(id),
                    FOREIGN KEY (dim_group_id) REFERENCES dim_groups(id)
                )
            """
            )

            # Dimension groups - collections of circuits with same (width, gate_count)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_groups (
                    id INTEGER PRIMARY KEY,
                    width INTEGER NOT NULL,
                    gate_count INTEGER NOT NULL,
                    circuit_count INTEGER DEFAULT 0,
                    is_processed BOOLEAN DEFAULT FALSE,
                    UNIQUE(width, gate_count)
                )
            """
            )

            # Job queue for processing tasks
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    parameters TEXT NOT NULL,
                    result TEXT,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circuits_hash ON circuits(circuit_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circuits_dim_group ON circuits(dim_group_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_circuits_representative ON circuits(representative_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dim_groups_dimensions ON dim_groups(width, gate_count)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")

            conn.commit()

    def _compute_circuit_hash(self, gates: List[Tuple], permutation: List[int]) -> str:
        """Compute a hash for a circuit based on gates and permutation."""
        # Create a deterministic string representation
        gates_str = str(sorted(gates))  # Sort for consistency
        perm_str = str(permutation)
        combined = f"{gates_str}|{perm_str}"

        # Create hash
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def store_circuit(self, circuit: CircuitRecord) -> int:
        """Store a circuit in the database."""
        # Compute hash if not provided
        if not circuit.circuit_hash:
            circuit.circuit_hash = self._compute_circuit_hash(
                circuit.gates, circuit.permutation
            )

        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO circuits (width, gate_count, gates, permutation, complexity_walk, 
                                       circuit_hash, dim_group_id, representative_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        circuit.width,
                        circuit.gate_count,
                        json.dumps(circuit.gates),
                        json.dumps(circuit.permutation),
                        (
                            json.dumps(circuit.complexity_walk)
                            if circuit.complexity_walk
                            else None
                        ),
                        circuit.circuit_hash,
                        circuit.dim_group_id,
                        circuit.representative_id,
                    ),
                )

                circuit_id = cursor.lastrowid

                # If representative_id is None, set it to point to itself
                if circuit.representative_id is None:
                    conn.execute(
                        "UPDATE circuits SET representative_id = ? WHERE id = ?",
                        (circuit_id, circuit_id),
                    )

                conn.commit()
                logger.info(
                    f"Stored circuit {circuit_id} with hash {circuit.circuit_hash}"
                )
                return circuit_id

            except sqlite3.IntegrityError as e:
                if "circuit_hash" in str(e):
                    # Circuit already exists
                    existing = self.get_circuit_by_hash(circuit.circuit_hash)
                    if existing:
                        logger.info(
                            f"Circuit with hash {circuit.circuit_hash} already exists as ID {existing.id}"
                        )
                        return existing.id
                raise

    def get_circuit(self, circuit_id: int) -> Optional[CircuitRecord]:
        """Get a circuit by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, gates, permutation, complexity_walk,
                       circuit_hash, dim_group_id, representative_id
                FROM circuits WHERE id = ?
            """,
                (circuit_id,),
            )

            row = cursor.fetchone()
            if row:
                return CircuitRecord(
                    id=row[0],
                    width=row[1],
                    gate_count=row[2],
                    gates=json.loads(row[3]),
                    permutation=json.loads(row[4]),
                    complexity_walk=json.loads(row[5]) if row[5] else None,
                    circuit_hash=row[6],
                    dim_group_id=row[7],
                    representative_id=row[8],
                )
        return None

    def get_circuit_by_hash(self, circuit_hash: str) -> Optional[CircuitRecord]:
        """Get a circuit by its hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, gates, permutation, complexity_walk,
                       circuit_hash, dim_group_id, representative_id
                FROM circuits WHERE circuit_hash = ?
            """,
                (circuit_hash,),
            )

            row = cursor.fetchone()
            if row:
                return CircuitRecord(
                    id=row[0],
                    width=row[1],
                    gate_count=row[2],
                    gates=json.loads(row[3]),
                    permutation=json.loads(row[4]),
                    complexity_walk=json.loads(row[5]) if row[5] else None,
                    circuit_hash=row[6],
                    dim_group_id=row[7],
                    representative_id=row[8],
                )
        return None

    def store_dim_group(self, dim_group: DimGroupRecord) -> int:
        """Store a dimension group in the database."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO dim_groups (width, gate_count, circuit_count, is_processed)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        dim_group.width,
                        dim_group.gate_count,
                        dim_group.circuit_count,
                        dim_group.is_processed,
                    ),
                )

                dim_group_id = cursor.lastrowid
                conn.commit()
                logger.info(
                    f"Created dimension group {dim_group_id} for ({dim_group.width}, {dim_group.gate_count})"
                )
                return dim_group_id

            except sqlite3.IntegrityError:
                # Dimension group already exists
                existing = self.get_dim_group(dim_group.width, dim_group.gate_count)
                if existing:
                    logger.info(
                        f"Dimension group for ({dim_group.width}, {dim_group.gate_count}) already exists as ID {existing.id}"
                    )
                    return existing.id
                raise

    def get_dim_group(self, width: int, gate_count: int) -> Optional[DimGroupRecord]:
        """Get a dimension group by width and gate count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, circuit_count, is_processed
                FROM dim_groups WHERE width = ? AND gate_count = ?
            """,
                (width, gate_count),
            )

            row = cursor.fetchone()
            if row:
                return DimGroupRecord(
                    id=row[0],
                    width=row[1],
                    gate_count=row[2],
                    circuit_count=row[3],
                    is_processed=bool(row[4]),
                )
        return None

    def get_dim_group_by_id(self, dim_group_id: int) -> Optional[DimGroupRecord]:
        """Get a dimension group by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, circuit_count, is_processed
                FROM dim_groups WHERE id = ?
            """,
                (dim_group_id,),
            )

            row = cursor.fetchone()
            if row:
                return DimGroupRecord(
                    id=row[0],
                    width=row[1],
                    gate_count=row[2],
                    circuit_count=row[3],
                    is_processed=bool(row[4]),
                )
        return None

    def add_circuit_to_dim_group(self, dim_group_id: int, circuit_id: int) -> bool:
        """Add a circuit to a dimension group and update counts."""
        with sqlite3.connect(self.db_path) as conn:
            # Update the circuit's dim_group_id
            conn.execute(
                "UPDATE circuits SET dim_group_id = ? WHERE id = ?",
                (dim_group_id, circuit_id),
            )

            # Update the dimension group's circuit count
            conn.execute(
                """
                UPDATE dim_groups 
                SET circuit_count = (
                    SELECT COUNT(*) FROM circuits WHERE dim_group_id = ?
                ) 
                WHERE id = ?
            """,
                (dim_group_id, dim_group_id),
            )

            conn.commit()
            logger.info(f"Added circuit {circuit_id} to dimension group {dim_group_id}")
            return True

    def get_circuits_in_dim_group(self, dim_group_id: int) -> List[CircuitRecord]:
        """Get all circuits in a dimension group."""
        circuits = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, gates, permutation, complexity_walk,
                       circuit_hash, dim_group_id, representative_id
                FROM circuits WHERE dim_group_id = ?
                ORDER BY id
            """,
                (dim_group_id,),
            )

            for row in cursor.fetchall():
                circuits.append(
                    CircuitRecord(
                        id=row[0],
                        width=row[1],
                        gate_count=row[2],
                        gates=json.loads(row[3]),
                        permutation=json.loads(row[4]),
                        complexity_walk=json.loads(row[5]) if row[5] else None,
                        circuit_hash=row[6],
                        dim_group_id=row[7],
                        representative_id=row[8],
                    )
                )

        return circuits

    def get_representatives_in_dim_group(
        self, dim_group_id: int
    ) -> List[CircuitRecord]:
        """Get all representative circuits in a dimension group (where representative_id points to itself)."""
        circuits = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, gates, permutation, complexity_walk,
                       circuit_hash, dim_group_id, representative_id
                FROM circuits WHERE dim_group_id = ? AND id = representative_id
                ORDER BY id
            """,
                (dim_group_id,),
            )

            for row in cursor.fetchall():
                circuits.append(
                    CircuitRecord(
                        id=row[0],
                        width=row[1],
                        gate_count=row[2],
                        gates=json.loads(row[3]),
                        permutation=json.loads(row[4]),
                        complexity_walk=json.loads(row[5]) if row[5] else None,
                        circuit_hash=row[6],
                        dim_group_id=row[7],
                        representative_id=row[8],
                    )
                )

        return circuits

    def get_equivalents_for_representative(
        self, representative_id: int
    ) -> List[CircuitRecord]:
        """Get all circuits that point to a specific representative."""
        circuits = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, gates, permutation, complexity_walk,
                       circuit_hash, dim_group_id, representative_id
                FROM circuits WHERE representative_id = ? AND id != representative_id
                ORDER BY id
            """,
                (representative_id,),
            )

            for row in cursor.fetchall():
                circuits.append(
                    CircuitRecord(
                        id=row[0],
                        width=row[1],
                        gate_count=row[2],
                        gates=json.loads(row[3]),
                        permutation=json.loads(row[4]),
                        complexity_walk=json.loads(row[5]) if row[5] else None,
                        circuit_hash=row[6],
                        dim_group_id=row[7],
                        representative_id=row[8],
                    )
                )

        return circuits

    def get_circuits_by_gate_composition(
        self, dim_group_id: int, gate_composition: Tuple[int, int, int]
    ) -> List[CircuitRecord]:
        """Get circuits in a dimension group with specific gate composition."""
        circuits = self.get_circuits_in_dim_group(dim_group_id)
        return [c for c in circuits if c.get_gate_composition() == gate_composition]

    def get_all_dim_groups(self) -> List[DimGroupRecord]:
        """Get all dimension groups."""
        dim_groups = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, width, gate_count, circuit_count, is_processed
                FROM dim_groups ORDER BY width, gate_count
            """
            )

            for row in cursor.fetchall():
                dim_groups.append(
                    DimGroupRecord(
                        id=row[0],
                        width=row[1],
                        gate_count=row[2],
                        circuit_count=row[3],
                        is_processed=bool(row[4]),
                    )
                )

        return dim_groups

    def mark_dim_group_processed(self, dim_group_id: int):
        """Mark a dimension group as processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE dim_groups SET is_processed = TRUE WHERE id = ?",
                (dim_group_id,),
            )
            conn.commit()
            logger.info(f"Marked dimension group {dim_group_id} as processed")

    def create_job(self, job: JobRecord) -> int:
        """Create a new job in the queue."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (job_type, status, priority, parameters)
                VALUES (?, ?, ?, ?)
            """,
                (job.job_type, job.status, job.priority, json.dumps(job.parameters)),
            )

            job_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created job {job_id} of type {job.job_type}")
            return job_id

    def get_pending_jobs(
        self, job_type: Optional[str] = None, limit: int = 10
    ) -> List[JobRecord]:
        """Get pending jobs from the queue."""
        jobs = []
        with sqlite3.connect(self.db_path) as conn:
            if job_type:
                cursor = conn.execute(
                    """
                    SELECT id, job_type, status, priority, parameters, result, error_message,
                           created_at, started_at, completed_at
                    FROM jobs WHERE status = 'pending' AND job_type = ?
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """,
                    (job_type, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, job_type, status, priority, parameters, result, error_message,
                           created_at, started_at, completed_at
                    FROM jobs WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """,
                    (limit,),
                )

            for row in cursor.fetchall():
                jobs.append(
                    JobRecord(
                        id=row[0],
                        job_type=row[1],
                        status=row[2],
                        priority=row[3],
                        parameters=json.loads(row[4]),
                        result=json.loads(row[5]) if row[5] else None,
                        error_message=row[6],
                        started_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    )
                )

        return jobs

    def update_job_status(
        self,
        job_id: int,
        status: str,
        result: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ):
        """Update job status and result."""
        with sqlite3.connect(self.db_path) as conn:
            if status == "running":
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, started_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (status, job_id),
                )
            elif status in ["completed", "failed"]:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, result = ?, error_message = ?, 
                                   completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (
                        status,
                        json.dumps(result) if result else None,
                        error_message,
                        job_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, result = ?, error_message = ?
                    WHERE id = ?
                """,
                    (
                        status,
                        json.dumps(result) if result else None,
                        error_message,
                        job_id,
                    ),
                )

            conn.commit()
            logger.info(f"Updated job {job_id} status to {status}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM circuits")
            total_circuits = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM dim_groups")
            total_dim_groups = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM circuits WHERE id = representative_id"
            )
            total_representatives = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM circuits WHERE id != representative_id"
            )
            total_equivalents = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")
            pending_jobs = cursor.fetchone()[0]

            return {
                "total_circuits": total_circuits,
                "total_dim_groups": total_dim_groups,
                "total_representatives": total_representatives,
                "total_equivalents": total_equivalents,
                "pending_jobs": pending_jobs,
            }

    def delete_circuit(self, circuit_id: int) -> bool:
        """Delete a circuit from the database."""
        with sqlite3.connect(self.db_path) as conn:
            # First check if any circuits point to this as representative
            cursor = conn.execute(
                "SELECT COUNT(*) FROM circuits WHERE representative_id = ? AND id != ?",
                (circuit_id, circuit_id),
            )
            dependent_count = cursor.fetchone()[0]

            if dependent_count > 0:
                logger.warning(
                    f"Cannot delete circuit {circuit_id} - {dependent_count} circuits depend on it as representative"
                )
                return False

            # Get the dim_group_id before deletion
            cursor = conn.execute(
                "SELECT dim_group_id FROM circuits WHERE id = ?", (circuit_id,)
            )
            row = cursor.fetchone()
            dim_group_id = row[0] if row else None

            # Delete the circuit
            conn.execute("DELETE FROM circuits WHERE id = ?", (circuit_id,))

            # Update dimension group count if needed
            if dim_group_id:
                conn.execute(
                    """
                    UPDATE dim_groups 
                    SET circuit_count = (
                        SELECT COUNT(*) FROM circuits WHERE dim_group_id = ?
                    ) 
                    WHERE id = ?
                """,
                    (dim_group_id, dim_group_id),
                )

            conn.commit()
            logger.info(f"Deleted circuit {circuit_id}")
            return True
