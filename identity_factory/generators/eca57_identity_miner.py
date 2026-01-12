"""
ECA57 Identity Miner Generator.
Uses SAT-based identity mining for ECA57 (Gate 57) circuits.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .base import (
    CircuitGenerator,
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    GeneratorStatus,
)

logger = logging.getLogger(__name__)


class ECA57IdentityMinerGenerator(CircuitGenerator):
    """
    Generator using SAT-based identity mining for ECA57 circuits.

    This wraps the sat_revsynth.eca57.identity_miner functionality.
    Finds nontrivial identity circuits for the ECA57 gate set.
    """

    def get_info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="eca57_identity_miner",
            display_name="ECA57 Identity Miner",
            description="SAT-based mining of nontrivial identity circuits for ECA57 (Gate 57). Finds circuits where composed permutation equals identity.",
            gate_sets=["eca57"],
            supports_pause=False,
            supports_incremental=True,
            config_schema={
                "type": "object",
                "properties": {
                    "solver": {
                        "type": "string",
                        "default": "minisat-gh",
                        "enum": ["minisat-gh", "glucose", "cadical"],
                        "description": "SAT solver to use",
                    },
                    "require_spread": {
                        "type": "boolean",
                        "default": True,
                        "description": "Require gates to use multiple target wires",
                    },
                    "min_targets_used": {
                        "type": "integer",
                        "default": 2,
                        "description": "Minimum number of distinct target wires",
                    },
                    "incremental": {
                        "type": "boolean",
                        "default": False,
                        "description": "Mine incrementally, filtering with found patterns",
                    },
                },
            },
        )

    def supports_gate_set(self, gate_set: str) -> bool:
        return gate_set.lower() == "eca57"

    def generate(
        self,
        width: int,
        gate_count: int,
        gate_set: str = "eca57",
        max_circuits: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> GenerationResult:
        """Generate identity circuits using ECA57 SAT mining."""

        if not self.supports_gate_set(gate_set):
            return GenerationResult(
                success=False,
                run_id=self._generate_run_id(),
                generator_name=self.get_info().name,
                error=f"Gate set '{gate_set}' not supported. Use 'eca57'.",
            )

        self._reset_state()
        run_id = self._generate_run_id()
        self._current_run_id = run_id
        self._status = GeneratorStatus.RUNNING

        config = config or {}
        solver_name = config.get("solver", "minisat-gh")
        require_spread = config.get("require_spread", True)
        min_targets_used = config.get("min_targets_used", 2)
        incremental = config.get("incremental", False)
        target_count = max_circuits or 100

        started_at = datetime.now()
        circuits_found = 0
        found_circuits = []

        # Update progress helper
        def update_progress(status_msg: str, found: int = 0):
            nonlocal circuits_found
            if found > 0:
                circuits_found = found
            elapsed = (datetime.now() - started_at).total_seconds()
            progress = GenerationProgress(
                run_id=run_id,
                generator_name=self.get_info().name,
                status=GeneratorStatus.RUNNING,
                circuits_found=circuits_found,
                circuits_stored=circuits_found,
                current_gate_count=gate_count,
                current_width=width,
                started_at=started_at,
                elapsed_seconds=elapsed,
                circuits_per_second=circuits_found / elapsed if elapsed > 0 else 0,
                current_status=status_msg,
            )
            if progress_callback:
                progress_callback(progress)
            self._update_progress(progress)

        try:
            logger.info(
                f"ECA57 generator starting: {width}w x {gate_count}g, max={target_count}"
            )
            update_progress(
                f"Starting ECA57 identity mining for {width}w x {gate_count}g..."
            )

            # Import ECA57 components
            from eca57.identity_miner import ECA57IdentityMiner
            from sat.solver import Solver

            solver = Solver(solver_name)
            miner = ECA57IdentityMiner(n_wires=width, solver=solver)

            logger.info(f"ECA57 IdentityMiner created, starting mining...")
            update_progress(f"Mining {gate_count}-gate identity circuits...")

            if incremental:
                # Mine incrementally from small to target gate count
                results = miner.mine_incremental(
                    min_gates=2, max_gates=gate_count, solutions_per_length=target_count
                )
                for result in results:
                    if self.is_cancel_requested():
                        break
                    circuits_found += result.total_found
                    found_circuits.extend(result.circuits)
                    update_progress(
                        f"Found {circuits_found} circuits...", circuits_found
                    )
            else:
                # Mine specific gate count
                logger.info(
                    f"Calling mine_identities(gate_count={gate_count}, max_solutions={target_count})"
                )
                result = miner.mine_identities(
                    gate_count=gate_count,
                    max_solutions=target_count,
                    require_spread=require_spread,
                    min_targets_used=min_targets_used,
                )
                logger.info(
                    f"mine_identities returned: success={result.success}, found={result.total_found}"
                )
                circuits_found = result.total_found
                found_circuits = result.circuits
                update_progress(
                    f"Found {circuits_found} identity circuits", circuits_found
                )

            completed_at = datetime.now()
            total_seconds = (completed_at - started_at).total_seconds()

            logger.info(
                f"ECA57 mining completed: {circuits_found} circuits in {total_seconds:.2f}s"
            )

            self._status = (
                GeneratorStatus.COMPLETED
                if not self.is_cancel_requested()
                else GeneratorStatus.CANCELLED
            )

            # Convert ECA57Circuit objects to dict format for database storage
            circuit_dicts = []
            for circuit in found_circuits:
                circuit_dicts.append(
                    {
                        "width": circuit.width(),
                        "gates": circuit.gates_as_tuples(),
                        "permutation": list(circuit.perm()),
                        "gate_count": len(circuit),
                    }
                )

            return GenerationResult(
                success=circuits_found > 0,
                run_id=run_id,
                generator_name=self.get_info().name,
                total_circuits=circuits_found,
                new_circuits=circuits_found,
                duplicates=0,
                width=width,
                gate_count=gate_count,
                gate_set=gate_set,
                config=config,
                started_at=started_at,
                completed_at=completed_at,
                total_seconds=total_seconds,
                circuits=circuit_dicts,  # Include generated circuit data
            )

        except Exception as e:
            logger.exception("ECA57 Identity Mining failed")
            self._status = GeneratorStatus.FAILED
            return GenerationResult(
                success=False,
                run_id=run_id,
                generator_name=self.get_info().name,
                error=str(e),
                width=width,
                gate_count=gate_count,
            )
