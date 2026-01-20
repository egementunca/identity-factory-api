"""
Skeleton Graph Generator.
Wraps the ECA57SkeletonSynthesizer for finding circuits with specific collision structures.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from .base import (
    CircuitGenerator,
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    GeneratorStatus,
)

logger = logging.getLogger(__name__)


class SkeletonGenerator(CircuitGenerator):
    """
    Generator using ECA57SkeletonSynthesizer to find circuits with non-commuting chains.
    """

    def get_info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="skeleton",
            display_name="Skeleton Chain",
            description="Find circuits with specific non-commuting collision chains. Forces structure in the skeleton graph.",
            gate_sets=["eca57"],
            supports_pause=False,
            supports_incremental=False,  # Single shot search usually
            config_schema={
                "type": "object",
                "properties": {
                    "chain_length": {
                        "type": "integer",
                        "minimum": 3,
                        "description": "Length of the collision chain to enforce. If not set, enforces full chain.",
                    },
                    "solver": {
                        "type": "string",
                        "default": "glucose4",
                        "enum": ["minisat-gh", "glucose3", "glucose4", "cadical"],
                        "description": "SAT solver to use",
                    },
                    "avoid_adjacent_identical": {
                        "type": "boolean",
                        "default": True,
                        "description": "Forbid identical adjacent gates",
                    }
                },
            },
        )

    def supports_gate_set(self, gate_set: str) -> bool:
        return gate_set.lower() in ["eca57", "mcx"]

    def generate(
        self,
        width: int,
        gate_count: int,
        gate_set: str = "eca57",
        max_circuits: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> GenerationResult:
        """Generate circuits using skeleton synthesizer."""

        self._reset_state()
        run_id = self._generate_run_id()
        self._current_run_id = run_id
        self._status = GeneratorStatus.RUNNING

        config = config or {}
        chain_length = config.get("chain_length")
        solver_name = config.get("solver", "glucose4")
        avoid_adjacent_identical = config.get("avoid_adjacent_identical", True)
        
        target_count = max_circuits or 1
        
        # If chain_length is not provided, default to gate_count-1 (full chain) if desired, 
        # or let the synthesizer handle None (which means full chain in the implementation).
        
        started_at = datetime.now()
        circuits_found = 0
        generated_circuits = []

        def update_progress(status_msg: str):
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
                f"Skeleton generation starting: {width}w x {gate_count}g, chain={chain_length}"
            )

            # Import here to avoid early dependency errors if paths aren't set up
            try:
                from sat.solver import Solver
                from synthesizers.eca57_skeleton_synthesizer import ECA57SkeletonSynthesizer
                from truth_table.truth_table import TruthTable
            except ImportError as e:
                raise ImportError(f"Could not import SAT synthesizer modules: {e}. Ensure sat_revsynth/src is in PYTHONPATH.")

            solver = Solver(solver_name)
            
            # We want Identity circuits? The user didn't specify target. 
            # Usually synthesis is for identity. ECA57SkeletonSynthesizer takes an output TruthTable.
            # Assuming Identity for now as that's the main use case ("noncommuting sat circuit generation").
            tt = TruthTable(width) # Identity by default

            for i in range(target_count):
                if self.is_cancel_requested():
                    break
                
                update_progress(f"Solving for circuit {i+1}...")
                
                # Create fresh solver/synthesizer for each attempt if we wanted multiple.
                # But SAT solvers are deterministic unless we add random constraints or block previous solutions.
                # The current ECA57SkeletonSynthesizer doesn't natively support "next".
                # If we want multiple, we'd need to block the previous solution.
                # For now, let's just try to find one.
                
                synth = ECA57SkeletonSynthesizer(
                    tt, 
                    gate_count=gate_count, 
                    solver=solver,
                    chain_length=chain_length,
                    avoid_adjacent_identical=avoid_adjacent_identical
                )
                
                # If we already found some, we might want to block them?
                # The basic wrapper here might just return one.
                # If max_circuits > 1, this loop will just find the SAME one again unless we do something.
                # Let's just run it once for now.
                
                circuit = synth.solve()
                
                if circuit:
                    circuits_found += 1
                    
                    # Convert to minimal dict format associated with standard
                    gates_data = []
                    for g in circuit.gates():
                        # ECA57 gate: target, c1, c2
                        gates_data.append((g.target, g.ctrl1, g.ctrl2))

                    generated_circuits.append(
                        {
                            "width": width,
                            "gates": gates_data,
                            "chain_length": chain_length
                        }
                    )
                    
                    update_progress("Found circuit")
                    
                    # If we only support one for now (since blocking isn't implemented in this wrapper), break
                    if target_count > 1 and i < target_count - 1:
                         logger.warning("Skeleton generator currently only yields 1 unique circuit per request.")
                         break
                    
                else:
                    update_progress("UNSAT - No circuit found")
                    break # If UNSAT, no point retrying with same constraints

            completed_at = datetime.now()
            total_seconds = (completed_at - started_at).total_seconds()
            
            self._status = GeneratorStatus.COMPLETED
            
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
                circuits=generated_circuits,
            )

        except Exception as e:
            logger.exception("Skeleton generation failed")
            self._status = GeneratorStatus.FAILED
            return GenerationResult(
                success=False,
                run_id=run_id,
                generator_name=self.get_info().name,
                error=str(e),
                width=width,
                gate_count=gate_count,
            )
