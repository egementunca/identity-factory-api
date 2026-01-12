"""
SAT Forward/Inverse Generator.
Wraps the existing seed_generator for forward + SAT inverse synthesis.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .base import (
    CircuitGenerator,
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    GeneratorStatus,
)

logger = logging.getLogger(__name__)


class SATForwardInverseGenerator(CircuitGenerator):
    """
    Generator using random forward circuit + SAT-based inverse synthesis.

    This wraps the existing seed_generator.py functionality.
    """

    def get_info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="sat_forward_inverse",
            display_name="SAT Forward/Inverse",
            description="Generate random forward circuit, then use SAT solver to find optimal inverse. Creates identity circuits with known structure.",
            gate_sets=["mcx"],  # Toffoli-based
            supports_pause=False,
            supports_incremental=True,
            config_schema={
                "type": "object",
                "properties": {
                    "max_inverse_gates": {
                        "type": "integer",
                        "default": 40,
                        "description": "Maximum gates in inverse circuit",
                    },
                    "solver": {
                        "type": "string",
                        "default": "minisat-gh",
                        "enum": ["minisat-gh", "glucose", "cadical"],
                        "description": "SAT solver to use",
                    },
                    "max_attempts": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum generation attempts",
                    },
                },
            },
        )

    def supports_gate_set(self, gate_set: str) -> bool:
        return gate_set.lower() == "mcx"

    def generate(
        self,
        width: int,
        gate_count: int,
        gate_set: str = "mcx",
        max_circuits: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> GenerationResult:
        """Generate identity circuits using forward + SAT inverse method."""

        if not self.supports_gate_set(gate_set):
            return GenerationResult(
                success=False,
                run_id=self._generate_run_id(),
                generator_name=self.get_info().name,
                error=f"Gate set '{gate_set}' not supported. Use 'mcx'.",
            )

        self._reset_state()
        run_id = self._generate_run_id()
        self._current_run_id = run_id
        self._status = GeneratorStatus.RUNNING

        config = config or {}
        max_inverse_gates = config.get("max_inverse_gates", 40)
        solver_name = config.get("solver", "minisat-gh")
        max_attempts = config.get("max_attempts", 10)
        target_count = max_circuits or 1

        started_at = datetime.now()
        circuits_found = 0
        generated_circuits = []

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
                f"SAT Forward/Inverse starting: {width}w x {gate_count}g, max={target_count}"
            )

            # Import core synthesis components directly
            import random

            from circuit.circuit import Circuit
            from sat.solver import Solver
            from synthesizers.circuit_synthesizer import CircuitSynthesizer
            from truth_table.truth_table import TruthTable

            solver = Solver(solver_name)

            for i in range(target_count):
                if self.is_cancel_requested():
                    break

                update_progress(f"Generating circuit {i+1}/{target_count}...")

                success = False
                for attempt in range(max_attempts):
                    try:
                        # Step 1: Generate random forward circuit
                        forward_circuit = Circuit(width)
                        for _ in range(gate_count):
                            # Random gate: X, CX, or CCX
                            r = random.random()
                            if r < 0.2:  # X gate
                                target = random.randint(0, width - 1)
                                forward_circuit.x(target)
                            elif r < 0.6 and width >= 2:  # CX gate
                                bits = random.sample(range(width), 2)
                                forward_circuit.cx(bits[0], bits[1])
                            elif width >= 3:  # CCX gate
                                bits = random.sample(range(width), 3)
                                forward_circuit.mcx(bits[:2], bits[2])
                            else:
                                target = random.randint(0, width - 1)
                                forward_circuit.x(target)

                        # Step 2: Get permutation and compute inverse
                        forward_perm = forward_circuit.tt().values()
                        n = len(forward_perm)
                        inverse_perm = [0] * n
                        for i_perm, v in enumerate(forward_perm):
                            inverse_perm[v] = i_perm

                        # Step 3: Synthesize inverse using SAT
                        inverse_tt = TruthTable(width, values=inverse_perm)

                        # Try different gate counts up to max_inverse_gates
                        inverse_circuit = None
                        for try_gates in range(1, min(max_inverse_gates, 15) + 1):
                            synth = CircuitSynthesizer(inverse_tt, try_gates, solver)
                            result = synth.solve()
                            if result is not None:
                                inverse_circuit = result
                                break

                        if inverse_circuit:
                            # Step 4: Combine forward + inverse
                            identity_circuit = forward_circuit + inverse_circuit

                            # Verify it's actually identity
                            result_perm = identity_circuit.tt().values()
                            if result_perm == list(range(n)):
                                circuits_found += 1
                                generated_circuits.append(
                                    {
                                        "width": width,
                                        "gates": identity_circuit.gates(),
                                        "forward_length": gate_count,
                                        "inverse_length": len(inverse_circuit),
                                    }
                                )
                                success = True
                                logger.info(
                                    f"Generated circuit {circuits_found}: {len(identity_circuit)} gates"
                                )
                                break

                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} failed: {e}")
                        continue

                if success:
                    update_progress(
                        f"Generated {circuits_found} circuits", circuits_found
                    )

            completed_at = datetime.now()
            total_seconds = (completed_at - started_at).total_seconds()

            logger.info(
                f"SAT Forward/Inverse completed: {circuits_found} circuits in {total_seconds:.2f}s"
            )

            self._status = (
                GeneratorStatus.COMPLETED
                if not self.is_cancel_requested()
                else GeneratorStatus.CANCELLED
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
                circuits=generated_circuits,  # Include generated circuit data
            )

        except Exception as e:
            logger.exception("SAT Forward/Inverse generation failed")
            self._status = GeneratorStatus.FAILED
            return GenerationResult(
                success=False,
                run_id=run_id,
                generator_name=self.get_info().name,
                error=str(e),
                width=width,
                gate_count=gate_count,
            )
