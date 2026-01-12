"""
Go Enumerator Generator.
Wraps the Go-based exhaustive circuit enumeration.
"""

import json
import logging
import os
import re
import struct
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import (
    CircuitGenerator,
    GenerationProgress,
    GenerationResult,
    GeneratorInfo,
    GeneratorStatus,
)

logger = logging.getLogger(__name__)


class GoEnumeratorGenerator(CircuitGenerator):
    """
    Generator using Go-based exhaustive circuit enumeration.

    This wraps the go-proj binary for parallel circuit generation.
    """

    GO_PROJ_DIR = Path(__file__).parent.parent.parent / "go-proj"

    def get_info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="go_enumerator",
            display_name="Go Exhaustive Enumerator",
            description="High-performance Go-based exhaustive enumeration of all circuits. Generates complete database for given wire/gate count.",
            gate_sets=["mcx"],  # Toffoli-based
            supports_pause=True,
            supports_incremental=False,
            config_schema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "default": "./db",
                        "description": "Directory to store output .gob files",
                    },
                    "load_existing": {
                        "type": "string",
                        "description": "Load from existing database file",
                    },
                    "memory_limit_mb": {
                        "type": "integer",
                        "default": 8192,
                        "description": "Memory limit in MB",
                    },
                },
            },
        )

    def supports_gate_set(self, gate_set: str) -> bool:
        return gate_set.lower() == "mcx"

    def _ensure_binary(self) -> Optional[Path]:
        """Ensure Go binary is compiled and return path."""
        binary_path = self.GO_PROJ_DIR / "circuit-enum"

        if not binary_path.exists():
            logger.info("Go binary not found, attempting to compile...")
            # Check if Go is available
            try:
                go_check = subprocess.run(
                    ["go", "version"], capture_output=True, text=True, timeout=10
                )
                if go_check.returncode != 0:
                    logger.error("Go is not installed or not in PATH")
                    return None
            except Exception as e:
                logger.error(f"Go check failed: {e}")
                return None

            # Compile main.go directly (the main file is in the root)
            logger.info(
                f"Compiling Go enumerator from {self.GO_PROJ_DIR / 'main.go'}..."
            )
            try:
                result = subprocess.run(
                    ["go", "build", "-o", "circuit-enum", "main.go"],
                    cwd=self.GO_PROJ_DIR,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.error(f"Go build failed: {result.stderr}")
                    return None
                logger.info("Go binary compiled successfully")
            except subprocess.TimeoutExpired:
                logger.error("Go build timed out")
                return None
            except Exception as e:
                logger.error(f"Failed to compile Go binary: {e}")
                return None

        if binary_path.exists():
            return binary_path
        return None

    def generate(
        self,
        width: int,
        gate_count: int,
        gate_set: str = "mcx",
        max_circuits: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> GenerationResult:
        """Generate circuits using Go exhaustive enumeration."""

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
        output_dir = config.get("output_dir", str(self.GO_PROJ_DIR / "db"))
        load_existing = config.get("load_existing")

        started_at = datetime.now()
        circuits_found = 0
        perms_found = 0

        try:
            binary_path = self._ensure_binary()
            if not binary_path:
                return GenerationResult(
                    success=False,
                    run_id=run_id,
                    generator_name=self.get_info().name,
                    error="Could not find or compile Go binary",
                )

            # Build command
            cmd = [str(binary_path), "-n", str(width), "-m", str(gate_count)]
            if load_existing:
                cmd.extend(["-load", load_existing])
            else:
                cmd.append("-new")

            # Ensure output dir exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Update progress callback
            def update_progress(status_msg: str, circuits: int = 0, perms: int = 0):
                nonlocal circuits_found, perms_found
                circuits_found = circuits
                perms_found = perms
                elapsed = (datetime.now() - started_at).total_seconds()
                progress = GenerationProgress(
                    run_id=run_id,
                    generator_name=self.get_info().name,
                    status=GeneratorStatus.RUNNING,
                    circuits_found=circuits_found,
                    circuits_stored=perms_found,
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

            update_progress(f"Starting Go enumeration for {width}w x {gate_count}g...")

            # Run Go binary
            process = subprocess.Popen(
                cmd,
                cwd=self.GO_PROJ_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Parse output for progress
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())

                if self.is_cancel_requested():
                    process.terminate()
                    break

                # Parse progress from Go output
                # Example: "@ 1.2M, now 45.3k/s, avg 42.1k/s ETA: 120 sec (mem: 2048 MiB)"
                if line.startswith("@"):
                    match = re.search(r"@ ([\d.]+)M", line)
                    if match:
                        circuits = int(float(match.group(1)) * 1_000_000)
                        update_progress(line.strip(), circuits=circuits)

                # Parse final stats
                # Example: "Canonical perms: 12345 (100.0x smaller)"
                if "Canonical perms:" in line:
                    match = re.search(r"Canonical perms: (\d+)", line)
                    if match:
                        perms_found = int(match.group(1))

            process.wait()

            completed_at = datetime.now()
            total_seconds = (completed_at - started_at).total_seconds()

            success = process.returncode == 0 or self.is_cancel_requested()
            self._status = (
                GeneratorStatus.COMPLETED
                if success and not self.is_cancel_requested()
                else (
                    GeneratorStatus.CANCELLED
                    if self.is_cancel_requested()
                    else GeneratorStatus.FAILED
                )
            )

            return GenerationResult(
                success=success,
                run_id=run_id,
                generator_name=self.get_info().name,
                total_circuits=circuits_found,
                new_circuits=perms_found,
                duplicates=circuits_found - perms_found,
                width=width,
                gate_count=gate_count,
                gate_set=gate_set,
                config=config,
                started_at=started_at,
                completed_at=completed_at,
                total_seconds=total_seconds,
            )

        except Exception as e:
            logger.exception("Go enumeration failed")
            self._status = GeneratorStatus.FAILED
            return GenerationResult(
                success=False,
                run_id=run_id,
                generator_name=self.get_info().name,
                error=str(e),
                width=width,
                gate_count=gate_count,
            )
