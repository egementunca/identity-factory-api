"""
Experiment Runner for Local Mixing experiments.

Manages background experiment jobs with:
- Subprocess execution of local_mixing CLI
- Real-time progress streaming
- Result collection and storage
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from identity_factory.local_mixing_utils import get_rust_binary_path
from identity_factory.api.experiment_models import (
    ExperimentConfig,
    ExperimentPreset,
    ExperimentProgress,
    ExperimentResults,
    ExperimentStatus,
    ExperimentType,
    ExperimentResults,
    ExperimentStatus,
    ExperimentType,
    ObfuscationParams,
    ObfuscationStrategy,
)

logger = logging.getLogger(__name__)

# Path to local_mixing project
LOCAL_MIXING_DIR = Path(__file__).parent.parent.parent / "local_mixing"
EXPERIMENTS_DIR = LOCAL_MIXING_DIR / "experiments"


@dataclass
class ExperimentJob:
    """Represents a running or completed experiment job."""

    job_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Runtime state
    process: Optional[subprocess.Popen] = None
    log_lines: List[str] = field(default_factory=list)
    current_round: int = 0
    current_gates: int = 0

    # Results
    initial_gates: int = 0
    final_gates: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None


class ExperimentRunner:
    """
    Manages experiment execution with background subprocess.
    """

    def __init__(self):
        self.jobs: Dict[str, ExperimentJob] = {}
        self._lock = asyncio.Lock()

    def get_presets(self) -> List[ExperimentPreset]:
        """Return pre-configured experiment presets matching research results."""
        return [
            # ======== RESEARCH ISSUE EXPERIMENTS ========
            # These presets replicate the experiments from the research heatmaps
            ExperimentPreset(
                id="research_5rounds_standard",
                name="Research: 3 Rounds Standard",
                description="64w 100g, 3 rounds abbutterfly (Local Friendly). "
                "Expected: ~500 gates. Good for local testing.",
                experiment_type=ExperimentType.EXPANSION,
                config=ExperimentConfig(
                    name="5R Standard 64w",
                    experiment_type=ExperimentType.EXPANSION,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=3,
                        shooting_count=500_000,
                        shooting_count_inner=0,
                        structure_block_size_min=10,
                        structure_block_size_max=30,
                        single_gate_replacements=0,  # No single gate replacements
                        no_ancilla_mode=False,  # Use ancilla expansion
                        skip_compression=False,
                        compression_window_size=100,
                        final_stability_threshold=12,
                        sat_mode=True,
                    ),
                ),
                tags=["research", "standard", "3-rounds"],
            ),
            ExperimentPreset(
                id="research_10rounds_standard",
                name="Research: 10 Rounds Standard",
                description="64w 100g, 10 rounds abbutterfly with ancilla expansion. "
                "Expected: ~1100 gates. Smaller 'red beam' than 5 rounds.",
                experiment_type=ExperimentType.EXPANSION,
                config=ExperimentConfig(
                    name="10R Standard 64w",
                    experiment_type=ExperimentType.EXPANSION,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=10,
                        shooting_count=500_000,
                        shooting_count_inner=0,
                        structure_block_size_min=10,
                        structure_block_size_max=30,
                        single_gate_replacements=0,
                        no_ancilla_mode=False,
                        skip_compression=False,
                        compression_window_size=100,
                        final_stability_threshold=12,
                    ),
                ),
                tags=["research", "standard", "10-rounds"],
            ),
            ExperimentPreset(
                id="research_30rounds_standard",
                name="Research: 30 Rounds Standard",
                description="64w 100g, 30 rounds abbutterfly with ancilla expansion. "
                "Expected: ~12000 gates. Much better mixing, slight red beam at very top.",
                experiment_type=ExperimentType.EXPANSION,
                config=ExperimentConfig(
                    name="30R Standard 64w",
                    experiment_type=ExperimentType.EXPANSION,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=30,
                        shooting_count=500_000,
                        shooting_count_inner=0,
                        structure_block_size_min=10,
                        structure_block_size_max=30,
                        single_gate_replacements=0,
                        no_ancilla_mode=False,
                        skip_compression=False,
                        compression_window_size=100,
                        final_stability_threshold=12,
                    ),
                ),
                tags=["research", "standard", "30-rounds", "high-memory"],
            ),
            ExperimentPreset(
                id="research_5rounds_gate_repl",
                name="Research: 5 Rounds + Gate Replacements",
                description="64w 100g, 5 rounds + single gate replacements (7x blowup per gate). "
                "Expected: ~50,000 gates. Shows blurry curve instead of red beam - better mixing!",
                experiment_type=ExperimentType.EXPANSION,
                config=ExperimentConfig(
                    name="5R GateRepl 64w",
                    experiment_type=ExperimentType.EXPANSION,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=5,
                        shooting_count=500_000,
                        shooting_count_inner=1_000,
                        structure_block_size_min=10,
                        structure_block_size_max=30,
                        single_gate_replacements=500,  # Enable single gate replacements
                        single_gate_mode=True,
                        no_ancilla_mode=False,
                        skip_compression=False,
                        compression_window_size=100,
                        final_stability_threshold=12,
                    ),
                ),
                tags=["research", "gate-replacements", "high-memory"],
            ),
            # ======== QUICK TEST EXPERIMENTS ========
            ExperimentPreset(
                id="quick_test_small",
                name="Quick Test (8w 20g)",
                description="Small circuit for quick testing. 8 wires, 20 gates, 2 rounds. "
                "Completes in seconds.",
                experiment_type=ExperimentType.EXPANSION,
                config=ExperimentConfig(
                    name="Quick Test",
                    experiment_type=ExperimentType.EXPANSION,
                    wires=8,
                    initial_gates=20,
                    obfuscation=ObfuscationParams(
                        rounds=2,
                        shooting_count=10_000,
                        shooting_count_inner=0,
                    ),
                ),
                tags=["quick", "testing"],
            ),
            ExperimentPreset(
                id="inflation_only",
                name="Inflation Only (No Compression)",
                description="Tests inflation without compression. Circuit only grows. "
                "Useful for measuring raw expansion factor.",
                experiment_type=ExperimentType.INFLATION_ONLY,
                config=ExperimentConfig(
                    name="Inflation Only",
                    experiment_type=ExperimentType.INFLATION_ONLY,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=1,
                        shooting_count=10_000,
                        skip_compression=True,
                    ),
                ),
                tags=["inflation", "no-compression"],
            ),
            ExperimentPreset(
                id="sat_compression",
                name="SAT Compression Mode",
                description="Uses SAT solver for compression instead of rainbow table. "
                "Slower but can find better reductions.",
                experiment_type=ExperimentType.SAT_COMPRESSION,
                config=ExperimentConfig(
                    name="SAT Mode",
                    experiment_type=ExperimentType.SAT_COMPRESSION,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=1,
                        shooting_count=10_000,
                        sat_mode=True,
                        compression_window_size_sat=10,
                        compression_sat_limit=1000,
                    ),
                ),
                tags=["sat", "compression"],
            ),
            ExperimentPreset(
                id="no_ancilla_mode",
                name="No Ancilla (In-Place Only)",
                description="Disables ancilla expansion. Works in-place without extra wires. "
                "Less effective mixing but lower expansion.",
                experiment_type=ExperimentType.CUSTOM,
                config=ExperimentConfig(
                    name="No Ancilla",
                    experiment_type=ExperimentType.CUSTOM,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        rounds=5,
                        shooting_count=100_000,
                        no_ancilla_mode=True,
                    ),
                ),
                tags=["no-ancilla", "in-place"],
            ),
            ExperimentPreset(
                id="rac_standard",
                name="RAC (Replace And Compress)",
                description="Uses RAC strategy for obfuscation. "
                "Different approach than butterfly - replaces and compresses gates iteratively.",
                experiment_type=ExperimentType.CUSTOM,
                config=ExperimentConfig(
                    name="RAC Standard",
                    experiment_type=ExperimentType.CUSTOM,
                    wires=64,
                    initial_gates=100,
                    obfuscation=ObfuscationParams(
                        strategy=ObfuscationStrategy.RAC,
                        rounds=5,
                        shooting_count=100_000,
                    ),
                ),
                tags=["rac", "replace-compress"],
            ),
        ]

    async def start_experiment(self, config: ExperimentConfig) -> ExperimentJob:
        """Start a new experiment job."""
        job_id = str(uuid.uuid4())[:8]

        job = ExperimentJob(
            job_id=job_id,
            config=config,
            status=ExperimentStatus.PENDING,
            started_at=datetime.now(),
        )

        async with self._lock:
            self.jobs[job_id] = job

        # Start the experiment in the background
        asyncio.create_task(self._run_experiment(job))

        return job

    async def _run_experiment(self, job: ExperimentJob):
        """Execute the experiment subprocess."""
        try:
            job.status = ExperimentStatus.RUNNING
            config = job.config

            # Create output directory
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_dir = EXPERIMENTS_DIR / date_str / f"{config.name}_{job.job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_dir = output_dir

            # Generate initial circuit
            initial_gate = output_dir / "initial.gate"
            
            bin_path = get_rust_binary_path()
            if not bin_path:
                raise RuntimeError("local_mixing binary not found. Please build it first.")

            gen_cmd = [
                str(bin_path),
                "gen",
                "--wires",
                str(config.wires),
                "--length",
                str(config.initial_gates),
            ]

            job.log_lines.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Generating initial circuit..."
            )

            result = await asyncio.to_thread(
                subprocess.run,
                gen_cmd,
                cwd=str(LOCAL_MIXING_DIR),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to generate circuit: {result.stderr}")

            # Write initial circuit
            with open(initial_gate, "w") as f:
                f.write(result.stdout)

            job.initial_gates = result.stdout.count(";")
            job.log_lines.append(f"  Generated {job.initial_gates} gates")

            # Build obfuscation config JSON (matches local_mixing ObfuscationConfig)
            obf = config.obfuscation
            obf_config = {
                "structure_block_size_min": obf.structure_block_size_min,
                "structure_block_size_max": obf.structure_block_size_max,
                "mix_prob_template": 0.40,
                "mix_prob_swaps": 0.50,
                "mix_prob_patch": 0.10,
                "single_gate_replacements": obf.single_gate_replacements,
                "shooting_count": obf.shooting_count,
                "shooting_count_inner": obf.shooting_count_inner,
                "rounds": obf.rounds,
                "sat_mode": obf.sat_mode,
                "no_ancilla_mode": obf.no_ancilla_mode,
                "single_gate_mode": obf.single_gate_mode,
                "skip_compression": obf.skip_compression,
                "compression_window_size": obf.compression_window_size,
                "compression_window_size_sat": obf.compression_window_size_sat,
                "compression_sat_limit": obf.compression_sat_limit,
                "final_stability_threshold": obf.final_stability_threshold,
                "chunk_split_base": obf.chunk_split_base,
                "reducer_active_wire_limit": 6,
                "reducer_window_sizes": [4, 6, 8, 10, 12, 16],
                "pair_replacement_mode": obf.pair_replacement_mode,
                "equal_replacement_mode": True,
                "lmdb_path": config.lmdb_path or "db",
            }

            obf_config_path = output_dir / "obf_config.json"
            with open(obf_config_path, "w") as f:
                json.dump(obf_config, f, indent=2)

            # Build CLI command with --config flag
            # Config file now provides all parameters (rounds, shooting, etc.)
            # CLI args only specify path, wires, and config location

            # Determine command based on strategy
            strategy = config.obfuscation.strategy
            command_name = strategy.value if hasattr(strategy, "value") else strategy

            # RAC uses different CLI args than butterfly strategies
            is_rac = command_name == "rac"
            obf_gate = output_dir / "obfuscated.gate"

            if is_rac:
                lmdb_dir = LOCAL_MIXING_DIR / "db"
                sqlite_path = lmdb_dir / "circuits.db"
                lmdb_data = lmdb_dir / "data.mdb"
                if not sqlite_path.exists():
                    raise RuntimeError(
                        "RAC requires SQLite db at local_mixing/db/circuits.db (rainbow tables)."
                    )
                if not lmdb_data.exists():
                    raise RuntimeError(
                        "RAC requires LMDB at local_mixing/db/data.mdb (ids_n*, ids_rev, perm tables)."
                    )

            if is_rac:
                # RAC command: rac --path <input> --rounds <rounds> --n <wires> --save <output>
                obf_cmd = [
                    str(bin_path),
                    "rac",
                    "--path",
                    str(initial_gate),
                    "--rounds",
                    str(config.obfuscation.rounds),
                    "-n",
                    str(config.wires),
                    "--save",
                    str(obf_gate),
                ]
            else:
                # HOTFIX: 'butterfly' subcommand does not support --path/--config args.
                # Map it to 'abbutterfly' which is the robust implementation.
                if command_name == "butterfly":
                    command_name = "abbutterfly"

                # Basic command structure for butterfly strategies
                obf_cmd = [
                    str(bin_path),
                    command_name,
                    "--path",
                    str(initial_gate),
                    "-n",
                    str(config.wires),
                    "--config",
                    str(obf_config_path),
                    "--rounds",
                    str(config.obfuscation.rounds),
                ]

                # Add strategy-specific flags
                if command_name == "abbutterfly" and config.obfuscation.bookendless:
                    obf_cmd.append("--bookendless")

                if config.lmdb_path:
                    # bbutterfly does not support lmdb-db flag in CLI args generally,
                    # but abbutterfly does. However, correct way is likely passed via config json or env?
                    # Based on analysis, abbutterfly takes --lmdb-db.
                    # bbutterfly takes env from config or default.
                    # Let's keep it safe: pass if supported.
                    if command_name == "abbutterfly":
                         obf_cmd.extend(["--lmdb-db", config.lmdb_path])
                    # For others, we assume config.json handles it via "lmdb_path" field.

            # Save config JSON for reproducibility
            config_json = output_dir / "config.json"
            with open(config_json, "w") as f:
                f.write(config.model_dump_json(indent=2))

            job.log_lines.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Starting obfuscation..."
            )
            job.log_lines.append(f"  Command: {' '.join(obf_cmd[:10])}...")

            # Run obfuscation with streaming output
            # Merge stderr into stdout so we can stream all output together
            # Limit Rayon parallelism to reduce memory usage for large-wire experiments
            import os

            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = "4"  # Limit parallel threads to prevent OOM

            process = await asyncio.create_subprocess_exec(
                *obf_cmd,
                cwd=str(LOCAL_MIXING_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                env=env,
            )
            job.process = process

            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                job.log_lines.append(line_str)

                # Parse progress from output
                if "Round" in line_str:
                    try:
                        parts = line_str.split("/")
                        if len(parts) >= 2:
                            job.current_round = int(parts[0].split()[-1])
                    except:
                        pass
                if "gates" in line_str.lower():
                    try:
                        # Try to extract gate count
                        import re

                        match = re.search(r"(\d+)\s*gates", line_str)
                        if match:
                            job.current_gates = int(match.group(1))
                    except:
                        pass

            await process.wait()

            if process.returncode != 0:
                # Log the exit code - all output has already been captured in log_lines
                job.log_lines.append(
                    f"ERROR: Process exited with code {process.returncode}"
                )
                raise RuntimeError(
                    f"Obfuscation failed with code {process.returncode}. Check log output above."
                )

            # Handle output based on strategy
            # RAC saves directly to obf_gate via --save; butterfly strategies use recent_circuit.txt
            if is_rac:
                # RAC already saved output to obf_gate
                if obf_gate.exists():
                    with open(obf_gate) as f:
                        job.final_gates = f.read().count(";")
                else:
                    job.final_gates = job.current_gates
            else:
                # Butterfly strategies write to recent_circuit.txt
                recent_circuit = LOCAL_MIXING_DIR / "recent_circuit.txt"
                if recent_circuit.exists():
                    shutil.move(str(recent_circuit), str(obf_gate))
                    with open(obf_gate) as f:
                        job.final_gates = f.read().count(";")
                else:
                    job.final_gates = job.current_gates

            job.log_lines.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Obfuscation complete"
            )
            job.log_lines.append(
                f"  Final: {job.final_gates} gates ({job.final_gates/max(job.initial_gates,1):.1f}x expansion)"
            )

            # Generate heatmap
            job.log_lines.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Generating heatmap..."
            )
            heatmap_cmd = [
                str(bin_path),
                "heatmap",
                "--c1",
                str(initial_gate),
                "--c2",
                str(obf_gate),
                "--num_wires",
                str(config.wires),
                "--inputs",
                "100",
            ]

            heatmap_result = await asyncio.to_thread(
                subprocess.run,
                heatmap_cmd,
                cwd=str(LOCAL_MIXING_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if heatmap_result.returncode == 0:
                heatmap_json = output_dir / "heatmap.json"
                with open(heatmap_json, "w") as f:
                    f.write(heatmap_result.stdout)
                job.log_lines.append("  Heatmap generated")

            # Generate alignment (DTW)
            if config.wires <= 64:
                job.log_lines.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Generating alignment..."
                )
                align_cmd = [
                    str(bin_path),
                    "align",
                    "--c1",
                    str(initial_gate),
                    "--c2",
                    str(obf_gate),
                    "--num_wires",
                    str(config.wires),
                    "--inputs",
                    "100",
                ]
                align_result = await asyncio.to_thread(
                    subprocess.run,
                    align_cmd,
                    cwd=str(LOCAL_MIXING_DIR),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if align_result.returncode == 0:
                    alignment_json = output_dir / "alignment.json"
                    with open(alignment_json, "w") as f:
                        f.write(align_result.stdout)
                    job.log_lines.append("  Alignment generated")
                else:
                    stderr = align_result.stderr.strip()
                    job.log_lines.append(
                        f"  Alignment failed{': ' + stderr if stderr else ''}"
                    )
            else:
                job.log_lines.append(
                    "  Alignment skipped (alignment supports <= 64 wires)"
                )

            # Save results
            results = {
                "job_id": job.job_id,
                "config": config.model_dump(),
                "initial_gates": job.initial_gates,
                "final_gates": job.final_gates,
                "expansion_factor": round(
                    job.final_gates / max(job.initial_gates, 1), 2
                ),
                "elapsed_seconds": (datetime.now() - job.started_at).total_seconds(),
            }

            results_json = output_dir / "results.json"
            with open(results_json, "w") as f:
                json.dump(results, f, indent=2)

            job.status = ExperimentStatus.COMPLETED
            job.completed_at = datetime.now()
            job.log_lines.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Experiment complete!"
            )

        except Exception as e:
            logger.exception(f"Experiment {job.job_id} failed")
            job.status = ExperimentStatus.FAILED
            job.error_message = str(e)
            job.log_lines.append(f"ERROR: {e}")
            job.completed_at = datetime.now()

    def get_job(self, job_id: str) -> Optional[ExperimentJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_progress(self, job: ExperimentJob) -> ExperimentProgress:
        """Get current progress for a job."""
        elapsed = (datetime.now() - job.started_at).total_seconds()

        # Estimate progress
        if job.status == ExperimentStatus.COMPLETED:
            progress_percent = 100.0
        elif job.status == ExperimentStatus.FAILED:
            progress_percent = 0.0
        elif job.config.obfuscation.rounds > 0:
            progress_percent = min(
                99.0, (job.current_round / job.config.obfuscation.rounds) * 100
            )
        else:
            progress_percent = 50.0 if job.status == ExperimentStatus.RUNNING else 0.0

        return ExperimentProgress(
            job_id=job.job_id,
            status=job.status,
            progress_percent=progress_percent,
            current_round=job.current_round,
            total_rounds=job.config.obfuscation.rounds,
            current_gates=job.current_gates,
            elapsed_seconds=elapsed,
            log_lines=job.log_lines[-50:],  # Last 50 lines
        )

    def get_results(self, job: ExperimentJob) -> Optional[ExperimentResults]:
        """Get results for a completed job."""
        if job.status not in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            return None

        # Try to load heatmap data
        heatmap_data = None
        heatmap_x = None
        heatmap_y = None
        alignment_c_star = None
        alignment_path = None
        alignment_matrix = None

        if job.output_dir:
            heatmap_json = job.output_dir / "heatmap.json"
            if heatmap_json.exists():
                try:
                    with open(heatmap_json) as f:
                        data = json.load(f)
                        heatmap_data = data.get("heatmap_data")
                        heatmap_x = data.get("x_size")
                        heatmap_y = data.get("y_size")
                except:
                    pass
            alignment_json = job.output_dir / "alignment.json"
            if alignment_json.exists():
                try:
                    with open(alignment_json) as f:
                        data = json.load(f)
                        alignment_c_star = data.get("c_star")
                        alignment_path = data.get("path")
                        alignment_matrix = data.get("d_matrix")
                except:
                    pass

        elapsed = (job.completed_at or datetime.now()) - job.started_at

        return ExperimentResults(
            job_id=job.job_id,
            status=job.status,
            config=job.config,
            started_at=job.started_at,
            completed_at=job.completed_at,
            elapsed_seconds=elapsed.total_seconds(),
            initial_gates=job.initial_gates,
            final_gates=job.final_gates,
            expansion_factor=round(job.final_gates / max(job.initial_gates, 1), 2),
            heatmap_data=heatmap_data,
            heatmap_x_size=heatmap_x,
            heatmap_y_size=heatmap_y,
            alignment_c_star=alignment_c_star,
            alignment_path=alignment_path,
            alignment_matrix=alignment_matrix,
            output_circuit_path=(
                str(job.output_dir / "obfuscated.gate") if job.output_dir else None
            ),
            results_json_path=(
                str(job.output_dir / "results.json") if job.output_dir else None
            ),
            log_output="\n".join(job.log_lines),
        )

    async def stream_progress(self, job_id: str) -> AsyncGenerator[str, None]:
        """Stream progress updates as Server-Sent Events."""
        job = self.get_job(job_id)
        if not job:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        last_line_count = 0

        while job.status in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING):
            progress = self.get_progress(job)

            # Only send if there are new log lines
            if len(job.log_lines) > last_line_count:
                new_lines = job.log_lines[last_line_count:]
                last_line_count = len(job.log_lines)

                update = {
                    "status": progress.status.value,
                    "progress_percent": progress.progress_percent,
                    "current_round": progress.current_round,
                    "current_gates": progress.current_gates,
                    "elapsed_seconds": progress.elapsed_seconds,
                    "new_lines": new_lines,
                }
                yield f"data: {json.dumps(update)}\n\n"

            await asyncio.sleep(0.5)

        # Send final update
        progress = self.get_progress(job)
        
        # Check for any remaining logs
        new_lines = []
        if len(job.log_lines) > last_line_count:
            new_lines = job.log_lines[last_line_count:]
            
        final_update = {
            "status": progress.status.value,
            "progress_percent": (
                100.0 if job.status == ExperimentStatus.COMPLETED else 0.0
            ),
            "final": True,
            "new_lines": new_lines,
        }
        yield f"data: {json.dumps(final_update)}\n\n"


# Global instance
experiment_runner = ExperimentRunner()
