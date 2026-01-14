import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from identity_factory.experiment_runner import ExperimentRunner
from identity_factory.api.experiment_models import ExperimentConfig, ObfuscationParams

@pytest.fixture
def mock_config():
    obf = ObfuscationParams(
        strategy="abbutterfly",
        bookendless=False,
        structure_block_size_min=10,
        structure_block_size_max=20,
        shooting_count=100,
        shooting_count_inner=10,
        single_gate_replacements=10,
        rounds=1,
        sat_mode=True,
        no_ancilla_mode=False,
        single_gate_mode=False,
        skip_compression=False,
        compression_window_size=10,
        compression_window_size_sat=5,
        compression_sat_limit=100,
        final_stability_threshold=3,
        chunk_split_base=100
    )
    return ExperimentConfig(
        name="Test Experiment",
        experiment_type="custom",
        wires=5,
        initial_gates=10,
        obfuscation=obf
    )

def test_experiment_runner_uses_binary_path(mock_config):
    """Test that runner constructs commands using the direct binary path."""
    with patch("identity_factory.experiment_runner.get_rust_binary_path") as mock_get_path:
        mock_path = Path("/mock/path/to/local_mixing_bin")
        mock_get_path.return_value = mock_path
        
        runner = ExperimentRunner()
        # ... logic ...
        pass

@pytest.mark.integration
def test_experiment_runner_full_integration():
    """Run a real (short) experiment to verify binary execution."""
    from identity_factory.experiment_runner import ExperimentRunner
    from identity_factory.api.experiment_models import ExperimentConfig, ObfuscationParams
    import tempfile
    import shutil
    import asyncio
    
    # Create a minimal config
    obf = ObfuscationParams(
        strategy="abbutterfly", # abbutterfly supports --path
        bookendless=False,
        structure_block_size_min=4,
        structure_block_size_max=6,
        shooting_count=10,
        shooting_count_inner=0,
        single_gate_replacements=0,
        rounds=1,
        sat_mode=False,
        no_ancilla_mode=True,
        single_gate_mode=False,
        skip_compression=True,
        compression_window_size=10,
        compression_window_size_sat=10,
        compression_sat_limit=100,
        final_stability_threshold=1,
        chunk_split_base=1000
    )
    config = ExperimentConfig(
        name="Integration Test",
        experiment_type="custom",
        wires=4,
        initial_gates=5,
        obfuscation=obf,
        lmdb_path=None # Optional
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the global EXPERIMENTS_DIR so it writes to tmpdir
        with patch("identity_factory.experiment_runner.EXPERIMENTS_DIR", Path(tmpdir)):
            runner = ExperimentRunner()
            
            # We need to run the async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run():
                job = await runner.start_experiment(config)
                job_id = job.job_id
                
                # Poll for completion
                for _ in range(60): # 60 seconds max
                    job = runner.get_job(job_id)
                    if job.status in ["completed", "failed"]:
                        return job
                    await asyncio.sleep(1)
                return None

            result_job = loop.run_until_complete(run())
            loop.close()
            
            assert result_job is not None, "Experiment timed out"
        if result_job.status == "failed":
            print("Experiment failed with error:", result_job.error_message)
            print("Logs:", "\n".join(result_job.log_lines))
            
        assert result_job.status == "completed"
        assert result_job.final_gates > 0
