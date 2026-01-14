import pytest
from pathlib import Path
from identity_factory.local_mixing_utils import get_rust_binary_path, parse_circuit_string, gates_to_string

def test_rust_binary_path_resolution():
    """Verify that we can find the local_mixing binary."""
    path = get_rust_binary_path()
    assert path is not None, "Could not find local_mixing binary"
    assert path.exists(), f"Binary path {path} does not exist"
    assert path.is_file(), f"Binary path {path} is not a file"
    # Basic check that it looks like an executable (chmod +x nature usually)
    assert os.access(path, os.X_OK), "Binary is not executable"

import os
import subprocess

@pytest.mark.integration
def test_local_mixing_cli_version():
    """Verify we can run the binary and get help/version output."""
    bin_path = get_rust_binary_path()
    assert bin_path is not None
    
    # Run with --help
    result = subprocess.run(
        [str(bin_path), "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Rainbow circuit generator" in result.stdout

def test_circuit_parsing_roundtrip():
    """Test that parsing and stringifying is consistent."""
    original = "[0,1,2] [1,2,0] [2,0,1]"
    gates = parse_circuit_string(original)
    assert len(gates) == 3
    assert gates[0] == (0, 1, 2)
    
    reconstructed = gates_to_string(gates)
    assert reconstructed == original
