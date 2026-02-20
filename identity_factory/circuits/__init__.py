"""
Circuit format utilities for the identity factory.

Provides unified conversion between all circuit representations.
"""

from .formats import (
    # Types
    Gate,
    GateList,

    # Wire encoding
    wire_to_char,
    char_to_wire,

    # Gate string format
    gate_string_to_list,
    list_to_gate_string,

    # Binary blob format
    gates_to_blob,
    blob_to_gates,
    raw_blob_to_gates,
    gates_to_raw_blob,

    # JSON format
    gates_to_json,
    json_to_gates,

    # File I/O
    write_eca57_file,
    parse_eca57_file,
    parse_gate_file,
    parse_big_identities_line,

    # MCT conversion
    mct_to_eca57,
    eca57_to_mct,

    # Hashing
    compute_circuit_hash,
    compute_circuit_hash_short,

    # Discovery
    find_circuit_files,
    index_circuit_directory,
)

__all__ = [
    'Gate',
    'GateList',
    'wire_to_char',
    'char_to_wire',
    'gate_string_to_list',
    'list_to_gate_string',
    'gates_to_blob',
    'blob_to_gates',
    'raw_blob_to_gates',
    'gates_to_raw_blob',
    'gates_to_json',
    'json_to_gates',
    'write_eca57_file',
    'parse_eca57_file',
    'parse_gate_file',
    'parse_big_identities_line',
    'mct_to_eca57',
    'eca57_to_mct',
    'compute_circuit_hash',
    'compute_circuit_hash_short',
    'find_circuit_files',
    'index_circuit_directory',
]
