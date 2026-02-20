"""
Unified Circuit Format Utilities

This module provides conversion between all circuit representations
used across the codebase. The canonical format is:
- Text: semicolon-delimited 3-char tokens (e.g., "012;47d;c74;")
- Binary: [width:u16][gates:u32][hash:32][data:N*3]
- JSON: {"format":"eca57","gates":[[0,1,2],...]}

Gate order is always [target, ctrl1, ctrl2] (ECA57 semantics).
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

# Type aliases
Gate = Tuple[int, int, int]  # (target, ctrl1, ctrl2)
GateList = List[Gate]


# =============================================================================
# Wire Encoding (for .gate format)
# =============================================================================

def wire_to_char(wire: int) -> str:
    """Convert wire index to single character (supports 0-82+)."""
    if wire < 10:
        return chr(ord('0') + wire)
    elif wire < 36:
        return chr(ord('a') + wire - 10)
    elif wire < 62:
        return chr(ord('A') + wire - 36)
    elif wire < 83:
        special = "!@#$%^&*()-_=+[]{}<>?"
        return special[wire - 62]
    else:
        # Overflow: prefix with ~
        return '~' + wire_to_char(wire - 83)


def char_to_wire(c: str) -> Tuple[int, int]:
    """Convert character(s) to wire index. Returns (wire, chars_consumed)."""
    if c[0] == '~':
        inner, consumed = char_to_wire(c[1:])
        return (inner + 83, consumed + 1)

    if '0' <= c[0] <= '9':
        return (ord(c[0]) - ord('0'), 1)
    elif 'a' <= c[0] <= 'z':
        return (ord(c[0]) - ord('a') + 10, 1)
    elif 'A' <= c[0] <= 'Z':
        return (ord(c[0]) - ord('A') + 36, 1)
    else:
        special = "!@#$%^&*()-_=+[]{}<>?"
        idx = special.find(c[0])
        if idx >= 0:
            return (62 + idx, 1)
        raise ValueError(f"Invalid wire character: {c[0]}")


# =============================================================================
# Gate String Format (semicolon-delimited)
# =============================================================================

def gate_string_to_list(s: str) -> GateList:
    """
    Parse semicolon-delimited gate string to list of tuples.

    Example: "012;47d;" -> [(0,1,2), (4,7,13)]
    """
    gates = []
    s = s.strip().rstrip(';')
    if not s:
        return gates

    for token in s.split(';'):
        token = token.strip()
        if not token:
            continue

        # Parse 3 wire indices from token
        idx = 0
        wires = []
        while idx < len(token) and len(wires) < 3:
            wire, consumed = char_to_wire(token[idx:])
            wires.append(wire)
            idx += consumed

        if len(wires) != 3:
            raise ValueError(f"Invalid gate token: {token}")

        gates.append((wires[0], wires[1], wires[2]))

    return gates


def list_to_gate_string(gates: GateList) -> str:
    """
    Convert list of gate tuples to semicolon-delimited string.

    Example: [(0,1,2), (4,7,13)] -> "012;47d;"
    """
    tokens = []
    for target, ctrl1, ctrl2 in gates:
        token = wire_to_char(target) + wire_to_char(ctrl1) + wire_to_char(ctrl2)
        tokens.append(token)
    return ';'.join(tokens) + ';' if tokens else ''


# =============================================================================
# Binary Blob Format
# =============================================================================

def gates_to_blob(width: int, gates: GateList) -> bytes:
    """
    Convert gates to binary blob format.

    Format: [width:u16 LE][gate_count:u32 LE][hash:32 bytes][gates:N*3 bytes]
    """
    gate_string = list_to_gate_string(gates)
    hash_bytes = hashlib.sha256(gate_string.encode()).digest()

    header = struct.pack('<HI', width, len(gates))
    gate_bytes = b''.join(
        bytes([target, ctrl1, ctrl2]) for target, ctrl1, ctrl2 in gates
    )

    return header + hash_bytes + gate_bytes


def blob_to_gates(data: bytes) -> Tuple[int, GateList, str]:
    """
    Parse binary blob to (width, gates, hash_hex).

    Returns: (width, gate_list, hash_hex)
    """
    if len(data) < 38:
        raise ValueError(f"Blob too short: {len(data)} bytes")

    width, gate_count = struct.unpack('<HI', data[:6])
    hash_bytes = data[6:38]
    hash_hex = hash_bytes.hex()

    expected_size = 38 + gate_count * 3
    if len(data) < expected_size:
        raise ValueError(f"Blob truncated: expected {expected_size}, got {len(data)}")

    gates = []
    for i in range(gate_count):
        offset = 38 + i * 3
        target, ctrl1, ctrl2 = data[offset], data[offset+1], data[offset+2]
        gates.append((target, ctrl1, ctrl2))

    return width, gates, hash_hex


def raw_blob_to_gates(data: bytes) -> GateList:
    """
    Parse raw gate blob (no header, just N*3 bytes).
    Used for LMDB gate data without the full header.
    """
    if len(data) % 3 != 0:
        raise ValueError(f"Raw blob size {len(data)} not divisible by 3")

    gates = []
    for i in range(len(data) // 3):
        offset = i * 3
        target, ctrl1, ctrl2 = data[offset], data[offset+1], data[offset+2]
        gates.append((target, ctrl1, ctrl2))

    return gates


def gates_to_raw_blob(gates: GateList) -> bytes:
    """Convert gates to raw blob (no header)."""
    return b''.join(bytes([t, c1, c2]) for t, c1, c2 in gates)


# =============================================================================
# JSON Format
# =============================================================================

def gates_to_json(width: int, gates: GateList, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert gates to canonical JSON format.

    Returns dict with format, version, width, gates, hash, source.
    """
    gate_string = list_to_gate_string(gates)
    hash_hex = hashlib.sha256(gate_string.encode()).hexdigest()

    result = {
        "format": "eca57",
        "version": 1,
        "width": width,
        "gates": [list(g) for g in gates],
        "gateString": gate_string,
        "hash": hash_hex[:16],
    }
    if source:
        result["source"] = source

    return result


def json_to_gates(obj: Dict[str, Any]) -> Tuple[int, GateList]:
    """
    Parse JSON object to (width, gates).

    Handles multiple JSON formats:
    - Canonical: {"format":"eca57", "gates":[[0,1,2],...]}
    - Legacy: {"gates":[[0,1,2],...], "width":N}
    - Go export: [[0,1,2],[1,2,0],...] (just array)
    """
    if isinstance(obj, list):
        # Bare array format (Go export)
        gates = [tuple(g) for g in obj]
        width = max(max(g) for g in gates) + 1 if gates else 0
        return width, gates

    gates_data = obj.get("gates", [])
    gates = [(g[0], g[1], g[2]) for g in gates_data]

    width = obj.get("width", 0)
    if width == 0 and gates:
        width = max(max(g) for g in gates) + 1

    return width, gates


# =============================================================================
# .eca57 File Format
# =============================================================================

def write_eca57_file(path: Union[str, Path], width: int, gates: GateList,
                     source: Optional[str] = None) -> None:
    """
    Write canonical .eca57 file with header.

    Format:
        # ECA57 Circuit
        # width: 32
        # gates: 150
        # hash: a1b2c3d4...
        # source: local_mixing/experiments/...
        012;47d;c74;...
    """
    gate_string = list_to_gate_string(gates)
    hash_hex = hashlib.sha256(gate_string.encode()).hexdigest()

    lines = [
        "# ECA57 Circuit",
        f"# width: {width}",
        f"# gates: {len(gates)}",
        f"# hash: {hash_hex}",
    ]
    if source:
        lines.append(f"# source: {source}")
    lines.append(gate_string)

    Path(path).write_text('\n'.join(lines) + '\n')


def parse_eca57_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse .eca57 file with header.

    Returns: {width, gates, hash, source, gate_string}
    """
    content = Path(path).read_text()
    lines = content.strip().split('\n')

    metadata = {}
    gate_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            # Parse header
            if ':' in line:
                key, value = line[2:].split(':', 1)
                metadata[key.strip().lower()] = value.strip()
        elif line and not line.startswith('#'):
            gate_lines.append(line)

    gate_string = ''.join(gate_lines)
    gates = gate_string_to_list(gate_string)

    return {
        "width": int(metadata.get("width", 0)),
        "gates": gates,
        "hash": metadata.get("hash", ""),
        "source": metadata.get("source", ""),
        "gate_string": gate_string,
    }


def parse_gate_file(path: Union[str, Path]) -> Tuple[int, GateList]:
    """
    Parse any .gate or .eca57 file, auto-detecting format.

    Returns: (width, gates)
    Width is inferred from max wire index if not in header.
    """
    content = Path(path).read_text().strip()

    if content.startswith('#'):
        # Has header (.eca57 format)
        result = parse_eca57_file(path)
        return result["width"], result["gates"]
    else:
        # Raw .gate format
        gates = gate_string_to_list(content)
        width = max(max(g) for g in gates) + 1 if gates else 0
        return width, gates


# =============================================================================
# big_identities.txt Format
# =============================================================================

def parse_big_identities_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a line from big_identities.txt.

    Format: table=ids_n16g0, wires=16, gates=162, circuit=012;47d;...;

    Returns: {table, width, gate_count, gates, gate_string} or None if invalid
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    parts = {}
    for part in line.split(', '):
        if '=' in part:
            key, value = part.split('=', 1)
            parts[key.strip()] = value.strip()

    if 'circuit' not in parts:
        return None

    gate_string = parts['circuit']
    gates = gate_string_to_list(gate_string)

    return {
        "table": parts.get("table", ""),
        "width": int(parts.get("wires", 0)),
        "gate_count": int(parts.get("gates", len(gates))),
        "gates": gates,
        "gate_string": gate_string,
    }


# =============================================================================
# MCT Format Conversion (for sat_revsynth compatibility)
# =============================================================================

def mct_to_eca57(mct_gates: List[Tuple[List[int], int]]) -> GateList:
    """
    Convert MCT format to ECA57.

    MCT: [([controls], target), ...]
    ECA57: [(target, ctrl1, ctrl2), ...]

    Note: Only works for gates with exactly 0, 1, or 2 controls.
    For 0 controls (NOT gate), uses ctrl1=ctrl2=target (self-loop).
    For 1 control (CNOT), ctrl2=ctrl1 (duplicate).
    """
    eca57_gates = []
    for controls, target in mct_gates:
        if len(controls) == 0:
            # NOT gate: flip unconditionally
            # ECA57 hack: use impossible condition or identity
            eca57_gates.append((target, target, target))
        elif len(controls) == 1:
            # CNOT: ctrl1 OR NOT ctrl2 with ctrl1=ctrl2
            eca57_gates.append((target, controls[0], controls[0]))
        elif len(controls) == 2:
            # Toffoli: direct mapping
            eca57_gates.append((target, controls[0], controls[1]))
        else:
            raise ValueError(f"Cannot convert MCT gate with {len(controls)} controls to ECA57")

    return eca57_gates


def eca57_to_mct(gates: GateList) -> List[Tuple[List[int], int]]:
    """
    Convert ECA57 format to MCT.

    Note: ECA57 semantics (ctrl1 AND NOT ctrl2) differ from MCT (all controls high).
    This conversion is approximate and may not preserve semantics!
    """
    mct_gates = []
    for target, ctrl1, ctrl2 in gates:
        if ctrl1 == ctrl2 == target:
            # NOT gate
            mct_gates.append(([], target))
        elif ctrl1 == ctrl2:
            # CNOT
            mct_gates.append(([ctrl1], target))
        else:
            # Toffoli
            mct_gates.append(([ctrl1, ctrl2], target))

    return mct_gates


# =============================================================================
# Hash Computation
# =============================================================================

def compute_circuit_hash(gates: GateList) -> str:
    """Compute SHA-256 hash of canonical gate string."""
    gate_string = list_to_gate_string(gates)
    return hashlib.sha256(gate_string.encode()).hexdigest()


def compute_circuit_hash_short(gates: GateList, length: int = 16) -> str:
    """Compute truncated hash for display/dedup."""
    return compute_circuit_hash(gates)[:length]


# =============================================================================
# Circuit Search/Discovery
# =============================================================================

def find_circuit_files(root: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    Find all circuit files under a directory.

    Default extensions: .gate, .eca57
    """
    if extensions is None:
        extensions = ['.gate', '.eca57']

    root = Path(root)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f'*{ext}'))

    return sorted(files)


def index_circuit_directory(root: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Index all circuit files in a directory.

    Returns list of {path, width, gate_count, hash, gates}.
    """
    index = []
    for path in find_circuit_files(root):
        try:
            width, gates = parse_gate_file(path)
            index.append({
                "path": str(path),
                "width": width,
                "gate_count": len(gates),
                "hash": compute_circuit_hash_short(gates),
                "gates": gates,
            })
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")

    return index
