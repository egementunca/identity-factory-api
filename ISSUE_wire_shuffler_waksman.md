# Wire Shuffler: Synthesizing Wire Permutation Circuits

## Overview

This project synthesizes **reversible circuits that permute wires** - given an input where wire i has value x[i], the output places x[i] on wire w[i] for some permutation w. These circuits are useful building blocks for obfuscation.

### What Problem Does This Solve?

Wire permutations serve multiple purposes in reversible circuit design:
1. **Obfuscation** - Reorder wires to obscure the circuit's structure
2. **Layout optimization** - Rearrange wires for better physical placement
3. **Composability** - Connect subcircuits that expect different wire orderings

The challenge: how do you build a circuit that implements an arbitrary wire permutation using only our available gates (ECA57)?

---

## Status (2026-02-06)

- `sat_revsynth`: added swap-based Waksman-style synthesis + identity filler + tests (commit `9ab473c`)
- `identity-factory-api`: added `/api/v1/waksman/*` endpoints + `waksman_circuits` SQLite table (commit `7b42b87`)
- `identity-factory-ui`: fixed `/playground-v2` build + refactored pages + added shared circuit utilities (commit `d666dff`)
- Shareable export for merge: `share/waksman_circuits_export_2026-02-06.json` + `share/import_waksman_circuits.py`
- Optional issue attachments (screenshots/logs): `share/issue_assets/waksman_db_summary.png`, `share/issue_assets/waksman_circuits_verify.png`

---

## Share / Merge Notes (for publishing)

### What’s safe to share

- The share package contains (small enough to publish):
  - `share/waksman_circuits_export_2026-02-06.json` (merge-only JSON export)
  - `share/wire_shuffler.db` (SQLite, includes `waksman_circuits`)
  - `share/wire_shuffler.lmdb/` (LMDB mirror)

### Verification (2026-02-06)

From `share/wire_shuffler.db`:
- `waksman_circuits`: 4 rows
- All `verify_ok = 1`

Summary screenshots: `share/issue_assets/waksman_db_summary.png`, `share/issue_assets/waksman_circuits_verify.png`

### Import into another `wire_shuffler.db`

```bash
python3 share/import_waksman_circuits.py \
  --json share/waksman_circuits_export_2026-02-06.json \
  --db-path /path/to/identity-factory-api/wire_shuffler.db
```

---

## Key Concepts

### Wire Permutation
A wire permutation is a bijection that maps each input wire position to an output wire position:
```
Input:   x[0]  x[1]  x[2]  x[3]
           ↓     ↓     ↓     ↓
         ┌─┴─────┴─────┴─────┴─┐
         │   Permutation w     │
         └─┬─────┬─────┬─────┬─┘
           ↓     ↓     ↓     ↓
Output: x[w[0]] x[w[1]] x[w[2]] x[w[3]]
```

Example: permutation `[2, 0, 1]` on 3 wires means:
- Output wire 0 gets input wire 2's value
- Output wire 1 gets input wire 0's value
- Output wire 2 gets input wire 1's value

### Cycle Type
Every permutation can be decomposed into cycles. For example:
- `[1, 0, 2]` = one 2-cycle (swap positions 0,1) + one fixed point (2)
- `[2, 0, 1]` = one 3-cycle (0→2→1→0)

Cycle type affects circuit complexity and is useful for categorizing permutations.

### Swap-Space Exclusion
We exclude the trivial "swap space" - permutations achievable by just relabeling wires without any computation. This focuses on permutations that require actual circuit logic.

---

## Current Implementation: SAT Synthesis

### How It Works

The script builds a SAT problem that encodes:
1. **Truth table constraints** - The circuit must implement permutation w
2. **Gate constraints** - Only valid ECA57 gates allowed
3. **Size bounds** - Find a circuit with at most N gates

The SAT solver finds a satisfying assignment = a valid circuit.

### Synthesis Script

**Location:** `sat_revsynth/scripts/synth_wire_shuffle.py`

**Features:**
- Synthesize circuit for a specific permutation
- Enumerate all permutations for a given width
- Solver racing (try multiple SAT solvers in parallel)
- Verification of synthesized circuits
- Cycle-type enumeration for organized generation

### Usage Examples

```bash
# Synthesize a specific 3-wire permutation [2,0,1]
python sat_revsynth/scripts/synth_wire_shuffle.py \
  --width 3 --perm 2,0,1 --max-gates 6 --verify

# Enumerate all permutations for 4 wires
python sat_revsynth/scripts/synth_wire_shuffle.py \
  --width 4 --all-perms --max-gates 8 --verify

# Generate with cycle-type organization
python sat_revsynth/scripts/synth_wire_shuffle.py \
  --width 5 --by-cycle-type --max-gates 10
```

---

## What's Implemented

### Synthesis and Storage

| Component | Location | Description |
|-----------|----------|-------------|
| Synthesis Script | `sat_revsynth/scripts/synth_wire_shuffle.py` | SAT-based circuit synthesis |
| Gate Semantics | `sat_revsynth/src/gates/eca57.py` | ECA57 gate definition and truth table |

### Database and API

| Component | Location | Description |
|-----------|----------|-------------|
| DB Schema | `identity-factory-api/identity_factory/wire_shuffler_db.py` | SQLite schema for storing circuits |
| Import Script | `identity-factory-api/scripts/import_wire_shuffler.py` | Bulk import synthesized circuits |
| Metrics Backfill | `identity-factory-api/scripts/backfill_wire_shuffler_metrics.py` | Compute metrics for existing circuits |
| API Endpoints | `identity-factory-api/identity_factory/api/wire_shuffler_endpoints.py` | REST API for querying |

### Web Interface

| Component | Location | Description |
|-----------|----------|-------------|
| Explorer Page | `identity-factory-ui/src/app/wire-shuffler/page.tsx` | Browse by width/cycle-type/gate count |

**UI Features:**
- Filter by width, cycle type, gate count
- View circuit details and metrics
- Export circuits for use in experiments

## Limitations of SAT Approach

SAT synthesis **does not scale** to larger widths:

| Width | Permutations | SAT Feasibility |
|-------|--------------|-----------------|
| 3 | 6 | Easy |
| 4 | 24 | Easy |
| 5 | 120 | Moderate |
| 6 | 720 | Slow |
| 7+ | 5040+ | Impractical |

The exponential growth of the search space makes SAT infeasible for large permutations.

---

## Waksman Network (Implemented)

### The Scaling Solution

Instead of SAT, use **rearrangeable permutation networks** - structured networks that can implement ANY permutation with predictable size.

### What is a Waksman Network?

A Waksman network is a recursive construction of 2x2 switches that can route any permutation:

```
        ┌─────┐     ┌─────┐     ┌─────┐
    ──►│ 2x2 │──►  │     │  ──►│ 2x2 │──►
        │switch│     │     │     │switch│
    ──►│     │──►  │Recur-│  ──►│     │──►
        └─────┘     │sive  │     └─────┘
                    │Waksman
    ──►│ 2x2 │──►  │     │  ──►│ 2x2 │──►
        │switch│     │     │     │switch│
    ──►│     │──►  │     │  ──►│     │──►
        └─────┘     └─────┘     └─────┘
```

**Key properties:**
- **Complexity:** O(n log n) switches for n wires
- **Deterministic:** No search required - routing algorithm is known
- **Diverse:** Multiple valid routings exist for each permutation

### How This Works for Us

1. **Route the permutation** using a swap-based network (selection-sort style).
2. **Compile each swap** to ECA57 gates (existing 6-gate swap gadget).
3. **(Optional) Obfuscate** identity slots with random identity circuits to hide structure.

**Benefits:**
- Scales to any width
- Predictable circuit size
- Multiple circuits per permutation (diversity)

### Status

**IMPLEMENTED (Waksman-style, swap-based)** - The current implementation uses
`SimpleSwapNetwork` for correctness and scalability. A true Waksman routing
algorithm (fixed topology with recursive routing) is still a future upgrade.

---

## File Reference

```
sat_revsynth/
├── scripts/
│   └── synth_wire_shuffle.py              # SAT synthesis script
│   └── synth_waksman.py                    # Waksman-style (swap-based) synthesis
└── src/
    └── gates/
        └── eca57.py                       # Gate definition
    └── synthesizers/
        ├── waksman.py                     # Waksman-style network + swap macro
        └── identity_filler.py             # Identity fillers for obfuscation

identity-factory-api/
├── identity_factory/
│   ├── wire_shuffler_db.py                # Database schema
│   └── api/
│       └── wire_shuffler_endpoints.py     # REST API
│       └── waksman_endpoints.py           # Waksman-style generation endpoints
└── scripts/
    ├── import_wire_shuffler.py            # Bulk import
    └── backfill_wire_shuffler_metrics.py  # Metrics computation

identity-factory-ui/
└── src/app/wire-shuffler/
    └── page.tsx                           # Explorer UI
```

---

## Next Steps

1. **Short term:** Continue using SAT for widths 3-6
2. **Medium term:** Implement true Waksman/Beneš routing (fixed topology)
3. **Long term:** Use topology-based generation for all permutation needs

---

## Questions?

For synthesis issues, check the `--verify` flag output. For understanding the gate semantics, see `sat_revsynth/src/gates/eca57.py`.
