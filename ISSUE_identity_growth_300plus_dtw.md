# Identity Growth Experiments: Analyzing Long Identity Circuits with DTW Alignment

## Overview

This project provides tools for **analyzing how identity circuits evolve** as gates are inserted or removed. The core question: when you have two related circuits (e.g., a circuit and its inverse, or before/after adding identity insertions), how do their internal states align?

### Why This Matters

In reversible circuit obfuscation, we often insert identity sequences (circuits that compute the identity function) to increase complexity. But we need to understand:
- How does the circuit's "state trajectory" change with insertions?
- Can we detect patterns that reveal the obfuscation strategy?
- How similar are two circuits structurally, beyond just gate count?

This tooling answers these questions through **state-space analysis** and **Dynamic Time Warping (DTW) alignment**.

---

## Status (2026-02-06)

- The core tooling is in `local_mixing/` + API/UI wrappers in `identity-factory-*`.
- The local DB artifacts used for these experiments (`local_mixing/db/*`) are **too large to publish** (SQLite + LMDB are tens of GB). Keep experiments reproducible via scripts/commands.
- Optional issue attachments (example outputs): `share/issue_assets/dtw_heatmap_wire_shuffle_align_8w_insert_mix_2026-02-03.png`, `share/issue_assets/dtw_alignment_wire_shuffle_align_8w_insert_mix_2026-02-03.png`

---

## Key Concepts

### State Trajectory
As a circuit executes gate-by-gate, it transforms an input state through intermediate states. We can represent this as a trajectory through state space. Two circuits implementing the same function may take very different paths.

### Dynamic Time Warping (DTW)
DTW is an algorithm that finds the optimal alignment between two sequences that may vary in speed or length. For circuits:
- Compare the state trajectory of circuit A vs circuit B
- Find which gates in A "correspond to" which gates in B
- Measure structural similarity even when circuits have different lengths

### Heatmaps
Visualize the pairwise distances between states at each step of two circuits. Bright spots indicate similar states; the DTW path shows the optimal alignment through this distance matrix.

---

## What's Implemented

### Analysis Commands (Rust CLI in `local_mixing/`)

| Command | File | Description |
|---------|------|-------------|
| `heatmap` | `local_mixing/src/main.rs` | Generate distance matrix between two circuit trajectories |
| `align` | `local_mixing/src/main.rs` | Compute DTW alignment path and similarity score |

**Core implementation:** `local_mixing/src/analysis/alignment/mod.rs`
- State tracing through circuit execution
- Distance matrix computation
- DTW algorithm implementation

### Visualization Scripts (Python)

| Script | Purpose |
|--------|---------|
| `local_mixing/scripts/plot_heatmap.py` | Render distance matrix as heatmap image |
| `local_mixing/scripts/plot_alignment.py` | Visualize DTW alignment path overlay |

### Experiment Integration

| Component | Location | Purpose |
|-----------|----------|---------|
| Experiment Runner | `identity-factory-api/identity_factory/experiment_runner.py` | Batch execution of alignment experiments |
| UI | `identity-factory-ui/src/app/experiments/page.tsx` | Web interface for running and viewing experiments |

---

## Interpretation

- Heatmap diagonal = circuits progress in sync
- Off-diagonal bright spots = similar states at different gate positions
- DTW path = optimal warping between the two trajectories

---

## Quick Start

```bash
# Generate a heatmap comparing two circuits
cd local_mixing
cargo run -- heatmap --c1 path/to/circuit_a.gate --c2 path/to/circuit_b.gate --num_wires N --inputs K > heatmap.json

# Compute DTW alignment
cargo run -- align --c1 path/to/circuit_a.gate --c2 path/to/circuit_b.gate --num_wires N --inputs K > alignment.json

# Visualize results
python scripts/plot_heatmap.py heatmap.json -o heatmap.png
python scripts/plot_alignment.py alignment.json -o alignment.png
```

---

## Current Research Goals

### Primary: Long Identity Analysis (300+ gates)
Compare long identity circuits to their inverses to understand:
- Does the forward/reverse trajectory show characteristic patterns?
- What's the "mixing rate" - how quickly do states diverge from a baseline?

## File Reference

```
local_mixing/
├── src/
│   ├── main.rs                          # CLI entrypoints (heatmap, align commands)
│   └── analysis/
│       └── alignment/
│           └── mod.rs                   # Core DTW and distance matrix logic
├── scripts/
│   ├── plot_heatmap.py                  # Heatmap visualization
│   └── plot_alignment.py                # Alignment path visualization

identity-factory-api/
└── identity_factory/
    └── experiment_runner.py             # Batch experiment execution

identity-factory-ui/
└── src/app/experiments/
    └── page.tsx                         # Experiments UI
```

---

## Questions?

If something is unclear or you need help running experiments, check the CLI help (`cargo run -- --help`) or ask in the group chat.
