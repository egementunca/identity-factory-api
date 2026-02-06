# Skeleton Chain Synthesis: Building Non-Commuting Gate Sequences with ECA57

## Overview

This project generates **skeleton chains** - sequences of gates where every adjacent pair of gates "collides" (shares a wire and doesn't commute). These chains form the backbone for identity circuit obfuscation.

### What Problem Does This Solve?

When obfuscating reversible circuits, we want to insert identity sequences that:
1. **Can't be trivially simplified** - adjacent gates shouldn't cancel or commute
2. **Have structural diversity** - many different valid chains exist
3. **Are verifiably correct** - the chain actually computes the identity function

Skeleton chains provide this by construction: they're maximally non-commuting sequences that wrap around to form identity circuits.

---

## Status (2026-02-06)

- Verified `ids_n4`..`ids_n7` in `local_mixing/db` are **identity** and **fully noncommuting** (all adjacent pairs collide).
- Exported a shareable slice: `share/skeleton_ids_n4_n7.lmdb/` + merge helper `share/merge_skeleton_identities_lmdb.py`.
- Fixed `local_mixing` LMDB open limit for `abbutterfly` (`DbsFull` when opening many named DBs) by increasing `set_max_dbs` (commit `f53f39a`).
- Optional issue attachments (screenshots/logs): `share/issue_assets/skeleton_db_stats.png`, `share/issue_assets/skeleton_identity_verify.png`, `share/issue_assets/skeleton_graph_example_2026-01-06.png`

---

## Key Concepts

### ECA57 (Elementary Cellular Automaton Rule 57)
ECA57 is a specific 3-bit reversible gate we use as our primitive. It acts on 3 adjacent wires and is:
- **Reversible** - every output maps back to a unique input
- **Universal** - can build any reversible function with enough gates
- **Self-inverse** - applying the same gate twice returns to the original state

### Gate Collision
Two gates "collide" if they share at least one wire AND don't commute (order matters). For skeleton chains, we require **all adjacent pairs to collide** - this prevents local simplification.

### Skeleton Chain Structure
```
Gate 1 -- collides with -- Gate 2 -- collides with -- Gate 3 -- ... -- Gate N
   └─────────────────────────── collides with (cyclic) ─────────────────────┘
```

Optional: for some workflows you may also require a *cyclic* collision between the last and first gate (“cyclic closure”). This is **not** required for identity correctness and is **not** enforced in all datasets.

---

## What's Implemented

### Core Synthesis Engine

| Component | Location | Description |
|-----------|----------|-------------|
| Synthesizer | `sat_revsynth/src/synthesizers/eca57_skeleton_synthesizer.py` | SAT-based skeleton chain generation |
| API Helpers | `sat_revsynth/src/synthesizers/skeleton_chain_api.py` | Programmatic interface for synthesis |
| Database Schema | `sat_revsynth/src/database/skeleton_db.py` | Storage format for generated chains |

The synthesizer uses **SAT constraints** to find gate sequences that:
- Compute the identity function
- Have all adjacent gates colliding
- Meet size requirements

### Scripts and Tools

| Script | Purpose |
|--------|---------|
| `sat_revsynth/scripts/generate_skeleton_chains.py` | Batch generation of chains |
| `sat_revsynth/scripts/verify_skeleton.py` | Verify a chain is valid (identity + collisions) |
| `sat_revsynth/scripts/explore_skeleton_db.py` | Browse and query the database |
| `sat_revsynth/examples/skeleton_obfuscation_pipeline.py` | End-to-end obfuscation demo |

### Web Interface

| Component | Location | Description |
|-----------|----------|-------------|
| Explorer Page | `identity-factory-ui/src/app/skeleton-explorer/page.tsx` | Browse chains by width/size |
| Graph Component | `identity-factory-ui/src/components/SkeletonGraph.tsx` | Visualize chain structure |

---

## Current Database Stats

Run `python3 sat_revsynth/scripts/explore_skeleton_db.py --stats` to get current counts.

**Important:** `explore_skeleton_db.py` defaults to `local_mixing/db` (the full local LMDB). If you want to query a different LMDB, pass `--db-path <path>`.

**Note:** n=3 (3-wire) chains haven't been generated yet.

---

## Share / Merge Notes (for publishing)

### What’s safe to share

- The full `local_mixing/db` is **too large** to publish (tens of GB).
- A small extracted LMDB containing only skeleton identity DBs is available at:
  - `share/skeleton_ids_n4_n7.lmdb/` (≈ 1.0 MB `data.mdb`)
  - Contains only: `ids_n4`, `ids_n5`, `ids_n6`, `ids_n7`

### Verification (2026-02-06)

Verified on the extracted `share/skeleton_ids_n4_n7.lmdb`:
- All circuits in `ids_n4`..`ids_n7` are **identity**
- All circuits are **fully noncommuting** (all adjacent pairs collide)

Screenshots: `share/issue_assets/skeleton_db_stats.png`, `share/issue_assets/skeleton_identity_verify.png`

### Merge into a colleague’s LMDB

```bash
python3 share/merge_skeleton_identities_lmdb.py \
  --src share/skeleton_ids_n4_n7.lmdb \
  --dst /path/to/their/local_mixing/db
```

---

## Quick Start

### Generate (batch, DB-backed)

Use the DB builder (parameterized CLI):

```bash
python3 sat_revsynth/scripts/generate_skeletons_batch.py 7 \
  --gates 12 14 --taxonomies 10 --variants 100 \
  --db sat_revsynth/skeleton_identity_db
```

### Generate (small demo `.gate` files)

This writes a few example `.gate` skeleton chains into `sat_revsynth/generated_skeletons/`:

```bash
python3 sat_revsynth/scripts/generate_skeleton_chains.py
```

### Verify a `.gate` skeleton chain
```bash
python3 sat_revsynth/scripts/verify_skeleton.py sat_revsynth/generated_skeletons/chain_4w_10g.gate
```

### Explore the Database
```bash
# Get statistics
python3 sat_revsynth/scripts/explore_skeleton_db.py --stats

# List available identity DBs (ids_n*)
python3 sat_revsynth/scripts/explore_skeleton_db.py --list

# Show sample circuits from one DB
python3 sat_revsynth/scripts/explore_skeleton_db.py --db ids_n7 --show-circuits 5

# Filter by GatePair taxonomy
python3 sat_revsynth/scripts/explore_skeleton_db.py --db ids_n7 --taxonomy OnCtrl1,OnActive,OnActive --show-circuits 5
```

### Run the Demo Pipeline
```bash
python sat_revsynth/examples/skeleton_obfuscation_pipeline.py
```

This demonstrates: synthesize chain -> verify -> generate bounded variants.

---

## QA Considerations

### Known Issues in Older Data

Some chains in the database may have issues:

| Issue | Description | How to Check |
|-------|-------------|--------------|
| Adjacent identical gates | Same gate appears twice in a row (trivially simplifiable) | Run verification script |
| Missing cyclic collision | Last gate doesn't collide with first gate | Check `all_adjacent_collide` function |

### Before Using for Experiments

Always run verification on chains before using them:
```bash
python sat_revsynth/scripts/verify_skeleton.py --batch --db-path <path>
```

The verifier checks:
1. Chain computes identity function
2. All adjacent pairs collide
3. Cyclic (last-to-first) collision holds

---

## Implementation Details

### How Collision Constraints Work

The synthesizer (`eca57_skeleton_synthesizer.py`) enforces collisions via SAT clauses:
- For each adjacent pair (g_i, g_{i+1}), add clauses requiring shared wire AND non-commutativity
- The SAT solver finds assignments satisfying all constraints simultaneously

### Database Verification

The explorer script (`explore_skeleton_db.py`) includes `all_adjacent_collide()` which re-verifies collision properties. Use this to audit the database.

---

## File Reference

```
sat_revsynth/
├── src/
│   ├── synthesizers/
│   │   ├── eca57_skeleton_synthesizer.py  # Core synthesis logic
│   │   └── skeleton_chain_api.py          # Programmatic API
│   └── database/
│       └── skeleton_db.py                 # DB schema and helpers
├── scripts/
│   ├── generate_skeleton_chains.py        # Batch generation
│   ├── verify_skeleton.py                 # Chain verification
│   └── explore_skeleton_db.py             # Database explorer
└── examples/
    └── skeleton_obfuscation_pipeline.py   # End-to-end demo

identity-factory-ui/
└── src/
    ├── app/skeleton-explorer/
    │   └── page.tsx                       # Explorer page
    └── components/
        └── SkeletonGraph.tsx              # Chain visualization
```

---

## Future Work

- Generate n=3 chains to complete the database
- Investigate chains with specific structural properties
- Performance optimization for larger widths

---

## Questions?

The verification script is your friend - when in doubt, verify. For deeper questions about the SAT encoding, see `eca57_skeleton_synthesizer.py`.
