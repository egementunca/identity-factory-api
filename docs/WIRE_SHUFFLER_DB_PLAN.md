# Wire Shuffler Database Plan

This document defines a **small, focused database** for SAT‑synthesized **ECA57 wire shufflers** and describes how to expose it through `identity-factory-api` and visualize it in `identity-factory-ui`.

The wire shuffler definition is fixed as:

```
y[i] = x[w[i]]
```

where `w` is a permutation on wire indices `0..n-1`.

---

## 1. Goals

- Store SAT‑synthesized circuits that implement **wire permutations**.
- Keep **rich metadata** for analysis (distance from identity, cycle structure).
- Provide **API filters** for width/gate count/metrics.
- Enable **UI exploration** (histograms, per‑perm details, minimal circuits).

Non‑goals:
- Ancilla support (future).
- Large widths (SAT‑only; small `n`).

---

## 2. Data Model (SQLite)

We use a new SQLite file, e.g.:

```
identity-factory-api/wire_shuffler.db
```

### 2.1 Tables

#### A) `wire_shuffler_runs`
Tracks each synthesis run (batch).

| field | type | notes |
| --- | --- | --- |
| id | INTEGER PK | |
| width | INTEGER | number of wires |
| min_gates | INTEGER | lower SAT bound |
| max_gates | INTEGER | upper SAT bound |
| solver | TEXT | e.g. `cadical153` |
| require_all_wires | BOOLEAN | SAT constraint |
| order | TEXT | `lex`, `hamming-desc`, `swap-desc`, `random` |
| seed | INTEGER | optional |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |
| status | TEXT | `running`, `complete`, `failed` |
| notes | TEXT | optional |
| git_sha | TEXT | optional |

#### B) `wire_permutations`
One row per **wire permutation** (unique `w`).

| field | type | notes |
| --- | --- | --- |
| id | INTEGER PK | |
| width | INTEGER | |
| wire_perm | TEXT | JSON array of length `width` |
| wire_perm_hash | TEXT | short hash for indexing |
| fixed_points | INTEGER | count of `w[i]==i` |
| hamming | INTEGER | `width - fixed_points` |
| cycles | INTEGER | number of cycles |
| swap_distance | INTEGER | `width - cycles` |
| cycle_type | TEXT | e.g. `3-2-1` |
| parity | TEXT | `even`/`odd` |
| is_identity | BOOLEAN | |
| created_at | TIMESTAMP | |

#### C) `wire_shuffler_circuits`
SAT results (1+ circuits per permutation).

| field | type | notes |
| --- | --- | --- |
| id | INTEGER PK | |
| run_id | INTEGER FK | links run |
| perm_id | INTEGER FK | links permutation |
| found | BOOLEAN | SAT/UNSAT within bound |
| gate_count | INTEGER | gates in circuit |
| gates | TEXT | JSON list of `(target, ctrl1, ctrl2)` |
| circuit_hash | TEXT | hash of gates+perm |
| full_perm | TEXT | JSON of length `2^width` (optional) |
| verify_ok | BOOLEAN | truth‑table check |
| synth_time_ms | INTEGER | time to SAT solution |
| is_best | BOOLEAN | minimal gate count among records for perm |
| created_at | TIMESTAMP | |

#### D) `wire_shuffler_metrics`
Obfuscation‑style metrics for each circuit.

| field | type | notes |
| --- | --- | --- |
| circuit_id | INTEGER PK | FK to circuits |
| width | INTEGER | |
| gate_count | INTEGER | |
| wires_used | INTEGER | count of touched wires |
| wire_coverage | REAL | wires_used / width |
| max_wire_degree | INTEGER | max gates touching a wire |
| avg_wire_degree | REAL | avg gates per wire |
| adjacent_collisions | INTEGER | count of adjacent non‑commuting pairs |
| adjacent_commutes | INTEGER | adjacent pairs that commute |
| total_collisions | INTEGER | all non‑commuting pairs |
| collision_density | REAL | total_collisions / total_pairs |
---

## 3. Suggested Indexes

```
CREATE INDEX idx_perm_hash ON wire_permutations(wire_perm_hash);
CREATE INDEX idx_perm_width ON wire_permutations(width);
CREATE INDEX idx_perm_hamming ON wire_permutations(width, hamming);
CREATE INDEX idx_perm_swapdist ON wire_permutations(width, swap_distance);
CREATE INDEX idx_circ_perm ON wire_shuffler_circuits(perm_id);
CREATE INDEX idx_circ_run ON wire_shuffler_circuits(run_id);
CREATE INDEX idx_circ_gatecount ON wire_shuffler_circuits(gate_count);
CREATE INDEX idx_circ_found ON wire_shuffler_circuits(found);
```

---

## 4. API Endpoints (new)

Namespace suggestion: `/api/v1/wire-shuffler`

### 4.1 Stats
```
GET /wire-shuffler/stats
```

Returns:
- total permutations
- total circuits
- counts by width
- counts by gate_count
- counts by hamming / swap_distance

### 4.2 List permutations
```
GET /wire-shuffler/permutations?width=4&hamming=4&swap_distance=3&limit=50&offset=0
```

### 4.3 List circuits
```
GET /wire-shuffler/circuits?width=4&gate_count=6&found=true&limit=50&offset=0
```

### 4.4 Get permutation details (with best circuit)
```
GET /wire-shuffler/permutation/{wire_perm_hash}
```

### 4.5 Get circuit details
```
GET /wire-shuffler/circuit/{id}
```

---

## 5. UI Integration (identity-factory-ui)

Add a new panel (similar to `Permutation Tables`):

**Filters**
- Width (n)
- Hamming distance (moved wires)
- Swap distance (min swaps)
- Gate count (min/ max)
- Found vs UNSAT

**Views**
1. Summary histograms:
   - hamming distribution
   - swap distance distribution
   - gate count distribution
2. Table of permutations with:
   - `w` (wire perm)
   - cycle notation (wire level)
   - hamming / swap distance
   - best gate count
3. Circuit detail:
   - gate list
   - ASCII diagram
   - full truth table (optional)

---

## 6. Ingestion Pipeline

The script already outputs JSON:

```
sat_revsynth/scripts/synth_wire_shuffle.py --out-json ...
```

Importer script (implemented):

```
identity-factory-api/scripts/import_wire_shuffler.py
```

Usage:

```
python identity-factory-api/scripts/import_wire_shuffler.py \
  --json /path/to/wire_shuffle.json \
  --min-gates 0 --max-gates 8 --solver cadical153 --order hamming-desc
```

Next step: add richer run metadata or integrate into the CLI.

1. Load JSON
2. Upsert `wire_permutations`
3. Insert `wire_shuffler_circuits`
4. Mark `is_best` for minimal gate count per perm

---

## 7. Data Size Expectations

Small widths only:

| width | perms | full_perm size |
| --- | --- | --- |
| 3 | 6 | 8 ints |
| 4 | 24 | 16 ints |
| 5 | 120 | 32 ints |
| 6 | 720 | 64 ints |

Storing `full_perm` is fine for `n<=6`. For larger `n`, keep only `wire_perm`
and compute `full_perm` on demand.

---

## 8. Minimal DDL (for reference)

```sql
CREATE TABLE IF NOT EXISTS wire_shuffler_runs (
  id INTEGER PRIMARY KEY,
  width INTEGER NOT NULL,
  min_gates INTEGER NOT NULL,
  max_gates INTEGER NOT NULL,
  solver TEXT NOT NULL,
  require_all_wires BOOLEAN DEFAULT 0,
  "order" TEXT,
  seed INTEGER,
  status TEXT,
  notes TEXT,
  git_sha TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wire_permutations (
  id INTEGER PRIMARY KEY,
  width INTEGER NOT NULL,
  wire_perm TEXT NOT NULL,
  wire_perm_hash TEXT NOT NULL,
  fixed_points INTEGER,
  hamming INTEGER,
  cycles INTEGER,
  swap_distance INTEGER,
  cycle_type TEXT,
  parity TEXT,
  is_identity BOOLEAN DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wire_shuffler_circuits (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  perm_id INTEGER NOT NULL,
  found BOOLEAN NOT NULL,
  gate_count INTEGER,
  gates TEXT,
  circuit_hash TEXT,
  full_perm TEXT,
  verify_ok BOOLEAN,
  synth_time_ms INTEGER,
  is_best BOOLEAN DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES wire_shuffler_runs(id),
  FOREIGN KEY (perm_id) REFERENCES wire_permutations(id)
);
```

---

## 9. Next Step Checklist

1. Add importer script (JSON -> SQLite).
2. Add API endpoints in `identity-factory-api`.
3. Add UI panel or extend `DualDatabaseView`.
4. Run small exhaustive batches (width 3,4).
5. Validate distributions; tune SAT gate bounds.
6. Write LMDB mirror during import (optional but requested).
7. Compute and store obfuscation metrics per circuit.

---

## 10. LMDB Mirror (Optional but Requested)

We will **also write a compact LMDB mirror** so large blobs (gates, full_perm)
are accessible with fast key lookups. LMDB is not ideal for multi‑field filters,
so **SQLite remains the primary query/index store**. LMDB is a secondary
artifact store with lightweight prefix scans.

### 10.1 LMDB Location

```
identity-factory-api/wire_shuffler.lmdb
```

### 10.2 Databases (LMDB named DBs)

**A) `ws_meta`**
- Key: `b"meta:<name>"`
  - `meta:version`
  - `meta:created_at`
  - `meta:last_run_id`
  - `meta:next_circuit_id`

**B) `ws_perm` (permutation records)**
- Key: `b"p:<width>:<perm_hash>"`
- Value: msgpack/json blob with:
  - `wire_perm`
  - `fixed_points`, `hamming`, `cycles`, `swap_distance`
  - `cycle_type`, `parity`, `is_identity`

**C) `ws_perm_by_width` (index)**
- Key: `b"pw:<width>:<perm_hash>"`
- Value: empty
- Enables prefix scans by width

**D) `ws_perm_by_hamming` (index)**
- Key: `b"ph:<width>:<hamming>:<perm_hash>"`
- Value: empty

**E) `ws_perm_by_swap` (index)**
- Key: `b"ps:<width>:<swap_distance>:<perm_hash>"`
- Value: empty

**F) `ws_circ` (circuit records)**
- Key: `b"c:<perm_hash>:<gate_count>:<circuit_hash>"`
- Value: msgpack/json blob with:
  - `gates`
  - `gate_count`
  - `found`
  - `verify_ok`
  - `full_perm` (optional, for n<=6)
  - `synth_time_ms`
  - `run_id`
  - `is_best`

**G) `ws_circ_by_perm` (index)**
- Key: `b"cp:<perm_hash>:<gate_count>:<circuit_hash>"`
- Value: empty

### 10.3 Key Encoding

All keys are ASCII with `:` separators for easy prefix scans. For example:

```
p:4:9a12f3bc
pw:4:9a12f3bc
ph:4:4:9a12f3bc
ps:4:3:9a12f3bc
c:9a12f3bc:6:0ff1aa9e
```

### 10.4 Why Dual Store

- **SQLite**: fast filtering, pagination, stats.
- **LMDB**: fast direct lookup by perm hash, easy blob retrieval, shareable with
  other tools.

### 10.5 API Strategy

- API continues to query SQLite for lists/filters.
- When a record is selected, API can optionally fetch the blob from LMDB
  (e.g., full gates or full permutation).

### 10.6 Minimal LMDB Write Flow

Importer (JSON -> DB):
1. Upsert SQLite tables (authoritative index).
2. Write LMDB records:
   - `ws_perm`, plus `ws_perm_by_width/hamming/swap`.
   - `ws_circ`, plus `ws_circ_by_perm`.
