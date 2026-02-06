# Share Package (Research Group)

This folder contains **small, merge-friendly slices** of the large local databases plus helper scripts.

## Skeleton chain identity DB (LMDB)

- `share/skeleton_ids_n4_n7.lmdb/` (≈ 1.1 MB `data.mdb`)
  - Contains only: `ids_n4`, `ids_n5`, `ids_n6`, `ids_n7`
  - Extracted from: `local_mixing/db` (which is too large to share).
  - Verified (2026-02-06):
    - All circuits are **identity**
    - All circuits are **fully noncommuting** (all adjacent pairs collide)

### Merge into an existing `local_mixing/db`

```bash
python3 share/merge_skeleton_identities_lmdb.py \
  --src share/skeleton_ids_n4_n7.lmdb \
  --dst /path/to/your/local_mixing/db
```

## Waksman (swap-based) permutation circuits export (SQLite)

- `share/waksman_circuits_export_2026-02-06.json`
  - Exported from: `identity-factory-api/wire_shuffler.db`
  - Contains the `waksman_circuits` entries + their permutations.
- Full copies (small enough to share):
  - `share/wire_shuffler.db`
  - `share/wire_shuffler.lmdb/`

### Import into another `wire_shuffler.db`

```bash
python3 share/import_waksman_circuits.py \
  --json share/waksman_circuits_export_2026-02-06.json \
  --db-path /path/to/identity-factory-api/wire_shuffler.db
```

## Issue attachments (optional)

- `share/issue_assets/` contains small screenshots/logs referenced from the draft `ISSUE_*.md` files at the repo root.
