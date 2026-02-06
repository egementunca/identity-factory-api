# Contributions Snapshot (2026-02-06)

This is a short “what changed / what to share” note intended for colleagues to keep up with recent work.

## Code (latest pushed commits)

- `sat_revsynth` (`main`): `9ab473c` — Waksman-style (swap-based) permutation synthesis tooling + identity filler + tests.
- `identity-factory-api` (`master`): `7b42b87` — Waksman endpoints (`/api/v1/waksman/*`) + SQLite `waksman_circuits` table support.
- `identity-factory-ui` (`master`): `d666dff` — UI refactor + `/playground-v2` build fix (Suspense) + shared circuit utilities.
- `local_mixing` (`feature/annealed-obfuscator`): `f53f39a` — Increased LMDB `max_dbs` for `abbutterfly` to avoid `DbsFull` when opening many named DBs.

## Databases / artifacts to share

Everything below is packaged under `share/`:

- Skeleton chain identity DB slice (LMDB): `share/skeleton_ids_n4_n7.lmdb/`
  - Contains only `ids_n4`..`ids_n7` extracted from `local_mixing/db` (full local DB is too large to publish).
  - Verified (2026-02-06): circuits are identity + fully noncommuting (adjacent collisions).
  - Merge helper: `share/merge_skeleton_identities_lmdb.py`.

- Wire shuffler DB (SQLite + LMDB mirror):
  - `share/wire_shuffler.db`
  - `share/wire_shuffler.lmdb/`

- Waksman circuits export (merge-only, JSON): `share/waksman_circuits_export_2026-02-06.json`
  - Import helper: `share/import_waksman_circuits.py`.

## Notes

- `local_mixing/db/*` is **not** included (SQLite + LMDB are tens of GB).
- For details, see:
  - `ISSUE_skeleton_chain_qa.md`
  - `ISSUE_wire_shuffler_waksman.md`
  - `ISSUE_identity_growth_300plus_dtw.md`

