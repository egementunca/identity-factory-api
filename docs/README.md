# Identity Factory API - Codebase Guide

## Purpose
The Identity Factory API is a FastAPI service for generating, unrolling, and analyzing identity circuits, plus a collection of data access endpoints for SAT, Go enumeration, cluster results, irreducible circuits, and ECA57 LMDB enumerations.

This guide describes the current codebase layout, runtime behavior, dependencies, and known gaps so you can assess cleanup/refactor work before migration.

## Quickstart
- Create a virtualenv and install: `pip install -e .`
- Run the server: `python start_api.py --port 8000`
- API docs: `http://localhost:8000/docs`

## Repository map
- `start_api.py` - CLI entry point, sets env vars used by `FactoryConfig` and launches FastAPI.
- `identity_factory/api/` - FastAPI routers, Pydantic models, server wiring.
- `identity_factory/` - Core pipeline (seed generation, unroll, debris, ML features), generators, automation, CLI.
- `identity_factory/cli.py` and `identity_factory/__main__.py` - Command-line interface (`python -m identity_factory`).
- `identity_factory/api/client.py` - Async Python client (currently out of sync with live routes).
- `tests/` - Integration + local_mixing_utils tests.

### Core modules (current behavior)
- `identity_factory/database.py` - SQLite storage for `circuits`, `dim_groups`, `jobs`.
- `identity_factory/seed_generator.py` - Random forward circuit + SAT inverse synthesis. Uses `sat_revsynth` if installed; otherwise seed generation is disabled but DB ops still work.
- `identity_factory/unroller.py` - Converts DB records to `sat_revsynth` circuits and calls `Circuit.unroll()` to generate equivalents.
- `identity_factory/post_processor.py` - Simplification via `reduce_by_swaps_and_cancellation()`. References legacy DB helpers; see Known Gaps.
- `identity_factory/debris_cancellation.py` - Debris insertion search and non-triviality scoring. DB storage methods are not wired in `database.py`.
- `identity_factory/ml_features.py` - Feature extraction + heuristic complexity prediction. Persistence is disabled.
- `identity_factory/job_queue.py` - Job queue scaffolding (sync + async). Default handlers are placeholders.
- `identity_factory/automation/scheduler.py` - Batch generation, unroll, quality filtering; uses DB methods that do not exist in the simplified DB.
- `identity_factory/experiment_runner.py` - Runs local_mixing experiments via subprocess and streams progress.
- `identity_factory/local_mixing_utils.py` - Parses ECA57 text format and shells out to local_mixing for canonicalization/compression.
- `identity_factory/rainbow_compressor.py` - Python rainbow table compressor using `go-proj/rainbow_table_3w.json`.

### Generators
- `identity_factory/generators/sat_forward_inverse.py` - Random forward + SAT inverse synthesis (mcx gate set).
- `identity_factory/generators/eca57_identity_miner.py` - SAT-based ECA57 identity mining.
- `identity_factory/generators/go_enumerator.py` - Wraps Go enumerator binary (expects `go-proj`).
- `identity_factory/generators/registry.py` - Generator registry used by API endpoints.

## Data stores and paths
| Store | Type | Path used by code | Used by |
| --- | --- | --- | --- |
| Main circuits DB | SQLite | `~/.identity_factory/circuits.db` | `/api/v1/*` endpoints, generator endpoints |
| Factory default DB | SQLite | `identity_circuits.db` | `FactoryConfig` (used by `IdentityFactory`) |
| Irreducible DB | SQLite | `~/.identity_factory/irreducible.db` | `/api/v1/irreducible/*` |
| Cluster DB | SQLite | `cluster_circuits.db` (repo root) | `/api/v1/cluster-database/*` |
| SAT DB | SQLite | `sat_circuits.db` (repo root) | `/api/v1/sat-database/*` |
| Go databases | Gob files | `go-proj/db/*.gob` (under repo) | `/api/v1/go-database/*` |
| ECA57 LMDB | LMDB | `$SAT_REVSYNTH_PATH/data/eca57_identities_lmdb` | `/api/v1/eca57-lmdb/*` |
| Precomputed identities | Text | `identity_circuits_analysis/` | `/api/v1/identities/*` |
| local_mixing experiments | Files | `../local_mixing/experiments/` | `/api/v1/experiments/*` |

Notes:
- The API uses `~/.identity_factory/circuits.db` by default, while `FactoryConfig` defaults to `identity_circuits.db`. This means the FastAPI endpoints and `IdentityFactory` can point at different databases unless you align them via env vars.
- `cluster_circuits.db` and `identity_circuits.db` exist in the repo root but are not automatically used by the API endpoints.

## Configuration and environment
`start_api.py` sets these env vars when passed on the CLI:
- `IDENTITY_FACTORY_DB_PATH`
- `IDENTITY_FACTORY_MAX_INVERSE_GATES`
- `IDENTITY_FACTORY_MAX_EQUIVALENTS`
- `IDENTITY_FACTORY_SAT_SOLVER`
- `IDENTITY_FACTORY_LOG_LEVEL`
- `IDENTITY_FACTORY_ENABLE_POST_PROCESSING`
- `IDENTITY_FACTORY_ENABLE_UNROLLING`

Other env vars used by API modules:
- `SAT_REVSYNTH_PATH` - Path to `sat_revsynth` (used by ECA57 LMDB endpoints).
- `ECA57_LMDB_PATH` - Override LMDB directory.

## External dependencies
- `sat_revsynth` (Python): provides `Circuit`, SAT solvers, unroll and reduction utilities.
- SAT solver binaries: `minisat-gh` (default), `glucose`, `cadical`.
- `eca57` modules: used by ECA57 mining and irreducible inverse synthesis.
- Go toolchain: required to build `go-proj` enumerators and helpers.
- `lmdb` Python package: required for `/api/v1/eca57-lmdb/*` endpoints.
- `local_mixing` Rust binary: required for local mixing endpoints and experiments.

## API surface
See `identity-factory-api/docs/API_REFERENCE.md` for the full endpoint list.

## Known gaps and legacy notes
These are important when planning cleanup or refactor work:
- Database path mismatch: FastAPI endpoints use `~/.identity_factory/circuits.db`, while `IdentityFactory` defaults to `identity_circuits.db` unless env vars are set.
- `post_processor.py`, `debris_cancellation.py`, and `automation/scheduler.py` reference DB methods that are not implemented in `database.py` (legacy API).
- `job_queue.py` default handlers are placeholders and do not invoke real pipeline logic.
- `identity_factory/api/client.py` targets endpoints that do not exist (`/generate/batch`, `/unroll`, `/simplify`, etc.) and is out of sync with current routes.
- `identity_factory/api/server.py` includes `/api/info` entries for endpoints that are not implemented in the router.
- `go_database_endpoints.py` requires `go-proj` binaries (`go_stats`, `go_explore`) and `db/*.gob` files which are not in this repo.
- `sat_database_endpoints.py` depends on `sat_circuits.db`, which is not present.
- `advanced_endpoints.py` expects `identity_circuits_analysis/` and optionally `go-proj/go_compress`.
