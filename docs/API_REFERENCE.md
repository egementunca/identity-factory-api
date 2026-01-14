# Identity Factory API - Route Reference

Base URL: `http://localhost:8000`
All API routes are mounted under `/api/v1` unless noted.

## Core identity-factory endpoints (`identity_factory/api/endpoints.py`)
- `POST /api/v1/generate` - Generate a single identity circuit (seed generation).
- `POST /api/v1/batch-generate` - Generate circuits for multiple dimensions.
- `GET /api/v1/circuits` - Search circuits (pagination + filters).
- `POST /api/v1/circuits/advanced-search` - Advanced filtering.
- `GET /api/v1/dim-groups` - List dimension groups.
- `GET /api/v1/dim-groups/{dim_group_id}` - Dimension group details.
- `GET /api/v1/dim-groups/{dim_group_id}/circuits` - Circuits in a group (`representatives_only` optional).
- `GET /api/v1/dim-groups/{dim_group_id}/compositions` - Group circuits by gate composition.
- `GET /api/v1/circuits/{circuit_id}` - Circuit details.
- `POST /api/v1/circuits/{circuit_id}/unroll` - Unroll a circuit to generate equivalents.
- `POST /api/v1/circuits/{circuit_id}/analyze-debris` - Debris cancellation analysis.
- `GET /api/v1/circuits/{circuit_id}/visualization` - ASCII diagram + permutation table.
- `GET /api/v1/stats` - Factory statistics.
- `GET /api/v1/generator/stats` - Seed generation stats.
- `GET /api/v1/health` - Health check.

## Generator endpoints (`identity_factory/api/generator_endpoints.py`)
- `GET /api/v1/generators/` - List generators.
- `GET /api/v1/generators/{generator_name}` - Generator details.
- `GET /api/v1/generators/gate-sets/` - Supported gate sets.
- `POST /api/v1/generators/run` - Start a generator run.
- `GET /api/v1/generators/runs/` - List runs.
- `GET /api/v1/generators/runs/{run_id}` - Run status.
- `GET /api/v1/generators/runs/{run_id}/result` - Run result.
- `POST /api/v1/generators/runs/{run_id}/cancel` - Cancel a run.
- `DELETE /api/v1/generators/runs/{run_id}` - Delete run record.

## Advanced endpoints (`identity_factory/api/advanced_endpoints.py`)
- `POST /api/v1/circuits/{circuit_id}/compress` - Compress circuit (Go or Python fallback).
- `GET /api/v1/identities/{width}/{gate_count}` - Precomputed identity circuits from text files.
- `GET /api/v1/identities/stats` - Identity file stats.
- `POST /api/v1/synthesize` - SAT synthesis for a target permutation.

## Go database endpoints (`identity_factory/api/go_database_endpoints.py`)
- `GET /api/v1/go-database/` - List available Go databases.
- `GET /api/v1/go-database/stats` - Summary stats (uses `go_stats` if available).
- `GET /api/v1/go-database/{file_name}` - File info.
- `GET /api/v1/go-database/{file_name}/circuits` - List circuits (uses `go_explore`).

## SAT database endpoints (`identity_factory/api/sat_database_endpoints.py`)
- `GET /api/v1/sat-database/stats` - SAT DB stats.
- `GET /api/v1/sat-database/circuits` - List SAT circuits.
- `GET /api/v1/sat-database/circuit/{circuit_id}` - Circuit detail.

## Cluster database endpoints (`identity_factory/api/cluster_endpoints.py`)
- `GET /api/v1/cluster-database/stats` - Cluster DB stats.
- `GET /api/v1/cluster-database/circuits` - List cluster circuits.
- `GET /api/v1/cluster-database/circuit/{circuit_id}` - Circuit detail + diagram.

## Irreducible endpoints (`identity_factory/api/irreducible_endpoints.py`)
- `POST /api/v1/irreducible/generate` - Generate forward circuits.
- `POST /api/v1/irreducible/find-inverse/{forward_id}` - Synthesize and store inverse.
- `GET /api/v1/irreducible/stats` - Irreducible DB stats.
- `GET /api/v1/irreducible/forward/{width}` - List forward circuits by width.
- `POST /api/v1/irreducible/batch-pipeline` - Generate -> inverse -> identity pipeline.

## ECA57 LMDB endpoints (`identity_factory/api/eca57_lmdb_endpoints.py`)
- `GET /api/v1/eca57-lmdb/stats` - LMDB stats.
- `GET /api/v1/eca57-lmdb/configurations` - List (width, gate_count) configs.
- `GET /api/v1/eca57-lmdb/circuits/{width}/{gate_count}` - Circuits with pagination.
- `GET /api/v1/eca57-lmdb/circuit/{width}/{gate_count}/{circuit_id}` - Circuit details.
- `GET /api/v1/eca57-lmdb/circuit/{width}/{gate_count}/{circuit_id}/equivalents` - Unrolled equivalents.

## Local mixing endpoints (`identity_factory/api/local_mixing_endpoints.py`)
- `POST /api/v1/local-mixing/load-circuit` - Parse circuit string to gates + permutation.
- `POST /api/v1/local-mixing/canonicalize` - Canonicalize circuit via local_mixing.
- `POST /api/v1/local-mixing/inflate-preview` - Preview R^-1 g R inflation sizes.
- `POST /api/v1/local-mixing/heatmap` - Gate overlap heatmap.
- `POST /api/v1/local-mixing/random-circuit` - Random circuit generator.
- `GET /api/v1/local-mixing/examples` - Example circuits.
- `POST /api/v1/local-mixing/random-identity` - Random identity circuit.

## Experiment endpoints (`identity_factory/api/experiment_endpoints.py`)
- `GET /api/v1/experiments/presets` - Preset list.
- `GET /api/v1/experiments/config-schema` - JSON schema + parameter descriptions.
- `POST /api/v1/experiments/start` - Start an experiment job.
- `GET /api/v1/experiments/{job_id}/status` - Job status.
- `GET /api/v1/experiments/{job_id}/stream` - Server-sent events (live progress).
- `GET /api/v1/experiments/{job_id}/results` - Final results.
- `GET /api/v1/experiments/{job_id}/logs` - Job logs.
- `DELETE /api/v1/experiments/{job_id}` - Cancel and remove job.

## Automation endpoints (`identity_factory/api/automation_endpoints.py`)
Note: This router is mounted at `/api/v1/automation` and its paths also start with `/automation`, so the effective paths are `/api/v1/automation/automation/*`.
- `POST /api/v1/automation/automation/generate`
- `POST /api/v1/automation/automation/unroll`
- `POST /api/v1/automation/automation/filter`
- `POST /api/v1/automation/automation/full-cycle`
- `GET /api/v1/automation/automation/stats`

## Misc
- `GET /api/info` - API metadata summary (root, not under `/api/v1`).
- `GET /logs` - In-memory log buffer.
- `GET /docs` and `GET /redoc` - OpenAPI UI.

