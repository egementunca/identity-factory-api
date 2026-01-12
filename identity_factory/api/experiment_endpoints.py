"""
API endpoints for Local Mixing Experiments.

Provides endpoints to:
- Get experiment presets and configuration schema
- Start experiments with custom configuration
- Stream experiment progress (SSE)
- Get experiment results
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from identity_factory.api.experiment_models import (
    ConfigSchemaResponse,
    ExperimentConfig,
    ExperimentProgress,
    ExperimentResults,
    ExperimentStatus,
    ExperimentStatusResponse,
    PresetsResponse,
    StartExperimentRequest,
    StartExperimentResponse,
)
from identity_factory.experiment_runner import experiment_runner

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/presets", response_model=PresetsResponse)
async def get_presets():
    """
    Get pre-configured experiment presets.

    Returns a list of ready-to-use experiment configurations for common use cases:
    - Expansion Study: Track circuit growth over rounds
    - Inflation Only: Test maximum inflation without compression
    - SAT Compression: Use SAT solver for optimal compression
    - Annealed: Simulated annealing approach
    """
    presets = experiment_runner.get_presets()
    return PresetsResponse(presets=presets)


@router.get("/config-schema", response_model=ConfigSchemaResponse)
async def get_config_schema():
    """
    Get JSON schema for experiment configuration.

    Returns the full schema with field descriptions for building dynamic forms.
    """
    schema = ExperimentConfig.model_json_schema()

    # Extract descriptions for easier UI rendering
    descriptions = {}

    def extract_descriptions(schema_obj, prefix=""):
        if "properties" in schema_obj:
            for key, value in schema_obj["properties"].items():
                full_key = f"{prefix}.{key}" if prefix else key
                if "description" in value:
                    descriptions[full_key] = value["description"]
                if "$ref" in value:
                    # Handle nested models
                    ref_name = value["$ref"].split("/")[-1]
                    if "$defs" in schema and ref_name in schema["$defs"]:
                        extract_descriptions(schema["$defs"][ref_name], full_key)
                elif "properties" in value:
                    extract_descriptions(value, full_key)

    extract_descriptions(schema)

    return ConfigSchemaResponse(
        schema=schema,
        parameter_descriptions=descriptions,
    )


@router.post("/start", response_model=StartExperimentResponse)
async def start_experiment(request: StartExperimentRequest):
    """
    Start a new experiment job.

    The experiment runs asynchronously in the background. Use the returned
    job_id to poll for status or stream progress via SSE.
    """
    try:
        job = await experiment_runner.start_experiment(request.config)

        return StartExperimentResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Experiment '{request.config.name}' started",
            started_at=job.started_at,
        )
    except Exception as e:
        logger.exception("Failed to start experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(job_id: str):
    """
    Get current status and progress of an experiment.
    """
    job = experiment_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    progress = experiment_runner.get_progress(job)
    results = (
        experiment_runner.get_results(job)
        if job.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)
        else None
    )

    return ExperimentStatusResponse(
        job_id=job_id,
        status=job.status,
        progress=progress,
        results=results,
    )


@router.get("/{job_id}/stream")
async def stream_experiment_progress(job_id: str):
    """
    Stream experiment progress via Server-Sent Events (SSE).

    Connect to this endpoint to receive real-time updates:
    - Log lines as they are produced
    - Current round and gate count
    - Final completion status

    Example client:
    ```javascript
    const eventSource = new EventSource(`/api/experiments/${jobId}/stream`);
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.new_lines);
    };
    ```
    """
    job = experiment_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    return StreamingResponse(
        experiment_runner.stream_progress(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/results", response_model=ExperimentResults)
async def get_experiment_results(job_id: str):
    """
    Get results of a completed experiment.

    Includes:
    - Final gate count and expansion factor
    - Heatmap data (if generated)
    - Paths to output files
    - Full log output
    """
    job = experiment_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    if job.status not in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
        raise HTTPException(
            status_code=400, detail=f"Experiment {job_id} is still {job.status.value}"
        )

    results = experiment_runner.get_results(job)
    if not results:
        raise HTTPException(status_code=500, detail="Failed to get results")

    return results


@router.get("/{job_id}/logs")
async def get_experiment_logs(job_id: str, tail: int = 100):
    """
    Get log output from an experiment.

    Args:
        tail: Number of lines to return (default 100, max 1000)
    """
    job = experiment_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    tail = min(tail, 1000)
    lines = job.log_lines[-tail:] if tail > 0 else job.log_lines

    return {
        "job_id": job_id,
        "status": job.status.value,
        "line_count": len(job.log_lines),
        "lines": lines,
    }


@router.delete("/{job_id}")
async def cancel_experiment(job_id: str):
    """
    Cancel a running experiment.
    """
    job = experiment_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    if job.status not in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel experiment in {job.status.value} state",
        )

    # Terminate subprocess if running
    if job.process:
        try:
            job.process.terminate()
        except:
            pass

    job.status = ExperimentStatus.CANCELLED
    job.log_lines.append("Experiment cancelled by user")

    return {"status": "cancelled", "job_id": job_id}
