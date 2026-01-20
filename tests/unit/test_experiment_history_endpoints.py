import json
from pathlib import Path

from fastapi.testclient import TestClient

from identity_factory.api import experiment_endpoints as endpoints
from identity_factory.api.experiment_models import ExperimentConfig, ObfuscationParams
from identity_factory.api.server import create_app


def _write_results(tmp_path: Path, job_id: str = "job123") -> None:
    config = ExperimentConfig(
        name="Demo Experiment",
        experiment_type="custom",
        wires=4,
        initial_gates=8,
        obfuscation=ObfuscationParams(),
    )
    data = {
        "job_id": job_id,
        "config": config.model_dump(),
        "initial_gates": 8,
        "final_gates": 12,
        "expansion_factor": 1.5,
        "elapsed_seconds": 3.2,
    }

    results_dir = tmp_path / "2026-01-01" / f"{config.name}_{job_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(data, f)


def test_history_endpoint_reads_results_file(tmp_path, monkeypatch):
    _write_results(tmp_path, job_id="job123")
    monkeypatch.setattr(endpoints, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(endpoints.experiment_runner, "jobs", {})

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/experiments/history")
    assert response.status_code == 200
    payload = response.json()

    assert payload["history"]
    item = payload["history"][0]
    assert item["job_id"] == "job123"
    assert item["name"] == "Demo Experiment"
    assert item["status"] == "completed"
    assert item["final_gates"] == 12


def test_summary_endpoint_reads_results_file(tmp_path, monkeypatch):
    _write_results(tmp_path, job_id="job456")
    monkeypatch.setattr(endpoints, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(endpoints.experiment_runner, "jobs", {})

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/experiments/job456/summary")
    assert response.status_code == 200
    payload = response.json()

    assert payload["job_id"] == "job456"
    assert payload["name"] == "Demo Experiment"
    assert payload["final_gates"] == 12


def test_summary_endpoint_missing_job_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoints, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(endpoints.experiment_runner, "jobs", {})

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/experiments/does-not-exist/summary")
    assert response.status_code == 404
