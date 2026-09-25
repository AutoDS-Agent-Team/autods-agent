import asyncio
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.models.job import Job
from app.services import job_service
from app.tasks import SERVICES, execute_job
from test_training import classification_csv, prepare_experiment


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def test_job_is_persisted_pending_and_unknown_job_is_404(test_app: FastAPI, monkeypatch) -> None:
    experiment_id = prepare_experiment(
        test_app, classification_csv(["yes", "no"]), target="target",
        task_type="binary_classification", models=["logistic_regression"],
    )
    queued: list[str] = []
    monkeypatch.setattr(job_service.execute_job, "delay", lambda job_id: queued.append(job_id))
    created = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/run")
    assert created.status_code == 202, created.text
    payload = created.json()
    assert payload["status"] == "PENDING"
    assert payload["stage"] == "PENDING"
    assert queued == [payload["job_id"]]
    assert request(test_app, "GET", f"/api/v1/jobs/{payload['job_id']}").json()["status"] == "PENDING"
    assert request(test_app, "GET", "/api/v1/jobs/not-a-job").status_code == 404


def test_history_pagination_and_reopen_are_read_only(test_app: FastAPI, monkeypatch) -> None:
    first = prepare_experiment(test_app, classification_csv(["yes", "no"]), target="target", task_type="binary_classification", models=["logistic_regression"])
    second = prepare_experiment(test_app, classification_csv(["yes", "no"]), target="target", task_type="binary_classification", models=["logistic_regression"])
    monkeypatch.setattr(job_service.execute_job, "delay", lambda _job_id: None)
    listing = request(test_app, "GET", "/api/v1/experiments?page=1&page_size=1")
    assert listing.status_code == 200
    assert listing.json()["total"] == 2
    assert len(listing.json()["items"]) == 1
    detail = request(test_app, "GET", f"/api/v1/experiments/{first}")
    assert detail.status_code == 200
    assert detail.json()["experiment_id"] == first
    assert detail.json()["model_runs"] == []
    assert request(test_app, "GET", f"/api/v1/experiments/{second}").status_code == 200


def test_worker_marks_success_and_failure_without_erasing_prior_state(monkeypatch) -> None:
    class Database:
        def __init__(self, job: Job) -> None:
            self.job = job
            self.commits = 0
        def get(self, model, identifier):
            return self.job if model is Job and identifier == self.job.id else None
        def commit(self) -> None:
            self.commits += 1
        def close(self) -> None:
            pass

    successful_job = Job(id="job-success", experiment_id="experiment", job_type="train", status="PENDING", stage="PENDING")
    successful_db = Database(successful_job)
    monkeypatch.setattr("app.tasks.SessionLocal", lambda: successful_db)
    monkeypatch.setitem(SERVICES, "train", ("TRAINING", lambda *_args: object()))
    execute_job.run(successful_job.id)
    assert successful_job.status == "COMPLETED"
    assert successful_job.stage == "COMPLETED"
    assert successful_job.started_at is not None and successful_job.completed_at is not None

    failed_job = Job(id="job-failure", experiment_id="experiment", job_type="optimize", status="PENDING", stage="PENDING")
    failed_db = Database(failed_job)
    monkeypatch.setattr("app.tasks.SessionLocal", lambda: failed_db)
    monkeypatch.setitem(SERVICES, "optimize", ("OPTIMIZING", lambda *_args: (_ for _ in ()).throw(ValueError("bad input"))))
    execute_job.run(failed_job.id)
    assert failed_job.status == "FAILED"
    assert failed_job.error_information == "optimize failed: ValueError."


def test_worker_training_reads_the_dataset_persisted_by_the_api(test_app: FastAPI, monkeypatch) -> None:
    experiment_id = prepare_experiment(
        test_app,
        classification_csv(["yes", "no"]),
        target="target",
        task_type="binary_classification",
        models=["logistic_regression"],
    )
    monkeypatch.setattr(job_service.execute_job, "delay", lambda _job_id: None)
    created = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/run")
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    monkeypatch.setattr("app.tasks.SessionLocal", test_app.state.testing_session_factory)
    monkeypatch.setattr("app.tasks.get_settings", lambda: test_app.state.testing_settings)
    execute_job.run(job_id)

    job = request(test_app, "GET", f"/api/v1/jobs/{job_id}").json()
    detail = request(test_app, "GET", f"/api/v1/experiments/{experiment_id}").json()
    assert job["status"] == "COMPLETED"
    assert len(detail["model_runs"]) == 1
    assert detail["model_runs"][0]["model_name"] == "logistic_regression"
    assert detail["model_runs"][0]["status"] == "COMPLETED"
