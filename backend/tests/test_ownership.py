import asyncio
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.models.job import Job
from app.models.model_run import ModelRun
from app.models.pipeline_plan import PipelinePlanRecord
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.services.security import get_current_user


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def register(app: FastAPI, email: str) -> dict[str, str]:
    result = request(app, "POST", "/api/v1/auth/register", json={"email": email, "password": "ownership-password-123"})
    assert result.status_code == 201, result.text
    return {"Authorization": f"Bearer {result.json()['access_token']}"}


def test_two_user_dataset_experiment_and_action_isolation(test_app: FastAPI) -> None:
    # Exercise the real JWT dependency instead of the legacy fixture shortcut.
    test_app.dependency_overrides.pop(get_current_user, None)
    owner = register(test_app, "owner@ownership.example")
    other = register(test_app, "other@ownership.example")
    uploaded = request(test_app, "POST", "/api/v1/datasets", headers=owner, files={"file": ("owned.csv", b"age,target\n20,yes\n21,no\n22,yes\n23,no\n", "text/csv")})
    assert uploaded.status_code == 201, uploaded.text
    dataset_id = uploaded.json()["id"]
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset_id}", headers=owner).status_code == 200
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset_id}/profile", headers=owner).status_code == 200
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset_id}", headers=other).status_code == 404
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset_id}/profile", headers=other).status_code == 404
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset_id}/auto-analysis", headers=other).status_code == 404
    assert request(test_app, "POST", f"/api/v1/datasets/{dataset_id}/analytics/query", headers=other, json={"operation": "row_count"}).status_code == 404
    assert request(test_app, "POST", f"/api/v1/datasets/{dataset_id}/anomalies", headers=other, json={}).status_code == 404
    assert request(test_app, "POST", "/api/v1/assistant/query", headers=other, json={"dataset_id": dataset_id, "question": "How many rows are there?"}).status_code == 404
    assert request(test_app, "POST", "/api/v1/experiments", headers=other, json={"dataset_id": dataset_id, "objective": "Predict target"}).status_code == 404
    created = request(test_app, "POST", "/api/v1/experiments", headers=owner, json={"dataset_id": dataset_id, "objective": "Predict target"})
    assert created.status_code == 201, created.text
    experiment_id = created.json()["experiment_id"]
    assert request(test_app, "GET", f"/api/v1/experiments/{experiment_id}", headers=owner).status_code == 200
    assert request(test_app, "GET", f"/api/v1/experiments/{experiment_id}", headers=other).status_code == 404
    history = request(test_app, "GET", "/api/v1/experiments", headers=other).json()
    assert experiment_id not in {item["experiment_id"] for item in history["items"]}
    denied = [
        ("POST", f"/api/v1/experiments/{experiment_id}/confirm", {"json": {"target_column": "target", "task_type": "binary_classification"}}),
        ("POST", f"/api/v1/experiments/{experiment_id}/plan", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/train", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/evaluate", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/optimize", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/run", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/jobs/train", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/predictions", {"files": {"file": ("input.csv", b"age\\n20\\n", "text/csv")}}),
        ("GET", f"/api/v1/experiments/{experiment_id}/explainability", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/report", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/report/pdf", {}),
        ("POST", f"/api/v1/experiments/{experiment_id}/reflection", {}),
    ]
    for method, path, kwargs in denied:
        assert request(test_app, method, path, headers=other, **kwargs).status_code == 404

    database = test_app.state.testing_session_factory()
    try:
        job = Job(experiment_id=experiment_id, job_type="train", status="PENDING", stage="PENDING")
        plan = PipelinePlanRecord(experiment_id=experiment_id, plan={}, provider_used="test")
        database.add_all([job, plan])
        database.commit()
        model_run = ModelRun(
            experiment_id=experiment_id, pipeline_plan_id=plan.id, model_name="test-model",
            status="COMPLETED", parameters={}, training_duration_seconds=0.0,
            failure_information=None, artifact_filename=None,
        )
        database.add(model_run)
        database.commit()
        prediction = PredictionRun(
            experiment_id=experiment_id, model_run_id=model_run.id, row_count=1,
            artifact_filename="owned-predictions.csv",
        )
        html_report = Report(experiment_id=experiment_id, artifact_filename="owned-report.html")
        pdf_report = Report(experiment_id=experiment_id, artifact_filename="owned-report.pdf")
        database.add_all([prediction, html_report, pdf_report])
        database.commit()
        for directory, filename, content in (
            (test_app.state.testing_settings.prediction_storage_path, prediction.artifact_filename, b"prediction\\n1\\n"),
            (test_app.state.testing_settings.report_storage_path, html_report.artifact_filename, b"<html>owned</html>"),
            (test_app.state.testing_settings.report_storage_path, pdf_report.artifact_filename, b"%PDF-1.4\\n%%EOF\\n"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / filename).write_bytes(content)
    finally:
        database.close()

    assert request(test_app, "GET", f"/api/v1/jobs/{job.id}", headers=owner).status_code == 200
    assert request(test_app, "GET", f"/api/v1/jobs/{job.id}", headers=other).status_code == 404
    for artifact_path in (f"/api/v1/predictions/{prediction.id}", f"/api/v1/reports/{html_report.id}", f"/api/v1/reports/{pdf_report.id}"):
        assert request(test_app, "GET", artifact_path, headers=owner).status_code == 200
        assert request(test_app, "GET", artifact_path, headers=other).status_code == 409


def test_google_and_password_users_have_identical_ownership_boundaries(test_app: FastAPI, monkeypatch) -> None:
    test_app.dependency_overrides.pop(get_current_user, None)
    from app.services import auth_service
    monkeypatch.setattr(auth_service.id_token, "verify_oauth2_token", lambda *_args: {"sub": "google-owner", "email": "google@ownership.example", "email_verified": True})
    google = request(test_app, "POST", "/api/v1/auth/google", json={"credential": "mock"})
    assert google.status_code == 200
    google_headers = {"Authorization": f"Bearer {google.json()['access_token']}"}
    password_headers = register(test_app, "password@ownership.example")
    upload = request(test_app, "POST", "/api/v1/datasets", headers=google_headers, files={"file": ("google.csv", b"x,target\n1,a\n2,b\n", "text/csv")})
    assert upload.status_code == 201
    assert request(test_app, "GET", f"/api/v1/datasets/{upload.json()['id']}", headers=password_headers).status_code == 404
    assert request(test_app, "GET", f"/api/v1/datasets/{upload.json()['id']}", headers=google_headers).status_code == 200
