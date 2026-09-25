import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.models.model_run import ModelRun
from app.schemas.pipeline_plan import MetricName
from app.services.evaluation_service import _select_comparison
from app.services.report_service import _metric
from app.services import explainability_service
from test_training import (
    classification_csv,
    prepare_experiment,
    regression_csv,
)


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def test_binary_evaluation_selects_from_validation_and_returns_final_test_confusion_matrix(test_app: FastAPI) -> None:
    experiment_id = prepare_experiment(
        test_app, classification_csv(["yes", "no"]), target="target",
        task_type="binary_classification", models=["logistic_regression", "random_forest"],
    )
    assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train").status_code == 200
    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["validation_comparison"]) == 2
    assert payload["selected_model_run_id"] in {item["model_run_id"] for item in payload["validation_comparison"]}
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(payload["final_test_metrics"])
    assert payload["final_test_confusion_matrix"]["labels"] == ["no", "yes"]
    for item in payload["validation_comparison"]:
        assert item["training_seconds"] >= 0
        assert item["inference_seconds_per_row"] >= 0
        assert item["complexity_summary"]
        if item["model_run_id"] == payload["selected_model_run_id"]:
            assert item["generalization_change"] is not None
        else:
            assert "Not evaluated on test" in item["generalization_status"]


def test_evaluation_handles_sparse_one_hot_validation_matrix(test_app: FastAPI) -> None:
    rows = ["age,segment,target"]
    rows.extend(f"{20 + index},segment_{index % 15},{'yes' if index % 2 else 'no'}" for index in range(80))
    experiment_id = prepare_experiment(
        test_app, ("\n".join(rows) + "\n").encode(), target="target",
        task_type="binary_classification", models=["logistic_regression"],
    )
    trained = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")
    assert trained.status_code == 200, trained.text
    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate")
    assert response.status_code == 200, response.text
    assert response.json()["validation_comparison"][0]["inference_seconds_per_row"] >= 0


def test_multiclass_and_regression_evaluation_metrics(test_app: FastAPI) -> None:
    multiclass = prepare_experiment(
        test_app, classification_csv(["red", "green", "blue"]), target="target",
        task_type="multiclass_classification", models=["logistic_regression"],
    )
    request(test_app, "POST", f"/api/v1/experiments/{multiclass}/train")
    multi_result = request(test_app, "POST", f"/api/v1/experiments/{multiclass}/evaluate").json()
    assert multi_result["final_test_metrics"]["roc_auc"] is None
    assert multi_result["final_test_confusion_matrix"] is not None

    regression = prepare_experiment(
        test_app, regression_csv(), target="price", task_type="regression", models=["ridge_regression"],
    )
    request(test_app, "POST", f"/api/v1/experiments/{regression}/train")
    regression_result = request(test_app, "POST", f"/api/v1/experiments/{regression}/evaluate").json()
    assert set(regression_result["final_test_metrics"]) == {"mae", "rmse", "r2"}
    assert regression_result["final_test_confusion_matrix"] is None

    prediction = request(test_app, "POST", f"/api/v1/experiments/{regression}/predictions", files={"file": ("future-homes.csv", b"size,rooms,area\n900,3,urban\n1200,4,rural\n", "text/csv")})
    assert prediction.status_code == 200, prediction.text
    regression_report = request(test_app, "POST", f"/api/v1/experiments/{regression}/report")
    assert regression_report.status_code == 200, regression_report.text
    report_html = request(test_app, "GET", regression_report.json()["download_url"]).text
    assert "Target distribution summary" in report_html
    assert "Actual vs predicted values" in report_html
    assert "Final untouched test metrics" in report_html
    assert "Predicted value distribution (supplied rows)" in report_html
    assert "Prediction range" in report_html
    # MAE/RMSE use raw numerical units; R² is the only regression metric shown as a percentage.
    assert "MAE</th><td>" in report_html and "RMSE</th><td>" in report_html


def test_lower_metric_selection_and_stable_tie_breaking() -> None:
    first = ModelRun(id="b", experiment_id="e", pipeline_plan_id="p", model_name="random_forest", status="COMPLETED", parameters={}, training_duration_seconds=0, failure_information=None, artifact_filename="b.joblib")
    second = ModelRun(id="a", experiment_id="e", pipeline_plan_id="p", model_name="logistic_regression", status="COMPLETED", parameters={}, training_duration_seconds=0, failure_information=None, artifact_filename="a.joblib")
    selected, _ = _select_comparison([(first, {"rmse": 1.0}), (second, {"rmse": 1.0})], MetricName.RMSE)
    assert selected.id == "a"


def test_report_metric_formatter_keeps_regression_errors_in_target_units() -> None:
    assert _metric(124924.836, "rmse") == "124,924.836"
    assert _metric(76644.96, "mae") == "76,644.960"
    assert _metric(0.859, "r2") == "85.9%"


def test_optimize_predict_explain_and_report_end_to_end(test_app: FastAPI) -> None:
    experiment_id = prepare_experiment(
        test_app, classification_csv(["yes", "no"]), target="target",
        task_type="binary_classification", models=["logistic_regression"],
    )
    request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")
    baseline_evaluation = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate").json()
    optimized = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/optimize")
    assert optimized.status_code == 200, optimized.text
    assert optimized.json()["trial_count"] <= 10
    reopened = request(test_app, "GET", f"/api/v1/experiments/{experiment_id}").json()
    assert len(reopened["evaluation"]["validation_comparison"]) == len(baseline_evaluation["validation_comparison"])
    assert reopened["evaluation"]["selected_model_run_id"] == baseline_evaluation["selected_model_run_id"]
    explanation = request(test_app, "GET", f"/api/v1/experiments/{experiment_id}/explainability")
    assert explanation.status_code == 200
    assert explanation.json()["global_feature_importance"]
    prediction_csv = b"age,income,segment\n24,35000,A\n42,51000,B\n"
    prediction = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/predictions", files={"file": ("future.csv", prediction_csv, "text/csv")})
    assert prediction.status_code == 200, prediction.text
    download = request(test_app, "GET", prediction.json()["download_url"])
    assert download.status_code == 200 and "prediction" in download.text
    invalid = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/predictions", files={"file": ("bad.csv", b"age,income\n1,2\n", "text/csv")})
    assert invalid.status_code == 400
    report = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/report")
    assert report.status_code == 200, report.text
    report_download = request(test_app, "GET", report.json()["download_url"])
    assert report_download.status_code == 200
    assert "Final untouched test metrics" in report_download.text
    assert "Prediction Outlook for Supplied Rows" in report_download.text
    assert "Verified output summary" in report_download.text
    assert "Predicted class distribution (supplied rows)" in report_download.text
    assert "Model-reported confidence distribution" in report_download.text
    report_history = request(test_app, "GET", "/api/v1/reports")
    assert report_history.status_code == 200
    assert report_history.json()["items"][0]["report_id"] == report.json()["report_id"]
    assert report_history.json()["items"][0]["format"] == "html"
    pdf_report = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/report/pdf")
    assert pdf_report.status_code == 200, pdf_report.text
    pdf_download = request(test_app, "GET", pdf_report.json()["download_url"])
    assert pdf_download.status_code == 200
    assert pdf_download.content.startswith(b"%PDF")


def test_bounded_shap_uses_transformed_feature_names(test_app: FastAPI, monkeypatch) -> None:
    experiment_id = prepare_experiment(
        test_app, classification_csv(["yes", "no"]), target="target",
        task_type="binary_classification", models=["random_forest"],
    )
    assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train").status_code == 200
    assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate").status_code == 200

    class FakeTreeExplainer:
        def __init__(self, _model) -> None:
            pass
        def shap_values(self, values):
            return np.ones((values.shape[0], values.shape[1]))

    monkeypatch.setattr(explainability_service.shap, "TreeExplainer", FakeTreeExplainer)
    response = request(test_app, "GET", f"/api/v1/experiments/{experiment_id}/explainability")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["method"] == "shap"
    assert 0 < payload["sample_size"] <= 200
    assert payload["metadata"]["bounded"] is True
    assert all("__" not in item["feature_name"] for item in payload["global_feature_importance"])


def test_available_titanic_and_iris_report_acceptance(test_app: FastAPI) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixtures = [
        (project_root / "titanic.csv", "Survived", "binary_classification"),
        (project_root / "iris.csv", "variety", "multiclass_classification"),
    ]
    for source, target, task_type in fixtures:
        assert source.is_file(), f"Acceptance fixture unavailable: {source.name}"
        experiment_id = prepare_experiment(test_app, source.read_bytes(), target=target, task_type=task_type, models=["logistic_regression"])
        assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train").status_code == 200
        assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate").status_code == 200
        report = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/report/pdf")
        assert report.status_code == 200, report.text
        document = request(test_app, "GET", report.json()["download_url"])
        assert document.status_code == 200 and document.content.startswith(b"%PDF")
