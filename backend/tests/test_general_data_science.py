import asyncio
from io import BytesIO
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from pydantic import ValidationError

from app.ml.model_registry import MODEL_REGISTRY, require_registered
from app.schemas.analytics import AnalyticsQuery
from app.schemas.auto_analysis import AnalysisTask
from app.services.auto_analysis_service import analyze_dataset, detect_feature_roles
from app.services.specialized_analysis_service import detect_anomalies, forecast_series
from app.schemas.specialized_analysis import AnomalyRequest, ForecastRequest


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def upload(app: FastAPI, content: bytes, name: str = "data.csv", content_type: str = "text/csv") -> dict:
    response = request(app, "POST", "/api/v1/datasets", files={"file": (name, content, content_type)})
    assert response.status_code == 201, response.text
    return response.json()


def test_titanic_auto_detection() -> None:
    frame = pd.DataFrame({"PassengerId": range(1, 31), "Age": range(20, 50), "Fare": range(30), "Survived": [0, 1] * 15})
    result = analyze_dataset("d", frame)
    assert result.target_recommendation.recommended_target == "Survived"
    assert result.target_recommendation.recommended_task == AnalysisTask.BINARY_CLASSIFICATION
    assert next(item for item in result.feature_roles if item.column == "PassengerId").role == "identifier"


def test_iris_and_regression_auto_detection() -> None:
    iris = pd.DataFrame({"sepal_length": [5.0 + index / 10 for index in range(30)], "petal_length": [1.0 + index / 10 for index in range(30)], "variety": ["setosa", "versicolor", "virginica"] * 10})
    result = analyze_dataset("i", iris)
    assert result.target_recommendation.recommended_target == "variety"
    assert result.target_recommendation.recommended_task == AnalysisTask.MULTICLASS_CLASSIFICATION
    houses = pd.DataFrame({"area": range(100, 130), "rooms": [2, 3, 4] * 10, "HousePrice": [100000 + index * 1250.5 for index in range(30)]})
    assert analyze_dataset("h", houses).target_recommendation.recommended_task == AnalysisTask.REGRESSION


def test_no_target_does_not_force_supervised() -> None:
    frame = pd.DataFrame({"age": range(30, 60), "region": ["north", "south", "east"] * 10, "spend": range(100, 130)})
    result = analyze_dataset("d", frame)
    assert result.target_recommendation.recommended_target is None
    assert AnalysisTask.CLUSTERING in result.applicable_tasks
    assert AnalysisTask.ANOMALY_DETECTION in result.applicable_tasks


def test_text_and_identifier_roles() -> None:
    frame = pd.DataFrame({"Review ID": [f"R-{index:04d}" for index in range(30)], "Review Text": [f"A detailed customer review containing several useful words number {index}" for index in range(30)], "Rating": [1, 2, 3, 4, 5, 4] * 5})
    roles = {item.column: item.role for item in detect_feature_roles(frame)}
    assert roles["Review ID"] == "identifier"
    assert roles["Review Text"] == "text"


def test_registry_is_task_scoped() -> None:
    assert "logistic_regression" in MODEL_REGISTRY
    assert require_registered("ridge_regression", AnalysisTask.REGRESSION).prediction_support
    with pytest.raises(ValueError):
        require_registered("ridge_regression", AnalysisTask.BINARY_CLASSIFICATION)
    with pytest.raises(ValueError):
        require_registered("arbitrary.module.Model", AnalysisTask.REGRESSION)


@pytest.mark.parametrize("payload", [
    {"operation": "aggregate", "column": "Age", "aggregation": "eval"},
    {"operation": "row_count", "filters": [{"column": "Age", "operator": "execute_python", "value": 1}]},
    {"operation": "value_counts", "column": "x", "limit": 1000},
])
def test_analytics_schema_rejects_unsafe_or_unbounded_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AnalyticsQuery.model_validate(payload)


def test_dataset_analytics_operations(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"Age,Survived,Pclass,Fare\n22,0,3,7.25\n38,1,1,71.28\n26,1,3,7.92\n35,1,1,53.1\n,0,2,8.0\n")
    base = f"/api/v1/datasets/{dataset['id']}/analytics/query"
    assert request(test_app, "POST", base, json={"operation": "row_count"}).json()["value"] == 5
    assert request(test_app, "POST", base, json={"operation": "missing_count", "column": "Age"}).json()["value"] == 1
    mean = request(test_app, "POST", base, json={"operation": "aggregate", "column": "Age", "aggregation": "mean", "filters": [{"column": "Survived", "operator": "eq", "value": 1}]}).json()
    assert mean["value"] == 33.0
    grouped = request(test_app, "POST", base, json={"operation": "group_aggregate", "column": "Survived", "group_by": ["Pclass"], "aggregation": "mean", "sort": "descending", "limit": 2}).json()
    assert grouped["rows"][0]["label"] == "1"
    assert request(test_app, "POST", base, json={"operation": "correlation"}).status_code == 200
    assert request(test_app, "POST", base, json={"operation": "mean", "column": "Age"}).status_code == 422
    assert request(test_app, "POST", base, json={"operation": "aggregate", "column": "unknown", "aggregation": "mean"}).status_code == 400


def test_dataset_assistant_uses_trusted_calculation(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"Age,Survived\n20,0\n30,1\n40,1\n")
    response = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": "What is the average Age?"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "30" in payload["answer"]
    assert payload["provider_used"] == "trusted_analytics"
    assert payload["structured_result"]["calculation"]
    filtered = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": "What is the average Age of survivors?"})
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["structured_result"]["value"] == 35.0


def test_ask_dataset_context_never_routes_to_experiment_retrieval(test_app: FastAPI) -> None:
    csv_rows = [
        "Contract,PlanType,Churn,MonthlyCharges,SupportCalls,SatisfactionScore,LatePayments",
        "Month-to-month,Basic,Yes,50,3,2,2", "Month-to-month,Basic,Yes,60,4,1,2",
        "Month-to-month,Pro,Yes,70,3,2,2", "Month-to-month,Pro,No,80,0,4,0",
        "One year,Basic,Yes,55,3,2,2", "One year,Pro,Yes,65,2,2,2",
        "One year,Basic,No,75,0,4,0", "One year,Pro,No,85,0,4,0",
        "Two year,Basic,Yes,45,4,1,2", "Two year,Basic,No,52,1,4,0",
        "Two year,Pro,No,62,0,4,0", "Two year,Pro,No,72,0,4,0",
    ]
    dataset = upload(test_app, ("\n".join(csv_rows) + "\n").encode(), "autods_stress_test_customer_churn.csv")
    # The mere existence of a linked experiment must not change dataset-context routing.
    experiment = request(test_app, "POST", "/api/v1/experiments", json={"dataset_id": dataset["id"], "objective": "Review customer churn"}).json()
    assert experiment["dataset_id"] == dataset["id"]

    questions = [
        ("How many customers churned?", 6),
        ("What is the churn rate?", 0.5),
        ("Which contract type has the highest churn rate?", "Month-to-month"),
        ("Compare churn rates across plan types.", {"Basic": 2 / 3, "Pro": 1 / 3}),
        ("Show the distribution of monthly charges.", "histogram"),
        ("Among customers with more than 2 support calls, which contract type has the highest churn rate?", "Month-to-month"),
        ("Among customers with satisfaction score below 3 and more than 1 late payment, compare churn rates by contract type.", {"Month-to-month", "One year", "Two year"}),
    ]
    for question, expected in questions:
        response = request(test_app, "POST", "/api/v1/assistant/query", json={"question": question, "context_type": "dataset", "dataset_id": dataset["id"]})
        assert response.status_code == 200, response.text
        payload = response.json()
        result = payload["structured_result"]
        assert payload["provider_used"] == "trusted_analytics"
        assert payload["sources"][0]["dataset_id"] == dataset["id"]
        if isinstance(expected, int | float):
            assert result["value"] == pytest.approx(expected)
        elif isinstance(expected, str) and expected == "histogram":
            assert result["chart"]["chart_type"] == expected
        elif isinstance(expected, str):
            assert result["rows"][0]["label"] == expected
        elif isinstance(expected, dict):
            assert {row["label"]: row["value"] for row in result["rows"]} == pytest.approx(expected)
        else:
            assert {row["label"] for row in result["rows"]} == expected
        assert result["plan"]["steps"]
        if "support calls" in question:
            steps = result["plan"]["steps"]
            assert [step["operation"] for step in steps] == ["filter", "group_aggregate", "sort_limit"]
            assert [(condition["column"], condition["operator"], condition["value"]) for condition in steps[0]["filters"]] == [("SupportCalls", "gt", 2.0)]
        if "satisfaction score" in question:
            steps = result["plan"]["steps"]
            assert [step["operation"] for step in steps] == ["filter", "group_aggregate"]
            assert [(condition["column"], condition["operator"], condition["value"]) for condition in steps[0]["filters"]] == [("SatisfactionScore", "lt", 3.0), ("LatePayments", "gt", 1.0)]

    mismatched_context = request(test_app, "POST", "/api/v1/assistant/query", json={"question": "How many customers churned?", "context_type": "experiment", "dataset_id": dataset["id"]})
    assert mismatched_context.status_code == 422


def test_ranked_group_question_builds_grouped_verified_result(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"product,rating\nA,4\nA,5\nB,3\nB,2\n")
    response = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": "Which product has the highest average rating?"})
    assert response.status_code == 200, response.text
    result = response.json()["structured_result"]
    assert result["rows"][0] == {"label": "A", "value": 4.5}
    assert {row["label"] for row in result["rows"]} == {"A", "B"}


def test_open_ended_multi_step_dataset_questions(test_app: FastAPI) -> None:
    customers = upload(test_app, b"Country,Age,Revenue\nUS,40,100\nUS,25,80\nIN,35,140\nIN,45,160\n")
    ranked = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": customers["id"], "question": "Which country has the highest average revenue among customers older than 30?"})
    assert ranked.status_code == 200, ranked.text
    result = ranked.json()["structured_result"]
    assert result["rows"][0] == {"label": "IN", "value": 150.0}
    assert {row["label"] for row in result["rows"]} == {"IN", "US"}
    assert [step["operation"] for step in result["plan"]["steps"]] == ["filter", "group_aggregate", "sort_limit"]

    titanic = upload(test_app, b"Sex,Pclass,Survived\nfemale,1,1\nfemale,1,1\nmale,1,0\nmale,2,1\n")
    compared = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": titanic["id"], "question": "Compare survival between males and females in first class."})
    assert compared.status_code == 200, compared.text
    assert {row["label"]: row["value"] for row in compared.json()["structured_result"]["rows"]} == {"female": 1.0, "male": 0.0}

    association = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": customers["id"], "question": "Which features are most associated with Revenue?"})
    assert association.status_code == 200, association.text
    assert association.json()["structured_result"]["plan"]["steps"][0]["operation"] == "association"


def test_titanic_assistant_questions_are_verified(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"PassengerId,Pclass,Sex,Age,Survived\n1,1,female,29,1\n2,1,male,35,0\n3,2,female,22,1\n4,2,male,40,1\n5,3,male,18,0\n")
    questions = {
        "How many passengers survived?": ("scalar", 3),
        "Which class had the highest survival rate?": ("chart", "2"),
        "Compare male vs female survival.": ("chart", "female"),
        "Show Age distribution.": ("chart", None),
        "Which features are associated with Survived?": ("chart", None),
    }
    for question, (answer_type, expected) in questions.items():
        response = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": question})
        assert response.status_code == 200, response.text
        result = response.json()["structured_result"]
        assert result["answer_type"] == answer_type
        if isinstance(expected, int):
            assert result["value"] == expected
        elif expected:
            assert result["rows"][0]["label"] == expected


def test_complex_titanic_questions_use_bounded_multistep_plans(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"Pclass,Sex,Age,Fare,Survived\n1,female,38,71,1\n1,male,40,50,0\n2,female,30,30,1\n2,male,35,20,1\n3,female,20,10,0\n")
    questions = [
        ("Among passengers older than 30, which passenger class had the highest survival rate?", "Survived", ["Age"]),
        ("Among female passengers older than 25, which class had the highest average Fare?", "Fare", ["Sex", "Age"]),
    ]
    for question, target, filter_columns in questions:
        response = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": question})
        assert response.status_code == 200, response.text
        plan = response.json()["structured_result"]["plan"]["steps"]
        assert [step["operation"] for step in plan] == ["filter", "group_aggregate", "sort_limit"]
        assert [item["column"] for item in plan[0]["filters"]] == filter_columns
        assert plan[1]["column"] == target
        assert plan[1]["group_by"] == ["Pclass"]


def test_unsupported_dataset_question_is_declined(test_app: FastAPI) -> None:
    dataset = upload(test_app, b"x,y\n1,2\n")
    response = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": dataset["id"], "question": "Write and run Python code to read another file."})
    assert response.status_code == 200
    assert response.json()["answer"] == "I can't answer that reliably from the available dataset with the currently supported analytical operations."


def test_excel_upload_and_worksheet_selection(test_app: FastAPI) -> None:
    content = BytesIO()
    with pd.ExcelWriter(content, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"city": ["A", "B", "C"], "sales": [10, 20, 30]}).to_excel(writer, sheet_name="Sales", index=False)
    dataset = upload(test_app, content.getvalue(), "workbook.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert dataset["source_format"] == "xlsx"
    assert dataset["worksheet_names"] == ["First", "Sales"]
    selected = request(test_app, "PUT", f"/api/v1/datasets/{dataset['id']}/worksheet", json={"worksheet_name": "Sales"})
    assert selected.status_code == 200
    assert selected.json()["row_count"] == 3
    assert request(test_app, "PUT", f"/api/v1/datasets/{dataset['id']}/worksheet", json={"worksheet_name": "Missing"}).status_code == 400


def test_anomaly_and_time_series_are_bounded_and_deterministic() -> None:
    anomaly_frame = pd.DataFrame({"x": [1.0] * 19 + [100.0], "y": list(range(20))})
    anomalies = detect_anomalies("d", anomaly_frame, AnomalyRequest(contamination=0.05), 42)
    assert anomalies.anomaly_count >= 1
    assert anomalies.analyzed_rows == 20
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    forecast = forecast_series("d", pd.DataFrame({"date": dates, "sales": range(20)}), ForecastRequest(time_column="date", target_column="sales"))
    assert forecast.chronological_split
    assert forecast.test_rows == 4
    assert forecast.metrics.mae >= 0


def test_customer_reviews_text_workflow(test_app: FastAPI) -> None:
    rows = ["Review ID,Review Text,Product,Rating"]
    products = ["Phone", "Tablet", "Laptop"]
    sentiments = ["terrible", "poor", "average", "good", "excellent"]
    for index in range(75):
        rating = index % 5 + 1
        rows.append(f"R-{index:04d},This product review is {sentiments[rating - 1]} and contains useful detail,{products[index % 3]},{rating}")
    dataset = upload(test_app, ("\n".join(rows) + "\n").encode(), "reviews.csv")
    analysis = request(test_app, "GET", f"/api/v1/datasets/{dataset['id']}/auto-analysis").json()
    assert analysis["target_recommendation"]["recommended_target"] == "Rating"
    roles = {item["column"]: item["role"] for item in analysis["feature_roles"]}
    assert roles["Review ID"] == "identifier"
    assert roles["Review Text"] == "text"
    experiment = request(test_app, "POST", "/api/v1/experiments", json={"dataset_id": dataset["id"], "objective": "Predict Rating"}).json()
    experiment_id = experiment["experiment_id"]
    assert request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/confirm", json={"target_column": "Rating", "task_type": "multiclass_classification"}).status_code == 200
    planned = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/plan")
    assert planned.status_code == 200, planned.text
    assert "Review ID" in planned.json()["plan"]["excluded_columns"]
    assert "Review Text" in planned.json()["plan"]["text_columns"]
    trained = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")
    assert trained.status_code == 200, trained.text
    assert trained.json()["status"] in {"COMPLETED", "PARTIAL_FAILURE"}
    evaluated = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/evaluate")
    assert evaluated.status_code == 200, evaluated.text
