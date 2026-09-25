import asyncio
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.api.routes.experiments import get_llm_router
from app.core.exceptions import TrainingFailureError
from app.llm.base import LLMProvider
from app.llm.router import LLMRouter
from app.ml.registry import (
    NUMERIC_IMPUTERS,
    classification_model_factories,
    trusted_factory,
)
from app.schemas.pipeline_plan import PipelinePlan
from app.services import training_service
from app.services.training_service import _prepare_features, build_preprocessor, split_dataset


class StaticProvider(LLMProvider):
    name = "gemini"

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def generate_structured(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        return self.plan


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def plan_payload(
    task_type: str,
    target: str,
    models: list[str],
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "target_column": target,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot",
        "numeric_scaling": "standard",
        "models": models,
        "primary_metric": "f1" if task_type != "regression" else "rmse",
    }


def prepare_experiment(
    app: FastAPI,
    csv_content: bytes,
    *,
    target: str,
    task_type: str,
    models: list[str],
    generate_plan: bool = True,
) -> str:
    dataset = request(
        app,
        "POST",
        "/api/v1/datasets",
        files={"file": ("training.csv", csv_content, "text/csv")},
    ).json()
    experiment = request(
        app,
        "POST",
        "/api/v1/experiments",
        json={"dataset_id": dataset["id"], "objective": f"Predict {target}"},
    ).json()
    confirmation = request(
        app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": target, "task_type": task_type},
    )
    assert confirmation.status_code == 200
    if generate_plan:
        provider = StaticProvider(plan_payload(task_type, target, models))
        app.dependency_overrides[get_llm_router] = lambda: LLMRouter(
            provider, provider, max_retries=0
        )
        planned = request(
            app,
            "POST",
            f"/api/v1/experiments/{experiment['experiment_id']}/plan",
        )
        assert planned.status_code == 200, planned.text
    return experiment["experiment_id"]


def classification_csv(classes: list[str], rows: int = 60) -> bytes:
    content = ["age,income,segment,target"]
    for index in range(rows):
        income = "" if index == 5 else str(30000 + index * 100)
        segment = "" if index == 7 else ("A" if index % 2 else "B")
        content.append(f"{20 + index},{income},{segment},{classes[index % len(classes)]}")
    return ("\n".join(content) + "\n").encode()


def regression_csv(rows: int = 60) -> bytes:
    content = ["size,rooms,area,price"]
    for index in range(rows):
        rooms = "" if index == 8 else str(1 + index % 5)
        area = "urban" if index % 2 else "rural"
        content.append(f"{500 + index * 10},{rooms},{area},{75000.5 + index * 1250.25}")
    return ("\n".join(content) + "\n").encode()


def test_binary_classification_trains_all_requested_baselines(
    test_app: FastAPI,
) -> None:
    experiment_id = prepare_experiment(
        test_app,
        classification_csv(["yes", "no"]),
        target="target",
        task_type="binary_classification",
        models=["logistic_regression", "random_forest", "xgboost"],
    )

    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    assert [run["model_name"] for run in payload["model_runs"]] == [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]
    assert all(run["status"] == "COMPLETED" for run in payload["model_runs"])
    assert sum(payload["split"][key] for key in ("training_rows", "validation_rows", "test_rows")) == 60


def test_multiclass_classification_training(test_app: FastAPI) -> None:
    experiment_id = prepare_experiment(
        test_app,
        classification_csv(["red", "green", "blue"]),
        target="target",
        task_type="multiclass_classification",
        models=["logistic_regression"],
    )

    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")

    assert response.status_code == 200
    assert response.json()["model_runs"][0]["status"] == "COMPLETED"


def test_regression_training(test_app: FastAPI) -> None:
    experiment_id = prepare_experiment(
        test_app,
        regression_csv(),
        target="price",
        task_type="regression",
        models=["ridge_regression", "random_forest", "xgboost"],
    )

    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "COMPLETED"


def test_preprocessor_imputes_encodes_scales_and_handles_unseen_categories() -> None:
    plan = PipelinePlan.model_validate(
        plan_payload("binary_classification", "target", ["logistic_regression"])
    )
    preprocessor = build_preprocessor(plan, ["number"], ["category"])
    train = pd.DataFrame(
        {"number": [1.0, np.nan, 3.0], "category": ["A", None, "B"]}
    )
    unseen = pd.DataFrame({"number": [np.nan], "category": ["never-seen"]})

    transformed_train = preprocessor.fit_transform(train)
    transformed_unseen = preprocessor.transform(unseen)

    assert transformed_train.shape[0] == 3
    assert transformed_unseen.shape[0] == 1
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    assert numeric_pipeline.named_steps["imputer"].statistics_[0] == 2.0
    assert numeric_pipeline.named_steps["scaler"].mean_[0] == 2.0


def test_id_like_text_column_is_excluded_before_text_preprocessing() -> None:
    dataframe = pd.DataFrame(
        {
            "Name": [f"Passenger number {index} with a unique full name" for index in range(24)],
            "Age": list(range(20, 44)),
            "Survived": [index % 2 for index in range(24)],
        }
    )

    features, _, numeric_columns, _, text_columns, excluded_columns = _prepare_features(
        dataframe, "Survived"
    )

    assert "Name" in excluded_columns
    assert "Name" not in text_columns
    assert "Name" not in features.columns
    assert numeric_columns == ["Age"]


def test_target_is_excluded_and_preprocessing_is_fit_on_training_only(
    test_app: FastAPI,
    test_storage_path: Path,
) -> None:
    experiment_id = prepare_experiment(
        test_app,
        regression_csv(40),
        target="price",
        task_type="regression",
        models=["ridge_regression"],
    )
    payload = request(
        test_app, "POST", f"/api/v1/experiments/{experiment_id}/train"
    ).json()
    artifact = joblib.load(
        test_storage_path / "models" / payload["model_runs"][0]["artifact_filename"]
    )

    assert "price" not in artifact["feature_columns"]
    assert set(artifact["train_indices"]).isdisjoint(artifact["validation_indices"])
    assert set(artifact["train_indices"]).isdisjoint(artifact["test_indices"])
    values = pd.Series([500 + index * 10 for index in range(40)])
    expected_train_mean = values.iloc[artifact["train_indices"]].mean()
    numeric_pipeline = artifact["preprocessor"].named_transformers_["numeric"]
    assert numeric_pipeline.named_steps["scaler"].mean_[0] == expected_train_mean


def test_unsupported_registry_operation_cannot_execute() -> None:
    with pytest.raises(TrainingFailureError, match="Unsupported trusted numeric imputer"):
        trusted_factory(NUMERIC_IMPUTERS, "execute_python", "numeric imputer")


def test_unsupported_model_cannot_execute() -> None:
    with pytest.raises(TrainingFailureError, match="Unsupported trusted model"):
        trusted_factory(
            classification_model_factories(42), "arbitrary_import.Model", "model"
        )


def test_training_without_validated_plan_is_rejected(test_app: FastAPI) -> None:
    experiment_id = prepare_experiment(
        test_app,
        classification_csv(["yes", "no"], rows=20),
        target="target",
        task_type="binary_classification",
        models=["logistic_regression"],
        generate_plan=False,
    )

    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")

    assert response.status_code == 409
    assert "validated pipeline plan" in response.json()["detail"]


def test_individual_model_failure_isolated(
    test_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = prepare_experiment(
        test_app,
        classification_csv(["yes", "no"]),
        target="target",
        task_type="binary_classification",
        models=["logistic_regression", "random_forest"],
    )
    original = training_service.classification_model_factories

    def factories(random_state: int) -> dict[str, Any]:
        values = original(random_state)

        def fail() -> Any:
            raise RuntimeError("synthetic model failure")

        values["logistic_regression"] = fail
        return values

    monkeypatch.setattr(training_service, "classification_model_factories", factories)

    response = request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/train")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PARTIAL_FAILURE"
    assert [run["status"] for run in payload["model_runs"]] == ["FAILED", "COMPLETED"]
    assert "RuntimeError" in payload["model_runs"][0]["failure_information"]


def test_fixed_random_state_produces_reproducible_splits() -> None:
    features = pd.DataFrame({"value": range(50), "category": ["A", "B"] * 25})
    target = pd.Series(["yes", "no"] * 25)

    first = split_dataset(
        features,
        target,
        classification=True,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    second = split_dataset(
        features,
        target,
        classification=True,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    assert first.x_train.index.tolist() == second.x_train.index.tolist()
    assert first.x_validation.index.tolist() == second.x_validation.index.tolist()
    assert first.x_test.index.tolist() == second.x_test.index.tolist()


def test_split_falls_back_cleanly_when_stratification_is_impossible() -> None:
    features = pd.DataFrame({"value": range(10)})
    target = pd.Series(["rare"] + ["common"] * 9)

    splits = split_dataset(
        features,
        target,
        classification=True,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    assert splits.stratified is False
    assert len(splits.x_train) + len(splits.x_validation) + len(splits.x_test) == 10
