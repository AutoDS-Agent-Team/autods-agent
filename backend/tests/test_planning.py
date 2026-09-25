import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from pydantic import ValidationError

from app.api.routes.experiments import get_llm_router
from app.core.exceptions import PlanningFailureError
from app.llm.base import LLMProvider, ProviderFailure
from app.llm.router import LLMRouter
from app.schemas.pipeline_plan import PipelinePlan


class StubProvider(LLMProvider):
    def __init__(self, name: str, results: list[Any]) -> None:
        self.name = name
        self.results = results
        self.calls = 0

    def generate_structured(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> str | dict[str, Any]:
        assert "SAFE_DATASET_CONTEXT" in prompt
        assert response_schema["type"] == "object"
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def valid_plan(**overrides: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "task_type": "binary_classification",
        "target_column": "churn",
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot",
        "numeric_scaling": "standard",
        "models": ["logistic_regression", "random_forest"],
        "primary_metric": "f1",
    }
    plan.update(overrides)
    return plan


def create_confirmed_experiment(test_app: FastAPI) -> str:
    upload = request(
        test_app,
        "POST",
        "/api/v1/datasets",
        files={
            "file": (
                "planning.csv",
                b"age,segment,churn\n20,A,yes\n30,B,no\n40,A,yes\n50,B,no\n",
                "text/csv",
            )
        },
    ).json()
    experiment = request(
        test_app,
        "POST",
        "/api/v1/experiments",
        json={"dataset_id": upload["id"], "objective": "Predict churn"},
    ).json()
    confirmation = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "churn", "task_type": "binary_classification"},
    )
    assert confirmation.status_code == 200
    return experiment["experiment_id"]


def route_with_router(test_app: FastAPI, router: LLMRouter, experiment_id: str) -> Response:
    test_app.dependency_overrides[get_llm_router] = lambda: router
    return request(test_app, "POST", f"/api/v1/experiments/{experiment_id}/plan")


def test_gemini_valid_structured_result_is_persisted(test_app: FastAPI) -> None:
    experiment_id = create_confirmed_experiment(test_app)
    gemini = StubProvider("gemini", [valid_plan()])
    ollama = StubProvider("ollama", [valid_plan()])

    response = route_with_router(
        test_app, LLMRouter(gemini, ollama, max_retries=0), experiment_id
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_used"] == "gemini"
    assert payload["plan"]["target_column"] == "churn"
    assert payload["schema_version"] == "1.0"
    assert payload["adaptive_decisions"]
    assert {item["model_name"] for item in payload["model_recommendations"]} == set(payload["plan"]["models"])
    assert gemini.calls == 1
    assert ollama.calls == 0


@pytest.mark.parametrize("category", ["provider_error", "timeout", "rate_limited"])
def test_gemini_failures_fall_back_to_ollama(category: str) -> None:
    gemini = StubProvider(
        "gemini", [ProviderFailure("gemini", category, "controlled failure")]
    )
    ollama = StubProvider("ollama", [valid_plan()])

    routed = LLMRouter(gemini, ollama, max_retries=0).generate_plan(
        "SAFE_DATASET_CONTEXT",
        confirmed_task_type="binary_classification",
        confirmed_target_column="churn",
    )

    assert routed.provider_used == "ollama"
    assert routed.plan.primary_metric == "f1"


def test_malformed_gemini_output_falls_back_to_ollama() -> None:
    gemini = StubProvider("gemini", ["not-json"])
    ollama = StubProvider("ollama", [json.dumps(valid_plan())])

    routed = LLMRouter(gemini, ollama, max_retries=0).generate_plan(
        "SAFE_DATASET_CONTEXT",
        confirmed_task_type="binary_classification",
        confirmed_target_column="churn",
    )

    assert routed.provider_used == "ollama"


def test_both_providers_fail_safely() -> None:
    gemini = StubProvider(
        "gemini", [ProviderFailure("gemini", "unavailable", "failed")]
    )
    ollama = StubProvider(
        "ollama", [ProviderFailure("ollama", "unavailable", "failed")]
    )

    with pytest.raises(PlanningFailureError, match="no provider returned"):
        LLMRouter(gemini, ollama, max_retries=0).generate_plan(
            "SAFE_DATASET_CONTEXT",
            confirmed_task_type="binary_classification",
            confirmed_target_column="churn",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"models": ["arbitrary_model"]},
        {"primary_metric": "arbitrary_metric"},
        {"models": ["ridge_regression"]},
        {"primary_metric": "rmse"},
    ],
)
def test_unsupported_or_incompatible_plan_values_are_rejected(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        PipelinePlan.validate_provider_output(
            valid_plan(**overrides),
            confirmed_task_type="binary_classification",
            confirmed_target_column="churn",
        )


def test_llm_cannot_modify_confirmed_target() -> None:
    with pytest.raises(ValidationError, match="confirmed experiment target"):
        PipelinePlan.validate_provider_output(
            valid_plan(target_column="invented"),
            confirmed_task_type="binary_classification",
            confirmed_target_column="churn",
        )


def test_llm_cannot_modify_confirmed_task() -> None:
    with pytest.raises(ValidationError, match="confirmed experiment task"):
        PipelinePlan.validate_provider_output(
            valid_plan(
                task_type="regression",
                models=["ridge_regression"],
                primary_metric="rmse",
            ),
            confirmed_task_type="binary_classification",
            confirmed_target_column="churn",
        )


def test_existing_plan_is_returned_without_calling_provider_again(
    test_app: FastAPI,
) -> None:
    experiment_id = create_confirmed_experiment(test_app)
    gemini = StubProvider("gemini", [valid_plan()])
    router = LLMRouter(gemini, StubProvider("ollama", [valid_plan()]), max_retries=0)

    first = route_with_router(test_app, router, experiment_id)
    second = route_with_router(test_app, router, experiment_id)

    assert first.status_code == second.status_code == 200
    assert first.json()["plan_id"] == second.json()["plan_id"]
    assert gemini.calls == 1


def test_class_weight_override_is_manual_validated_and_persisted(test_app: FastAPI) -> None:
    experiment_id = create_confirmed_experiment(test_app)
    route_with_router(test_app, LLMRouter(StubProvider("gemini", [valid_plan()]), StubProvider("ollama", [valid_plan()]), max_retries=0), experiment_id)
    response = request(test_app, "PATCH", f"/api/v1/experiments/{experiment_id}/plan/adaptive", json={"balanced_class_weight_models": ["random_forest"]})
    assert response.status_code == 200
    assert response.json()["plan"]["balanced_class_weight_models"] == ["random_forest"]
    assert request(test_app, "PATCH", f"/api/v1/experiments/{experiment_id}/plan/adaptive", json={"balanced_class_weight_models": ["xgboost"]}).status_code == 400


def test_confirmation_cannot_change_after_plan_persistence(test_app: FastAPI) -> None:
    experiment_id = create_confirmed_experiment(test_app)
    response = route_with_router(
        test_app,
        LLMRouter(
            StubProvider("gemini", [valid_plan()]),
            StubProvider("ollama", [valid_plan()]),
            max_retries=0,
        ),
        experiment_id,
    )
    assert response.status_code == 200

    changed = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment_id}/confirm",
        json={"target_column": "age", "task_type": "regression"},
    )

    assert changed.status_code == 400
    assert "cannot change" in changed.json()["detail"]


def test_planning_before_confirmation_is_rejected(test_app: FastAPI) -> None:
    upload = request(
        test_app,
        "POST",
        "/api/v1/datasets",
        files={"file": ("raw.csv", b"x,target\n1,A\n2,B\n", "text/csv")},
    ).json()
    experiment = request(
        test_app,
        "POST",
        "/api/v1/experiments",
        json={"dataset_id": upload["id"], "objective": "Predict target"},
    ).json()
    router = LLMRouter(
        StubProvider("gemini", [valid_plan()]),
        StubProvider("ollama", [valid_plan()]),
        max_retries=0,
    )

    response = route_with_router(test_app, router, experiment["experiment_id"])

    assert response.status_code == 409
    assert "Confirm" in response.json()["detail"]
