import asyncio
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response


def request(
    app: FastAPI,
    method: str,
    path: str,
    **kwargs: Any,
) -> Response:
    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def upload_csv(app: FastAPI, content: bytes) -> dict[str, Any]:
    response = request(
        app,
        "POST",
        "/api/v1/datasets",
        files={"file": ("experiment.csv", content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()


def create_experiment(
    app: FastAPI,
    dataset_id: str,
    objective: str,
) -> Response:
    return request(
        app,
        "POST",
        "/api/v1/experiments",
        json={"dataset_id": dataset_id, "objective": objective},
    )


def test_experiment_creation_returns_candidates_and_persists_state(
    test_app: FastAPI,
) -> None:
    dataset = upload_csv(
        test_app,
        b"customer_id,tenure,churn\n1,3,yes\n2,8,no\n3,2,yes\n4,9,no\n",
    )

    response = create_experiment(
        test_app,
        dataset["id"],
        "Predict which customers are likely to churn.",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["dataset_id"] == dataset["id"]
    assert payload["objective"] == "Predict which customers are likely to churn."
    assert payload["suggested_target_column"] == "churn"
    assert payload["suggested_task_type"] == "binary_classification"
    assert payload["confidence"] == "high"
    assert payload["is_ambiguous"] is False
    assert payload["status"] == "AWAITING_CONFIRMATION"
    assert [item["name"] for item in payload["target_candidates"]] == [
        "tenure",
        "churn",
    ]
    assert set(payload["supported_task_types"]) == {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }


def test_experiment_creation_rejects_nonexistent_dataset(test_app: FastAPI) -> None:
    response = create_experiment(test_app, "missing", "Predict churn")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found."}


def test_experiment_creation_rejects_empty_objective(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,target\n1,yes\n2,no\n")

    response = create_experiment(test_app, dataset["id"], "   ")

    assert response.status_code == 422


def test_valid_target_confirmation_marks_experiment_ready(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,churn\n1,yes\n2,no\n3,yes\n")
    experiment = create_experiment(test_app, dataset["id"], "Predict churn").json()

    response = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "churn", "task_type": "binary_classification"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["confirmed_target_column"] == "churn"
    assert payload["confirmed_task_type"] == "binary_classification"
    assert payload["status"] == "READY_FOR_PLANNING"


def test_confirmation_rejects_nonexistent_target(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,target\n1,yes\n2,no\n")
    experiment = create_experiment(test_app, dataset["id"], "Predict target").json()

    response = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "missing", "task_type": "binary_classification"},
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_confirmation_rejects_nonexistent_experiment(test_app: FastAPI) -> None:
    response = request(
        test_app,
        "POST",
        "/api/v1/experiments/missing/confirm",
        json={"target_column": "target", "task_type": "binary_classification"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Experiment not found."}


def test_confirmation_rejects_unsupported_task(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,target\n1,yes\n2,no\n")
    experiment = create_experiment(test_app, dataset["id"], "Predict target").json()

    response = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "target", "task_type": "clustering"},
    )

    assert response.status_code == 422


def test_confirmation_rejects_incompatible_class_structure(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,target\n1,yes\n2,no\n3,yes\n")
    experiment = create_experiment(test_app, dataset["id"], "Predict target").json()

    response = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "target", "task_type": "multiclass_classification"},
    )

    assert response.status_code == 400
    assert "between 3 and 100" in response.json()["detail"]


def test_binary_target_inference(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,outcome\n1,yes\n2,no\n3,yes\n")

    payload = create_experiment(test_app, dataset["id"], "Predict outcome").json()

    assert payload["suggested_task_type"] == "binary_classification"
    assert payload["is_ambiguous"] is False


def test_multiclass_target_inference(test_app: FastAPI) -> None:
    dataset = upload_csv(
        test_app,
        b"feature,species\n1,setosa\n2,versicolor\n3,virginica\n4,setosa\n",
    )

    payload = create_experiment(test_app, dataset["id"], "Predict species").json()

    assert payload["suggested_task_type"] == "multiclass_classification"
    assert payload["is_ambiguous"] is False


def test_regression_target_inference(test_app: FastAPI) -> None:
    dataset = upload_csv(
        test_app,
        b"feature,price\n1,10.5\n2,12.25\n3,14.75\n4,17.125\n",
    )

    payload = create_experiment(test_app, dataset["id"], "Predict price").json()

    assert payload["suggested_task_type"] == "regression"
    assert payload["confidence"] == "high"


def test_low_cardinality_integer_target_is_explicitly_ambiguous(
    test_app: FastAPI,
) -> None:
    dataset = upload_csv(test_app, b"feature,label\n10,0\n20,1\n30,0\n40,1\n")

    payload = create_experiment(test_app, dataset["id"], "Predict label").json()

    assert payload["suggested_target_column"] == "label"
    assert payload["suggested_task_type"] is None
    assert payload["confidence"] == "low"
    assert payload["is_ambiguous"] is True
    candidate = next(
        item for item in payload["target_candidates"] if item["name"] == "label"
    )
    assert candidate["compatible_task_types"] == [
        "binary_classification",
        "regression",
    ]


def test_constant_target_is_not_suggested_and_is_rejected(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"feature,target\n1,same\n2,same\n3,same\n")
    experiment = create_experiment(test_app, dataset["id"], "Predict target").json()

    assert "target" not in {
        candidate["name"] for candidate in experiment["target_candidates"]
    }
    assert experiment["suggested_target_column"] is None

    response = request(
        test_app,
        "POST",
        f"/api/v1/experiments/{experiment['experiment_id']}/confirm",
        json={"target_column": "target", "task_type": "binary_classification"},
    )
    assert response.status_code == 400
    assert "at least two" in response.json()["detail"]
