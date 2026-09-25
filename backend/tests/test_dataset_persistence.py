from pathlib import Path

from fastapi import FastAPI

from app.models.experiment import Experiment
from app.models.user import User
from app.services.security import get_current_user
from tests.test_datasets import request, upload_csv


def test_saved_dataset_lists_and_reloads_from_persisted_file(test_app: FastAPI, test_storage_path: Path) -> None:
    uploaded = upload_csv(
        test_app,
        b"price,bedrooms\n500000,3\n600000,4\n",
        filename="Housing.csv",
    ).json()

    listed = request(test_app, "GET", "/api/v1/datasets")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [uploaded["id"]]
    assert listed.json()["items"][0]["status"] == "READY"
    assert (test_storage_path / "datasets" / uploaded["stored_filename"]).is_file()

    # A new request loads the stored file again, rather than relying on upload-time memory.
    profile = request(test_app, "GET", f"/api/v1/datasets/{uploaded['id']}/profile")
    assert profile.status_code == 200
    assert profile.json()["summary"]["row_count"] == 2

    answer = request(test_app, "POST", "/api/v1/assistant/query", json={"dataset_id": uploaded["id"], "question": "What is the average house price?"})
    assert answer.status_code == 200
    body = answer.json()
    assert "### Mean of price" in body["answer"]
    assert "**Rows Analyzed:** **2**" in body["answer"]
    assert "**Source:** **Housing.csv**" in body["answer"]
    assert "**How calculated:**" in body["answer"]
    assert "### Suggested Analysis" in body["answer"]
    assert body["structured_result"]["value"] == 550000.0


def test_list_is_scoped_and_delete_removes_only_owner_dataset(test_app: FastAPI, test_storage_path: Path) -> None:
    owner_dataset = upload_csv(test_app, b"value\n1\n", filename="owner.csv").json()
    session = test_app.state.testing_session_factory()
    other = User(id="22222222-2222-2222-2222-222222222222", email="other@example.com", password_hash="test-hash")
    session.add(other)
    session.commit()
    session.close()
    test_app.dependency_overrides[get_current_user] = lambda: other
    other_dataset = upload_csv(test_app, b"value\n2\n", filename="other.csv").json()

    listed = request(test_app, "GET", "/api/v1/datasets")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [other_dataset["id"]]
    assert request(test_app, "DELETE", f"/api/v1/datasets/{owner_dataset['id']}").status_code == 404

    deleted = request(test_app, "DELETE", f"/api/v1/datasets/{other_dataset['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"dataset_id": other_dataset["id"], "deleted": True}
    assert not (test_storage_path / "datasets" / other_dataset["stored_filename"]).exists()


def test_delete_refuses_dataset_with_existing_experiment(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"value,target\n1,0\n2,1\n", filename="linked.csv").json()
    session = test_app.state.testing_session_factory()
    session.add(Experiment(id="experiment-1", user_id="11111111-1111-1111-1111-111111111111", dataset_id=dataset["id"], user_objective="Predict target", status="CREATED"))
    session.commit()
    session.close()

    response = request(test_app, "DELETE", f"/api/v1/datasets/{dataset['id']}")
    assert response.status_code == 409
    assert "linked to an experiment" in response.json()["detail"]
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset['id']}").status_code == 200


def test_cascade_delete_removes_dataset_and_linked_experiment(test_app: FastAPI) -> None:
    dataset = upload_csv(test_app, b"value,target\n1,0\n2,1\n", filename="linked-cascade.csv").json()
    session = test_app.state.testing_session_factory()
    session.add(Experiment(id="experiment-cascade", user_id="11111111-1111-1111-1111-111111111111", dataset_id=dataset["id"], user_objective="Predict target", status="CREATED"))
    session.commit()
    session.close()

    response = request(test_app, "DELETE", f"/api/v1/datasets/{dataset['id']}?cascade=true")
    assert response.status_code == 200
    assert request(test_app, "GET", f"/api/v1/datasets/{dataset['id']}").status_code == 404
    session = test_app.state.testing_session_factory()
    assert session.get(Experiment, "experiment-cascade") is None
    session.close()
