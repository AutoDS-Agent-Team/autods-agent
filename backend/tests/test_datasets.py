import asyncio
import math
from pathlib import Path
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


def upload_csv(app: FastAPI, content: bytes, filename: str = "sample.csv") -> Response:
    return request(
        app,
        "POST",
        "/api/v1/datasets",
        files={"file": (filename, content, "text/csv")},
    )


def test_valid_csv_upload_and_metadata_lookup(test_app: FastAPI) -> None:
    response = upload_csv(test_app, b"name,score\nAda,98\nLinus,91\n")

    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["original_filename"] == "sample.csv"
    assert uploaded["row_count"] == 2
    assert uploaded["column_count"] == 2
    assert uploaded["has_header"] is True
    assert uploaded["generated_column_names"] is False
    assert uploaded["file_size"] > 0
    assert uploaded["stored_filename"].endswith(".csv")

    metadata_response = request(
        test_app,
        "GET",
        f"/api/v1/datasets/{uploaded['id']}",
    )
    assert metadata_response.status_code == 200
    assert metadata_response.json() == uploaded


def test_non_csv_upload_is_rejected(test_app: FastAPI) -> None:
    response = request(
        test_app,
        "POST",
        "/api/v1/datasets",
        files={"file": ("notes.txt", b"name,score\nAda,98\n", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only CSV files are supported."}


def test_empty_and_header_only_csv_files_are_rejected(test_app: FastAPI) -> None:
    empty_response = upload_csv(test_app, b"")
    header_only_response = upload_csv(test_app, b"name,score\n")

    assert empty_response.status_code == 400
    assert "empty" in empty_response.json()["detail"].lower()
    assert header_only_response.status_code == 400
    assert "usable row" in header_only_response.json()["detail"].lower()


def test_unparseable_csv_is_rejected(test_app: FastAPI) -> None:
    response = upload_csv(test_app, b'name,score\n"Ada,98\n')

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The uploaded file is not a valid CSV dataset."
    }


def test_oversized_upload_is_rejected(test_app: FastAPI) -> None:
    oversized_content = b"value\n" + (b"1\n" * 600_000)
    response = upload_csv(test_app, oversized_content)

    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]


def test_profile_reports_expected_dataset_and_column_information(
    test_app: FastAPI,
) -> None:
    csv_content = (
        b"id,value,category,constant\n"
        b"1,10,A,fixed\n"
        b"2,20,B,fixed\n"
        b"2,20,B,fixed\n"
        b"4,,A,fixed\n"
        b"5,30,C,fixed\n"
    )
    uploaded = upload_csv(test_app, csv_content).json()

    response = request(
        test_app,
        "GET",
        f"/api/v1/datasets/{uploaded['id']}/profile",
    )

    assert response.status_code == 200
    profile = response.json()
    assert profile["dataset"]["id"] == uploaded["id"]
    assert profile["dataset"]["has_header"] is True
    assert profile["summary"]["row_count"] == 5
    assert profile["summary"]["column_count"] == 4
    assert profile["summary"]["duplicate_row_count"] == 1
    assert profile["summary"]["total_missing_values"] == 1
    assert profile["summary"]["numerical_columns"] == ["id", "value"]
    assert profile["summary"]["categorical_columns"] == ["category", "constant"]
    assert profile["summary"]["constant_columns"] == ["constant"]
    assert profile["summary"]["possible_id_columns"] == ["id"]

    columns = {column["name"]: column for column in profile["columns"]}
    assert columns["value"]["missing_count"] == 1
    assert columns["value"]["missing_percentage"] == 20.0
    assert columns["value"]["numeric_statistics"]["mean"] == 20.0
    assert columns["category"]["top_values"][0] == {"value": "A", "count": 2}
    assert columns["constant"]["is_constant"] is True
    assert columns["id"]["is_possible_id"] is True
    assert len(profile["correlations"]) == 1
    assert math.isfinite(profile["correlations"][0]["coefficient"])


def test_profile_recognizes_consistent_iso_dates_with_a_temporal_name(
    test_app: FastAPI,
) -> None:
    uploaded = upload_csv(
        test_app,
        b"date,value\n2024-01-01,10\n2024-01-02,11\n2024-01-03,12\n",
    ).json()

    response = request(test_app, "GET", f"/api/v1/datasets/{uploaded['id']}/profile")

    assert response.status_code == 200
    columns = {column["name"]: column for column in response.json()["columns"]}
    assert columns["date"]["logical_type"] == "datetime"


def test_headerless_csv_preserves_first_row_and_generates_column_names(
    test_app: FastAPI,
) -> None:
    csv_content = (
        b"5.1,3.5,1.4,0.2,setosa\n"
        b"4.9,3.0,1.4,0.2,setosa\n"
        b"6.5,2.8,4.6,1.5,versicolor\n"
    )
    upload_response = upload_csv(test_app, csv_content, filename="headerless.csv")

    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    assert uploaded["has_header"] is False
    assert uploaded["generated_column_names"] is True
    assert uploaded["row_count"] == 3
    assert uploaded["column_count"] == 5

    profile_response = request(
        test_app,
        "GET",
        f"/api/v1/datasets/{uploaded['id']}/profile",
    )
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert [column["name"] for column in profile["columns"]] == [
        "column_1",
        "column_2",
        "column_3",
        "column_4",
        "column_5",
    ]
    assert profile["summary"]["numerical_columns"] == [
        "column_1",
        "column_2",
        "column_3",
        "column_4",
    ]
    assert profile["summary"]["categorical_columns"] == ["column_5"]

    columns = {column["name"]: column for column in profile["columns"]}
    assert columns["column_1"]["numeric_statistics"]["count"] == 3
    assert columns["column_1"]["numeric_statistics"]["min"] == 4.9
    assert columns["column_5"]["logical_type"] == "categorical"
    assert columns["column_5"]["top_values"][0] == {
        "value": "setosa",
        "count": 2,
    }


def test_single_numeric_value_has_json_safe_statistics(test_app: FastAPI) -> None:
    uploaded = upload_csv(test_app, b"value\n1\n").json()
    response = request(
        test_app,
        "GET",
        f"/api/v1/datasets/{uploaded['id']}/profile",
    )

    assert response.status_code == 200
    statistics = response.json()["columns"][0]["numeric_statistics"]
    assert statistics["std"] is None
    assert "NaN" not in response.text
    assert "Infinity" not in response.text


def test_mostly_numeric_object_columns_are_logically_numerical(
    test_app: FastAPI,
) -> None:
    metadata_row = "150,4,setosa,versicolor,virginica\n"
    data_rows = "".join(
        f"{5 + index / 10:.1f},3.0,1.4,0.2,setosa\n" for index in range(20)
    )
    uploaded = upload_csv(
        test_app,
        (metadata_row + data_rows).encode(),
        filename="metadata-preamble.csv",
    ).json()

    response = request(
        test_app,
        "GET",
        f"/api/v1/datasets/{uploaded['id']}/profile",
    )

    assert response.status_code == 200
    profile = response.json()
    assert profile["dataset"]["has_header"] is False
    assert profile["summary"]["row_count"] == 20
    assert profile["summary"]["numerical_columns"] == [
        "column_1",
        "column_2",
        "column_3",
        "column_4",
    ]
    assert profile["summary"]["categorical_columns"] == ["column_5"]
    columns = {column["name"]: column for column in profile["columns"]}
    assert columns["column_3"]["logical_type"] == "numerical"
    assert columns["column_3"]["numeric_statistics"]["count"] == 20
    assert columns["column_5"]["logical_type"] == "categorical"


def test_nonexistent_dataset_returns_404(test_app: FastAPI) -> None:
    metadata = request(test_app, "GET", "/api/v1/datasets/missing")
    profile = request(test_app, "GET", "/api/v1/datasets/missing/profile")

    assert metadata.status_code == 404
    assert profile.status_code == 404
    assert metadata.json() == {"detail": "Dataset not found."}


def test_numeric_zipcode_is_profiled_as_categorical_code(test_app: FastAPI) -> None:
    uploaded = upload_csv(
        test_app,
        b"zipcode,price\n98101,500000\n98102,620000\n98101,540000\n",
        filename="homes.csv",
    ).json()
    response = request(test_app, "GET", f"/api/v1/datasets/{uploaded['id']}/profile")
    assert response.status_code == 200
    columns = {column["name"]: column for column in response.json()["columns"]}
    assert columns["zipcode"]["logical_type"] == "categorical"
    assert "zipcode" not in response.json()["summary"]["numerical_columns"]


def test_date_named_parseable_column_is_profiled_as_datetime(test_app: FastAPI) -> None:
    uploaded = upload_csv(
        test_app,
        b"sale_date,price\n2024-01-01,500000\n2024-02-01,620000\n2024-03-01,540000\n",
        filename="dated-homes.csv",
    ).json()
    response = request(test_app, "GET", f"/api/v1/datasets/{uploaded['id']}/profile")
    assert response.status_code == 200
    columns = {column["name"]: column for column in response.json()["columns"]}
    assert columns["sale_date"]["logical_type"] == "datetime"


def test_unsafe_filename_cannot_escape_storage_directory(
    test_app: FastAPI,
    test_storage_path: Path,
) -> None:
    response = upload_csv(
        test_app,
        b"name,score\nAda,98\n",
        filename="../../escaped.csv",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["original_filename"] == "escaped.csv"
    stored_files = list((test_storage_path / "datasets").glob("*.csv"))
    assert len(stored_files) == 1
    assert stored_files[0].name == payload["stored_filename"]
    assert not (test_storage_path / "escaped.csv").exists()
