import csv
from pathlib import Path, PurePosixPath
from uuid import uuid4
from zipfile import BadZipFile

import pandas as pd
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    DatasetNotFoundError,
    DatasetInUseError,
    DatasetTooLargeError,
    DatasetValidationError,
)
from app.models.dataset import Dataset
from app.models.evaluation_result import EvaluationResult
from app.models.experiment import Experiment
from app.models.job import Job
from app.models.model_run import ModelRun
from app.models.optimization_result import OptimizationResult
from app.models.pipeline_plan import PipelinePlanRecord
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.models.user import User
from app.schemas.dataset import DatasetProfileResponse, DatasetResponse
from app.services.profiling_service import profile_dataframe, read_csv_dataset
from app.services.artifact_service import confined_artifact_path

UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_CSV_CONTENT_TYPES = {
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
}
ALLOWED_EXCEL_CONTENT_TYPES = {
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _safe_original_filename(filename: str | None) -> str:
    normalized = (filename or "dataset.csv").replace("\\", "/")
    basename = PurePosixPath(normalized).name.strip().replace("\x00", "")
    return (basename or "dataset.csv")[:255]


def _storage_root(settings: Settings) -> Path:
    root = settings.dataset_storage_path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stored_path(settings: Settings, stored_filename: str) -> Path:
    root = _storage_root(settings)
    candidate = (root / stored_filename).resolve()
    if candidate.parent != root:
        raise DatasetValidationError("Invalid dataset storage path.")
    return candidate


def _validate_upload_metadata(file: UploadFile) -> str:
    original_filename = _safe_original_filename(file.filename)
    suffix = Path(original_filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise DatasetValidationError("Only CSV files are supported.")
    allowed_types = ALLOWED_CSV_CONTENT_TYPES if suffix == ".csv" else ALLOWED_EXCEL_CONTENT_TYPES
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise DatasetValidationError("The uploaded file type is not supported.")
    return original_filename


async def create_dataset(
    file: UploadFile,
    database: Session,
    settings: Settings,
    user: User,
) -> Dataset:
    original_filename = _validate_upload_metadata(file)
    source_format = Path(original_filename).suffix.lower().lstrip(".")
    stored_filename = f"{uuid4().hex}.{source_format}"
    destination = _stored_path(settings, stored_filename)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_size = 0
    file_created = False

    try:
        with destination.open("xb") as stored_file:
            file_created = True
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise DatasetTooLargeError(
                        f"File exceeds the {settings.max_upload_size_mb} MB upload limit."
                    )
                stored_file.write(chunk)

        if file_size == 0:
            raise DatasetValidationError("The uploaded CSV file is empty.")

        try:
            if source_format == "xlsx":
                workbook = pd.ExcelFile(destination, engine="openpyxl")
                if not workbook.sheet_names:
                    raise DatasetValidationError("The workbook contains no worksheets.")
                worksheet_name = workbook.sheet_names[0]
                dataframe = pd.read_excel(workbook, sheet_name=worksheet_name)
                parsed_csv = None
            else:
                parsed_csv = read_csv_dataset(destination)
                dataframe = parsed_csv.dataframe
                worksheet_name = None
        except (
            csv.Error,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            BadZipFile,
        ) as error:
            message = "The uploaded file is not a valid CSV dataset." if source_format == "csv" else "The uploaded file is not a valid XLSX workbook."
            raise DatasetValidationError(message) from error

        row_count, column_count = dataframe.shape
        if column_count == 0 or row_count == 0 or dataframe.dropna(how="all").empty:
            raise DatasetValidationError(
                "The CSV dataset must contain at least one usable row and one column."
            )

        dataset = Dataset(
            user_id=user.id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_size=file_size,
            row_count=int(row_count),
            column_count=int(column_count),
            has_header=True if source_format == "xlsx" else parsed_csv.has_header,
            source_format=source_format,
            worksheet_name=worksheet_name,
        )
        database.add(dataset)
        database.flush()
        database.refresh(dataset)
        database.commit()
        return dataset
    except Exception:
        database.rollback()
        if file_created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def get_dataset_or_raise(dataset_id: str, database: Session) -> Dataset:
    dataset = database.get(Dataset, dataset_id)
    if dataset is None:
        raise DatasetNotFoundError("Dataset not found.")
    return dataset


def list_user_datasets(user_id: str, database: Session) -> list[Dataset]:
    """Return only the authenticated user's persisted datasets, newest first."""
    statement = select(Dataset).where(Dataset.user_id == user_id).order_by(Dataset.updated_at.desc())
    return list(database.scalars(statement).all())


def delete_dataset(dataset: Dataset, database: Session, settings: Settings, cascade: bool = False) -> None:
    """Delete a dataset; linked experiments are removed only after explicit cascade opt-in."""
    experiment_ids = list(database.scalars(select(Experiment.id).where(Experiment.dataset_id == dataset.id)).all())
    if experiment_ids and not cascade:
        raise DatasetInUseError("This dataset is linked to an experiment and cannot be deleted. Delete the dependent experiment first.")
    stored_path = _stored_path(settings, dataset.stored_filename)
    artifact_paths: list[Path] = []
    if experiment_ids:
        for model in database.scalars(select(ModelRun).where(ModelRun.experiment_id.in_(experiment_ids))).all():
            if model.artifact_filename:
                artifact_paths.append(confined_artifact_path(settings.model_storage_path, model.artifact_filename, ".joblib"))
        for prediction in database.scalars(select(PredictionRun).where(PredictionRun.experiment_id.in_(experiment_ids))).all():
            artifact_paths.append(confined_artifact_path(settings.prediction_storage_path, prediction.artifact_filename, ".csv"))
        for report in database.scalars(select(Report).where(Report.experiment_id.in_(experiment_ids))).all():
            suffix = ".pdf" if report.artifact_filename.endswith(".pdf") else ".html"
            artifact_paths.append(confined_artifact_path(settings.report_storage_path, report.artifact_filename, suffix))
        # Explicit ordering keeps cascade deletion reliable in SQLite tests and PostgreSQL.
        for model in (Job, Report, PredictionRun, OptimizationResult, EvaluationResult):
            database.query(model).filter(model.experiment_id.in_(experiment_ids)).delete(synchronize_session=False)
        database.query(ModelRun).filter(ModelRun.experiment_id.in_(experiment_ids)).delete(synchronize_session=False)
        database.query(PipelinePlanRecord).filter(PipelinePlanRecord.experiment_id.in_(experiment_ids)).delete(synchronize_session=False)
        database.query(Experiment).filter(Experiment.id.in_(experiment_ids)).delete(synchronize_session=False)
    database.delete(dataset)
    try:
        database.commit()
    except Exception:
        database.rollback()
        raise
    # Artifacts are removed only after the database transaction succeeds.
    stored_path.unlink(missing_ok=True)
    for artifact_path in artifact_paths:
        artifact_path.unlink(missing_ok=True)


def get_dataset_metadata(
    dataset_id: str,
    database: Session,
    settings: Settings,
) -> Dataset:
    dataset = get_dataset_or_raise(dataset_id, database)
    if dataset.has_header is None and getattr(dataset, "source_format", "csv") == "csv":
        parsed_csv = read_csv_dataset(
            _stored_path(settings, dataset.stored_filename),
        )
        dataset.has_header = parsed_csv.has_header
        dataset.row_count = int(parsed_csv.dataframe.shape[0])
        dataset.column_count = int(parsed_csv.dataframe.shape[1])
        database.commit()
        database.refresh(dataset)
    return dataset


def get_dataset_profile(
    dataset_id: str,
    database: Session,
    settings: Settings,
) -> DatasetProfileResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    assert dataset.has_header is not None
    dataframe = load_dataset_dataframe(dataset, settings)
    if (dataset.row_count, dataset.column_count) != dataframe.shape:
        dataset.row_count = int(dataframe.shape[0])
        dataset.column_count = int(dataframe.shape[1])
        database.commit()
        database.refresh(dataset)
    summary, columns, correlations = profile_dataframe(
        dataframe,
        top_values_limit=settings.profile_top_values_limit,
    )
    return DatasetProfileResponse(
        dataset=DatasetResponse.model_validate(dataset),
        summary=summary,
        columns=columns,
        correlations=correlations,
    )


def load_dataset_dataframe(dataset: Dataset, settings: Settings) -> pd.DataFrame:
    """Load a persisted dataset using its trusted, persisted header decision."""
    if getattr(dataset, "source_format", "csv") == "xlsx":
        return pd.read_excel(_stored_path(settings, dataset.stored_filename), sheet_name=dataset.worksheet_name or 0, engine="openpyxl")
    assert dataset.has_header is not None
    return read_csv_dataset(
        _stored_path(settings, dataset.stored_filename),
        has_header=dataset.has_header,
    ).dataframe


def worksheet_names(dataset: Dataset, settings: Settings) -> list[str]:
    if getattr(dataset, "source_format", "csv") != "xlsx":
        return []
    return list(pd.ExcelFile(_stored_path(settings, dataset.stored_filename), engine="openpyxl").sheet_names)


def select_worksheet(dataset: Dataset, worksheet_name: str, database: Session, settings: Settings) -> Dataset:
    names = worksheet_names(dataset, settings)
    if worksheet_name not in names:
        raise DatasetValidationError("The selected worksheet does not exist.")
    dataframe = pd.read_excel(_stored_path(settings, dataset.stored_filename), sheet_name=worksheet_name, engine="openpyxl")
    if dataframe.empty or dataframe.shape[1] == 0:
        raise DatasetValidationError("The selected worksheet has no usable tabular data.")
    dataset.worksheet_name = worksheet_name
    dataset.row_count, dataset.column_count = map(int, dataframe.shape)
    database.commit()
    database.refresh(dataset)
    return dataset
