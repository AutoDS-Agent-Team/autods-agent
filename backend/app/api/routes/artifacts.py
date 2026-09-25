from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import TrainingNotReadyError
from app.db.dependencies import get_db
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.services.artifact_service import confined_artifact_path
from app.models.experiment import Experiment
from app.models.user import User
from app.services.security import get_current_user
from app.schemas.evaluation import ReportHistoryItem, ReportHistoryResponse

router = APIRouter(tags=["artifacts"])


@router.get("/reports", response_model=ReportHistoryResponse)
def list_report_definitions(
    database: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> ReportHistoryResponse:
    rows = database.execute(
        select(Report, Experiment.user_objective)
        .join(Experiment, Report.experiment_id == Experiment.id)
        .where(Experiment.user_id == user.id)
        .order_by(Report.created_at.desc())
    ).all()
    return ReportHistoryResponse(items=[
        ReportHistoryItem(
            report_id=report.id,
            experiment_id=report.experiment_id,
            objective=objective,
            format="pdf" if report.artifact_filename.endswith(".pdf") else "html",
            download_url=f"/api/v1/reports/{report.id}",
            created_at=report.created_at,
        )
        for report, objective in rows
    ])


@router.get("/predictions/{prediction_run_id}")
def download_predictions_definition(
    prediction_run_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> FileResponse:
    record = database.get(PredictionRun, prediction_run_id)
    if record is None:
        raise TrainingNotReadyError("Prediction result not found.")
    experiment = database.get(Experiment, record.experiment_id)
    if experiment is None or experiment.user_id != user.id:
        raise TrainingNotReadyError("Prediction result not found.")
    path = confined_artifact_path(settings.prediction_storage_path, record.artifact_filename, ".csv")
    if not path.is_file():
        raise TrainingNotReadyError("Prediction artifact is unavailable.")
    return FileResponse(path, media_type="text/csv", filename="predictions.csv")


@router.get("/reports/{report_id}")
def download_report_definition(
    report_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> FileResponse:
    record = database.get(Report, report_id)
    if record is None:
        raise TrainingNotReadyError("Report not found.")
    experiment = database.get(Experiment, record.experiment_id)
    if experiment is None or experiment.user_id != user.id:
        raise TrainingNotReadyError("Report not found.")
    extension = ".pdf" if record.artifact_filename.endswith(".pdf") else ".html"
    path = confined_artifact_path(settings.report_storage_path, record.artifact_filename, extension)
    if not path.is_file():
        raise TrainingNotReadyError("Report artifact is unavailable.")
    return FileResponse(path, media_type="application/pdf" if extension == ".pdf" else "text/html", filename=f"autods-agent-report{extension}")
