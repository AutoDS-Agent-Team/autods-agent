"""Administrator-invoked, opt-in cleanup of records and confined artifacts."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.services.artifact_service import confined_artifact_path


def _remove(root, filename: str, suffix: str, dry_run: bool) -> None:
    path = confined_artifact_path(root, filename, suffix)
    if path.is_file() and not dry_run:
        path.unlink()


def cleanup_expired_data(database: Session, settings: Settings, now: datetime | None = None, dry_run: bool = False) -> dict[str, int]:
    """Delete only aged persisted records; retention=0 is always a no-op."""
    counts = {"datasets": 0, "experiments": 0, "model_artifacts": 0, "prediction_artifacts": 0, "report_artifacts": 0}
    if settings.data_retention_days <= 0:
        return counts
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=settings.data_retention_days)
    datasets = database.scalars(select(Dataset).where(Dataset.created_at < cutoff)).all()
    experiment_ids = set(database.scalars(select(Experiment.id).where(Experiment.updated_at < cutoff)).all())
    for dataset in datasets:
        experiment_ids.update(database.scalars(select(Experiment.id).where(Experiment.dataset_id == dataset.id)).all())
    for experiment_id in experiment_ids:
        for model in database.scalars(select(ModelRun).where(ModelRun.experiment_id == experiment_id)).all():
            if model.artifact_filename:
                _remove(settings.model_storage_path, model.artifact_filename, ".joblib", dry_run); counts["model_artifacts"] += 1
        for prediction in database.scalars(select(PredictionRun).where(PredictionRun.experiment_id == experiment_id)).all():
            _remove(settings.prediction_storage_path, prediction.artifact_filename, ".csv", dry_run); counts["prediction_artifacts"] += 1
        for report in database.scalars(select(Report).where(Report.experiment_id == experiment_id)).all():
            _remove(settings.report_storage_path, report.artifact_filename, ".pdf" if report.artifact_filename.endswith(".pdf") else ".html", dry_run); counts["report_artifacts"] += 1
        if not dry_run:
            experiment = database.get(Experiment, experiment_id)
            if experiment:
                database.delete(experiment)
        counts["experiments"] += 1
    for dataset in datasets:
        suffix = Path(dataset.stored_filename).suffix.lower()
        if suffix in {".csv", ".xlsx"}:
            _remove(settings.dataset_storage_path, dataset.stored_filename, suffix, dry_run)
        if not dry_run and database.get(Dataset, dataset.id):
            database.delete(dataset)
        counts["datasets"] += 1
    if not dry_run:
        database.commit()
    return counts
