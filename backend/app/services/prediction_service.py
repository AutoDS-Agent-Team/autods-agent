from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import DatasetValidationError, TrainingNotReadyError
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.prediction_run import PredictionRun
from app.schemas.evaluation import PredictionResponse
from app.services.artifact_service import confined_artifact_path
from app.services.evaluation_service import load_model_artifact


def create_predictions(experiment_id: str, content: bytes, filename: str, database: Session, settings: Settings) -> PredictionResponse:
    if not filename.lower().endswith(".csv") or not content:
        raise DatasetValidationError("Provide a non-empty CSV prediction file.")
    experiment = database.get(Experiment, experiment_id)
    if experiment is None or not experiment.selected_model_run_id:
        raise TrainingNotReadyError("Evaluate and select a model before predictions.")
    run = database.get(ModelRun, experiment.selected_model_run_id)
    if run is None:
        raise TrainingNotReadyError("Selected model is unavailable.")
    try:
        dataframe = pd.read_csv(BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError("Prediction file is not a readable CSV.") from error
    artifact = load_model_artifact(run, settings)
    missing = [column for column in artifact.feature_columns if column not in dataframe.columns]
    if missing:
        raise DatasetValidationError(f"Prediction file is missing required feature columns: {', '.join(missing)}.")
    if dataframe.empty:
        raise DatasetValidationError("Prediction file has no data rows.")
    # Extra columns, including an optional historical target column, are ignored explicitly.
    features = dataframe.loc[:, artifact.feature_columns].copy()
    try:
        transformed = artifact.preprocessor.transform(features)
        values = artifact.model.predict(transformed)
    except Exception as error:
        raise DatasetValidationError("Prediction file values are incompatible with the trained pipeline.") from error
    output = pd.DataFrame({"prediction": artifact.target_encoder.inverse_transform(values) if artifact.target_encoder else values})
    if artifact.target_encoder is not None and hasattr(artifact.model, "predict_proba"):
        probabilities = artifact.model.predict_proba(transformed)
        output["confidence"] = probabilities.max(axis=1)
    run_id = str(uuid4())
    artifact_filename = f"{run_id}.csv"
    output.to_csv(confined_artifact_path(settings.prediction_storage_path, artifact_filename, ".csv"), index=False)
    record = PredictionRun(id=run_id, experiment_id=experiment.id, model_run_id=run.id, row_count=len(output), artifact_filename=artifact_filename)
    database.add(record)
    database.commit()
    database.refresh(record)
    return PredictionResponse(prediction_run_id=record.id, experiment_id=experiment.id, model_run_id=run.id, row_count=record.row_count, status="COMPLETED", download_url=f"/api/v1/predictions/{record.id}", created_at=record.created_at)
