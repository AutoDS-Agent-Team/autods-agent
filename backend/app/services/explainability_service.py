from __future__ import annotations

import numpy as np
import joblib
import shap
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import TrainingNotReadyError
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.schemas.evaluation import ExplainabilityResponse, FeatureImportance
from app.services.evaluation_service import load_model_artifact
from app.services.evaluation_service import model_artifact_path
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.training_service import _prepare_features


def _readable_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def get_explainability(experiment_id: str, database: Session, settings: Settings) -> ExplainabilityResponse:
    experiment = database.get(Experiment, experiment_id)
    if experiment is None:
        raise TrainingNotReadyError("Experiment not found.")
    if not experiment.selected_model_run_id:
        raise TrainingNotReadyError("Evaluate models before requesting explainability.")
    run = database.get(ModelRun, experiment.selected_model_run_id)
    if run is None:
        raise TrainingNotReadyError("Selected model is unavailable.")
    artifact = load_model_artifact(run, settings)
    model = artifact.model
    names = [_readable_name(str(name)) for name in artifact.preprocessor.get_feature_names_out()]
    raw_artifact = joblib.load(model_artifact_path(settings, run.artifact_filename))
    try:
        dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
        dataframe = load_dataset_dataframe(dataset, settings)
        features, _, _, _, _, _ = _prepare_features(dataframe, raw_artifact["target_column"])
        sample = features.loc[raw_artifact["train_indices"], artifact.feature_columns].head(settings.shap_max_samples)
        transformed = artifact.preprocessor.transform(sample)
        if hasattr(model, "feature_importances_"):
            values = shap.TreeExplainer(model).shap_values(transformed)
        elif hasattr(model, "coef_"):
            values = shap.LinearExplainer(model, transformed).shap_values(transformed)
        else:
            raise ValueError("unsupported model")
        values = np.asarray(values)
        if values.ndim == 3:
            values = np.mean(np.abs(values), axis=(0, 2))
        else:
            values = np.mean(np.abs(values), axis=0)
        if len(values) != len(names):
            raise ValueError("SHAP feature metadata mismatch")
        importance = sorted(
            [FeatureImportance(feature_name=name, importance=float(value)) for name, value in zip(names, values, strict=True)],
            key=lambda item: (-item.importance, item.feature_name),
        )
        return ExplainabilityResponse(
            experiment_id=experiment.id, selected_model_run_id=run.id, model_name=run.model_name,
            method="shap", sample_size=len(sample), metadata={"bounded": True, "max_samples": settings.shap_max_samples}, global_feature_importance=importance,
        )
    except Exception:
        # Native explanation remains a deliberate safe fallback; explanation errors never affect the experiment.
        pass
    try:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            method = "native_feature_importance"
            directions = [None] * len(values)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_, dtype=float)
            if coefficients.ndim > 1:
                coefficients = np.mean(np.abs(coefficients), axis=0)
                signed = coefficients
                directions = [None] * len(coefficients)
            else:
                signed = coefficients
                directions = ["positive" if value >= 0 else "negative" for value in signed]
            values = np.abs(coefficients)
            method = "native_coefficients"
        else:
            raise TrainingNotReadyError("Native explainability is unsupported for the selected model.")
    except ValueError as error:
        raise TrainingNotReadyError("Native explainability could not read the selected model.") from error
    if len(names) != len(values):
        raise TrainingNotReadyError("Explainability feature metadata is inconsistent.")
    importance = sorted(
        [FeatureImportance(feature_name=name, importance=float(value), direction=direction)
         for name, value, direction in zip(names, values, directions, strict=True)],
        key=lambda item: (-item.importance, item.feature_name),
    )
    return ExplainabilityResponse(
        experiment_id=experiment.id, selected_model_run_id=run.id,
        model_name=run.model_name, method=method, sample_size=0, metadata={"fallback": "shap_unavailable_or_unsupported"},
        global_feature_importance=importance,
    )
