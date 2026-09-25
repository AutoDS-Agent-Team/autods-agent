from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ExperimentNotFoundError, TrainingNotReadyError
from app.models.evaluation_result import EvaluationResult
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.schemas.evaluation import (
    ConfusionMatrixResponse,
    EvaluationResponse,
    ModelComparisonItem,
)
from app.schemas.experiment import ExperimentStatus, MLTaskType
from app.schemas.pipeline_plan import MetricName, PipelinePlan
from app.schemas.training import ModelRunStatus
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.training_service import _load_training_definition, _prepare_features


HIGHER_IS_BETTER = {
    MetricName.ACCURACY,
    MetricName.PRECISION,
    MetricName.RECALL,
    MetricName.F1,
    MetricName.ROC_AUC,
    MetricName.R2,
}


@dataclass(frozen=True)
class LoadedArtifact:
    model: Any
    preprocessor: Any
    target_encoder: Any
    feature_columns: list[str]


def model_artifact_path(settings: Settings, filename: str) -> Path:
    root = settings.model_storage_path.expanduser().resolve()
    candidate = (root / filename).resolve()
    if candidate.parent != root or candidate.suffix != ".joblib":
        raise TrainingNotReadyError("Invalid trusted model artifact reference.")
    return candidate


def load_model_artifact(run: ModelRun, settings: Settings) -> LoadedArtifact:
    if not run.artifact_filename:
        raise TrainingNotReadyError("The model run has no trusted artifact.")
    path = model_artifact_path(settings, run.artifact_filename)
    if not path.is_file():
        raise TrainingNotReadyError("The trusted model artifact is unavailable.")
    artifact = joblib.load(path)
    required = {"model", "preprocessor", "target_encoder", "feature_columns"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise TrainingNotReadyError("The trusted model artifact is invalid.")
    return LoadedArtifact(
        model=artifact["model"],
        preprocessor=artifact["preprocessor"],
        target_encoder=artifact["target_encoder"],
        feature_columns=list(artifact["feature_columns"]),
    )


def _classification_metrics(
    model: Any, y_true: np.ndarray, x_transformed: Any, *, multiclass: bool
) -> dict[str, float | None]:
    predicted = model.predict(x_transformed)
    average = "weighted" if multiclass else "binary"
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, average=average, zero_division=0)),
        "roc_auc": None,
    }
    if not multiclass and len(np.unique(y_true)) == 2:
        try:
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(x_transformed)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(x_transformed)
            else:
                return metrics
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except (IndexError, ValueError):
            pass
    return metrics


def calculate_metrics(
    model: Any,
    x_transformed: Any,
    y_true: pd.Series,
    *,
    task_type: MLTaskType,
    target_encoder: Any,
) -> tuple[dict[str, float | None], ConfusionMatrixResponse | None]:
    if task_type == MLTaskType.REGRESSION:
        actual = pd.to_numeric(y_true, errors="raise").to_numpy()
        predicted = model.predict(x_transformed)
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(root_mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
        }, None

    if target_encoder is None:
        raise TrainingNotReadyError("Classification artifact has no target encoder.")
    actual = target_encoder.transform(y_true)
    multiclass = task_type == MLTaskType.MULTICLASS_CLASSIFICATION
    metrics = _classification_metrics(model, actual, x_transformed, multiclass=multiclass)
    predicted = model.predict(x_transformed)
    labels = list(range(len(target_encoder.classes_)))
    matrix = confusion_matrix(actual, predicted, labels=labels).astype(int).tolist()
    return metrics, ConfusionMatrixResponse(
        labels=[str(label) for label in target_encoder.classes_], matrix=matrix
    )


def _select_comparison(
    comparison: list[tuple[ModelRun, dict[str, float | None]]], primary: MetricName
) -> tuple[ModelRun, dict[str, float | None]]:
    usable = [(run, metrics) for run, metrics in comparison if metrics.get(primary.value) is not None]
    if not usable:
        raise TrainingNotReadyError("No successful model produced the required validation metric.")
    # Ties use the model name, then the generated run ID, ascending. This is stable and documented.
    multiplier = 1 if primary in HIGHER_IS_BETTER else -1
    return sorted(
        usable,
        key=lambda item: (-multiplier * float(item[1][primary.value]), item[0].model_name, item[0].id),
    )[0]


def _complexity_summary(model: Any) -> str:
    if hasattr(model, "estimators_"):
        estimators = list(model.estimators_)
        nodes = sum(int(tree.tree_.node_count) for tree in estimators if hasattr(tree, "tree_"))
        return f"{len(estimators):,} trees · {nodes:,} total nodes (size proxy)"
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        rounds = int(booster.num_boosted_rounds())
        return f"{rounds:,} boosting rounds · depth {model.get_params().get('max_depth', 'default')}"
    coefficients = getattr(model, "coef_", None)
    if coefficients is not None:
        return f"{int(np.size(coefficients)):,} learned coefficients"
    return "Complexity details unavailable"


def _response(record: EvaluationResult, selected: ModelRun) -> EvaluationResponse:
    comparison = [ModelComparisonItem(**entry) for entry in record.validation_comparison]
    selected_entry = next(entry for entry in comparison if entry.model_run_id == selected.id)
    matrix = (
        ConfusionMatrixResponse(**record.final_test_confusion_matrix)
        if record.final_test_confusion_matrix else None
    )
    return EvaluationResponse(
        experiment_id=record.experiment_id,
        evaluation_id=record.id,
        primary_metric=record.primary_metric,
        validation_comparison=comparison,
        selected_model_run_id=selected.id,
        selected_model_name=selected.model_name,
        validation_metrics=selected_entry.metrics,
        final_test_metrics=record.final_test_metrics,
        final_test_confusion_matrix=matrix,
        status="COMPLETED",
        created_at=record.created_at,
    )


def evaluate_experiment(experiment_id: str, database: Session, settings: Settings) -> EvaluationResponse:
    experiment, _, plan = _load_training_definition(experiment_id, database)
    all_completed_runs = list(database.scalars(select(ModelRun).where(
        ModelRun.experiment_id == experiment.id,
        ModelRun.status == ModelRunStatus.COMPLETED.value,
    ).order_by(ModelRun.model_name, ModelRun.id)))
    # Optimized artifacts are retained for the Optimization section, not silently
    # mixed into the baseline candidate comparison used for model selection.
    runs = [run for run in all_completed_runs if run.parameters.get("run_kind") != "optimized"]
    if not runs:
        raise TrainingNotReadyError("Train at least one successful model before evaluation.")
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    features, target, _, _, _, _ = _prepare_features(dataframe, plan.target_column)
    comparison: list[tuple[ModelRun, dict[str, float | None], float, str]] = []
    loaded: dict[str, LoadedArtifact] = {}
    for run in runs:
        artifact = load_model_artifact(run, settings)
        loaded[run.id] = artifact
        validation_indices = joblib.load(model_artifact_path(settings, run.artifact_filename))["validation_indices"]
        x_validation = features.loc[validation_indices, artifact.feature_columns]
        y_validation = target.loc[validation_indices]
        transformed_validation = artifact.preprocessor.transform(x_validation)
        # ColumnTransformer may return a scipy sparse matrix (e.g. one-hot
        # encoded categorical data); len(sparse_matrix) is intentionally
        # ambiguous, while shape[0] is the verified sample count.
        inference_rows = min(int(transformed_validation.shape[0]), 512)
        inference_start = perf_counter()
        artifact.model.predict(transformed_validation[:inference_rows])
        inference_seconds_per_row = (perf_counter() - inference_start) / max(1, inference_rows)
        metrics, _ = calculate_metrics(
            artifact.model, transformed_validation, y_validation,
            task_type=plan.task_type, target_encoder=artifact.target_encoder,
        )
        comparison.append((run, metrics, inference_seconds_per_row, _complexity_summary(artifact.model)))
    selected, selected_metrics = _select_comparison([(run, metrics) for run, metrics, _, _ in comparison], plan.primary_metric)
    artifact = loaded[selected.id]
    stored = joblib.load(model_artifact_path(settings, selected.artifact_filename))
    test_indices = stored["test_indices"]
    x_test = features.loc[test_indices, artifact.feature_columns]
    y_test = target.loc[test_indices]
    test_metrics, matrix = calculate_metrics(
        artifact.model, artifact.preprocessor.transform(x_test), y_test,
        task_type=plan.task_type, target_encoder=artifact.target_encoder,
    )
    entries = []
    for run, metrics, inference_seconds_per_row, complexity_summary in comparison:
        validation_score = metrics.get(plan.primary_metric.value)
        delta = float(test_metrics[plan.primary_metric.value]) - float(validation_score) if run.id == selected.id and test_metrics.get(plan.primary_metric.value) is not None and validation_score is not None else None
        entries.append({
            "model_run_id": run.id, "model_name": run.model_name, "metrics": metrics,
            "training_seconds": run.training_duration_seconds,
            "inference_seconds_per_row": inference_seconds_per_row,
            "complexity_summary": complexity_summary,
            "generalization_change": delta,
            "generalization_status": "Selected model: test-minus-validation change" if run.id == selected.id else "Not evaluated on test; selection remains validation-only",
        })
    record = database.scalar(select(EvaluationResult).where(EvaluationResult.experiment_id == experiment.id))
    if record is None:
        record = EvaluationResult(
            experiment_id=experiment.id,
            selected_model_run_id=selected.id,
            primary_metric=plan.primary_metric.value,
            validation_comparison=entries,
            final_test_metrics=test_metrics,
            final_test_confusion_matrix=matrix.model_dump() if matrix else None,
        )
        database.add(record)
    else:
        record.selected_model_run_id = selected.id
        record.primary_metric = plan.primary_metric.value
        record.validation_comparison = entries
        record.final_test_metrics = test_metrics
        record.final_test_confusion_matrix = matrix.model_dump() if matrix else None
    experiment.selected_model_run_id = selected.id
    experiment.status = ExperimentStatus.EVALUATED.value
    database.commit()
    database.refresh(record)
    return _response(record, selected)


def get_evaluation(experiment_id: str, database: Session) -> EvaluationResult:
    if database.get(Experiment, experiment_id) is None:
        raise ExperimentNotFoundError("Experiment not found.")
    record = database.scalar(select(EvaluationResult).where(EvaluationResult.experiment_id == experiment_id))
    if record is None:
        raise TrainingNotReadyError("Evaluate the successful model runs first.")
    return record
