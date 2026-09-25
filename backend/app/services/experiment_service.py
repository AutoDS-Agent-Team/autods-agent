import re

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import DatasetNotFoundError, ExperimentNotFoundError, ExperimentValidationError
from app.models.experiment import Experiment
from app.models.user import User
from app.schemas.experiment import (
    ExperimentAnalysisResponse,
    ExperimentConfirmRequest,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentStatus,
    InferenceConfidence,
    MLTaskType,
    TargetCandidate,
)
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.profiling_service import profile_dataframe

MAX_CLASSIFICATION_CLASSES = 100
AMBIGUOUS_INTEGER_CLASS_LIMIT = 20
SUPPORTED_TASK_TYPES = list(MLTaskType)


def _normalize_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _task_inference(
    series: pd.Series,
    logical_type: str,
) -> tuple[list[MLTaskType], MLTaskType | None, InferenceConfidence, str | None]:
    values = series.dropna()
    unique_count = int(values.nunique())

    if logical_type != "numerical":
        if unique_count == 2:
            return (
                [MLTaskType.BINARY_CLASSIFICATION],
                MLTaskType.BINARY_CLASSIFICATION,
                InferenceConfidence.HIGH,
                None,
            )
        return (
            [MLTaskType.MULTICLASS_CLASSIFICATION],
            MLTaskType.MULTICLASS_CLASSIFICATION,
            InferenceConfidence.HIGH,
            None,
        )

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    integer_like = bool(len(numeric)) and bool((numeric % 1 == 0).all())
    if integer_like and 2 <= unique_count <= AMBIGUOUS_INTEGER_CLASS_LIMIT:
        classification_task = (
            MLTaskType.BINARY_CLASSIFICATION
            if unique_count == 2
            else MLTaskType.MULTICLASS_CLASSIFICATION
        )
        return (
            [classification_task, MLTaskType.REGRESSION],
            None,
            InferenceConfidence.LOW,
            "Low-cardinality integer values could represent class labels or a numeric outcome.",
        )

    return (
        [MLTaskType.REGRESSION],
        MLTaskType.REGRESSION,
        InferenceConfidence.HIGH,
        None,
    )


def _target_candidates(dataframe: pd.DataFrame) -> list[TargetCandidate]:
    _, columns, _ = profile_dataframe(dataframe)
    candidates: list[TargetCandidate] = []
    for column in columns:
        series = dataframe[column.name]
        non_null_count = int(series.notna().sum())
        if column.is_constant or non_null_count < 2 or column.is_possible_id:
            continue
        if (
            column.logical_type != "numerical"
            and column.unique_count > MAX_CLASSIFICATION_CLASSES
        ):
            continue

        compatible, suggested, confidence, ambiguity_reason = _task_inference(
            series, column.logical_type
        )
        candidates.append(
            TargetCandidate(
                name=column.name,
                logical_type=column.logical_type,
                unique_count=column.unique_count,
                non_null_count=non_null_count,
                is_possible_id=column.is_possible_id,
                compatible_task_types=compatible,
                suggested_task_type=suggested,
                confidence=confidence,
                is_ambiguous=ambiguity_reason is not None,
                ambiguity_reason=ambiguity_reason,
            )
        )
    return candidates


def _suggest_target(
    objective: str, candidates: list[TargetCandidate]
) -> TargetCandidate | None:
    normalized_objective = f" {_normalize_words(objective)} "
    matches = [
        candidate
        for candidate in candidates
        if f" {_normalize_words(candidate.name)} " in normalized_objective
    ]
    return matches[0] if len(matches) == 1 else None


def create_experiment(
    request: ExperimentCreateRequest,
    database: Session,
    settings: Settings,
    user: User,
) -> ExperimentAnalysisResponse:
    dataset = get_dataset_metadata(request.dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    dataframe = load_dataset_dataframe(dataset, settings)
    candidates = _target_candidates(dataframe)
    suggested_target = _suggest_target(request.objective, candidates)
    suggested_task = suggested_target.suggested_task_type if suggested_target else None

    if suggested_target is None:
        confidence = InferenceConfidence.NONE
        ambiguity_reason = (
            "The objective did not identify exactly one eligible dataset column; "
            "select a target manually."
        )
    else:
        confidence = suggested_target.confidence
        ambiguity_reason = suggested_target.ambiguity_reason

    experiment = Experiment(
        user_id=user.id,
        dataset_id=dataset.id,
        user_objective=request.objective,
        detected_task_type=suggested_task.value if suggested_task else None,
        suggested_target_column=suggested_target.name if suggested_target else None,
        status=ExperimentStatus.AWAITING_CONFIRMATION.value,
    )
    database.add(experiment)
    database.commit()
    database.refresh(experiment)

    return ExperimentAnalysisResponse(
        experiment_id=experiment.id,
        dataset_id=experiment.dataset_id,
        objective=experiment.user_objective,
        supported_task_types=SUPPORTED_TASK_TYPES,
        target_candidates=candidates,
        suggested_target_column=experiment.suggested_target_column,
        suggested_task_type=suggested_task,
        confidence=confidence,
        is_ambiguous=ambiguity_reason is not None,
        ambiguity_reason=ambiguity_reason,
        status=ExperimentStatus(experiment.status),
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
    )


def _get_experiment_or_raise(experiment_id: str, database: Session) -> Experiment:
    experiment = database.get(Experiment, experiment_id)
    if experiment is None:
        raise ExperimentNotFoundError("Experiment not found.")
    return experiment


def _validate_confirmation(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: MLTaskType,
) -> None:
    if target_column not in dataframe.columns:
        raise ExperimentValidationError(
            "The selected target column does not exist in the dataset."
        )

    target = dataframe[target_column].dropna()
    unique_count = int(target.nunique())
    if len(target) < 2 or unique_count < 2:
        raise ExperimentValidationError(
            "The selected target column must contain at least two usable values and classes."
        )

    if task_type == MLTaskType.BINARY_CLASSIFICATION and unique_count != 2:
        raise ExperimentValidationError(
            "Binary classification requires exactly two target classes."
        )
    if task_type == MLTaskType.MULTICLASS_CLASSIFICATION and not (
        3 <= unique_count <= MAX_CLASSIFICATION_CLASSES
    ):
        raise ExperimentValidationError(
            "Multiclass classification requires between 3 and 100 target classes."
        )
    if task_type == MLTaskType.REGRESSION:
        numeric = pd.to_numeric(target, errors="coerce")
        if int(numeric.notna().sum()) != len(target):
            raise ExperimentValidationError(
                "Regression requires a fully numeric target column."
            )


def confirm_experiment(
    experiment_id: str,
    request: ExperimentConfirmRequest,
    database: Session,
    settings: Settings,
) -> ExperimentResponse:
    experiment = _get_experiment_or_raise(experiment_id, database)
    if experiment.status in {
        ExperimentStatus.PLAN_READY.value,
        ExperimentStatus.TRAINED.value,
        ExperimentStatus.TRAINING_PARTIAL_FAILURE.value,
        ExperimentStatus.TRAINING_FAILED.value,
    }:
        raise ExperimentValidationError(
            "The confirmed task and target cannot change after a pipeline plan exists."
        )
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    _validate_confirmation(dataframe, request.target_column, request.task_type)

    experiment.confirmed_target_column = request.target_column
    experiment.confirmed_task_type = request.task_type.value
    experiment.status = ExperimentStatus.READY_FOR_PLANNING.value
    database.commit()
    database.refresh(experiment)

    return ExperimentResponse(
        experiment_id=experiment.id,
        dataset_id=experiment.dataset_id,
        objective=experiment.user_objective,
        detected_task_type=experiment.detected_task_type,
        suggested_target_column=experiment.suggested_target_column,
        confirmed_task_type=experiment.confirmed_task_type,
        confirmed_target_column=experiment.confirmed_target_column,
        status=ExperimentStatus(experiment.status),
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
    )
