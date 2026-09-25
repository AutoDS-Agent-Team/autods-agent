from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    ExperimentNotFoundError,
    TrainingFailureError,
    TrainingNotReadyError,
)
from app.ml.registry import (
    CATEGORICAL_IMPUTERS,
    ENCODERS,
    NUMERIC_IMPUTERS,
    SCALERS,
    classification_model_factories,
    regression_model_factories,
    trusted_factory,
)
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.pipeline_plan import PipelinePlanRecord
from app.schemas.experiment import ExperimentStatus, MLTaskType
from app.schemas.pipeline_plan import PipelinePlan
from app.schemas.training import (
    ModelRunResponse,
    ModelRunStatus,
    SplitSummary,
    TrainingResponse,
    TrainingStatus,
)
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.profiling_service import profile_dataframe


@dataclass(frozen=True)
class DatasetSplits:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    stratified: bool


def _can_stratify(target: pd.Series, holdout_size: float) -> bool:
    counts = target.value_counts()
    class_count = len(counts)
    holdout_rows = math.ceil(len(target) * holdout_size)
    remaining_rows = len(target) - holdout_rows
    return (
        class_count >= 2
        and int(counts.min()) >= 2
        and holdout_rows >= class_count
        and remaining_rows >= class_count
    )


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    classification: bool,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> DatasetSplits:
    if validation_size + test_size >= 1:
        raise TrainingFailureError("Validation and test sizes must total less than 1.")
    if len(features) < 5:
        raise TrainingFailureError("At least five usable target rows are required for training.")

    stratify_first = target if classification and _can_stratify(target, test_size) else None
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_first,
    )
    relative_validation_size = validation_size / (1 - test_size)
    stratify_second = (
        y_remaining
        if classification and _can_stratify(y_remaining, relative_validation_size)
        else None
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_remaining,
        y_remaining,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=stratify_second,
    )
    return DatasetSplits(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        stratified=stratify_first is not None and stratify_second is not None,
    )


def build_preprocessor(
    plan: PipelinePlan,
    numeric_columns: list[str],
    categorical_columns: list[str],
    text_columns: list[str] | None = None,
    settings: Settings | None = None,
) -> ColumnTransformer:
    text_columns = text_columns or []
    settings = settings or Settings()
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    trusted_factory(
                        NUMERIC_IMPUTERS,
                        plan.numeric_imputation.value,
                        "numeric imputer",
                    ),
                ),
                (
                    "scaler",
                    trusted_factory(SCALERS, plan.numeric_scaling.value, "scaler"),
                ),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    trusted_factory(
                        CATEGORICAL_IMPUTERS,
                        plan.categorical_imputation.value,
                        "categorical imputer",
                    ),
                ),
                (
                    "encoder",
                    trusted_factory(
                        ENCODERS, plan.categorical_encoding.value, "encoder"
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    for column in text_columns:
        transformers.append((f"text_{column}", TfidfVectorizer(max_features=settings.text_tfidf_max_features, ngram_range=(1, 2), min_df=2), column))
    if not transformers:
        raise TrainingFailureError("The dataset has no usable feature columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _artifact_root(settings: Settings) -> Path:
    root = settings.model_storage_path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_path(settings: Settings, filename: str) -> Path:
    root = _artifact_root(settings)
    candidate = (root / filename).resolve()
    if candidate.parent != root:
        raise TrainingFailureError("Invalid model artifact path.")
    return candidate


def _load_training_definition(
    experiment_id: str,
    database: Session,
) -> tuple[Experiment, PipelinePlanRecord, PipelinePlan]:
    experiment = database.get(Experiment, experiment_id)
    if experiment is None:
        raise ExperimentNotFoundError("Experiment not found.")
    if not experiment.confirmed_task_type or not experiment.confirmed_target_column:
        raise TrainingNotReadyError("Confirm the experiment before training.")
    record = database.scalar(
        select(PipelinePlanRecord).where(
            PipelinePlanRecord.experiment_id == experiment.id
        )
    )
    if record is None:
        raise TrainingNotReadyError("Generate a validated pipeline plan before training.")
    plan = PipelinePlan.validate_provider_output(
        record.plan,
        confirmed_task_type=experiment.confirmed_task_type,
        confirmed_target_column=experiment.confirmed_target_column,
    )
    return experiment, record, plan


def _prepare_features(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str], list[str]]:
    dataframe = dataframe.loc[dataframe[target_column].notna()].copy()
    target = dataframe[target_column]
    features = dataframe.drop(columns=[target_column])
    _, profiles, _ = profile_dataframe(dataframe)
    numeric_columns = [
        profile.name
        for profile in profiles
        if profile.name != target_column and profile.logical_type == "numerical"
    ]
    text_columns = [profile.name for profile in profiles if profile.name != target_column and profile.logical_type == "text"]
    excluded_columns = [profile.name for profile in profiles if profile.name != target_column and profile.is_possible_id]
    categorical_columns = [
        profile.name
        for profile in profiles
        if profile.name != target_column and profile.logical_type not in {"numerical", "text"} and not profile.is_possible_id
    ]
    numeric_columns = [column for column in numeric_columns if column not in excluded_columns]
    # An identifier can also look like free text (for example a unique full-name
    # field).  Keep the feature-role lists aligned with the frame after those
    # identifier columns are removed.
    text_columns = [column for column in text_columns if column not in excluded_columns]
    features = features.drop(columns=excluded_columns, errors="ignore")
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_columns:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    for column in text_columns:
        features[column] = features[column].fillna("").astype(str)
    return features, target, numeric_columns, categorical_columns, text_columns, excluded_columns


def _split_summary(splits: DatasetSplits, settings: Settings) -> SplitSummary:
    return SplitSummary(
        training_rows=len(splits.x_train),
        validation_rows=len(splits.x_validation),
        test_rows=len(splits.x_test),
        random_state=settings.training_random_state,
        stratified=splits.stratified,
    )


def _stored_response(
    experiment: Experiment,
    plan_record: PipelinePlanRecord,
    runs: list[ModelRun],
) -> TrainingResponse:
    completed = sum(run.status == ModelRunStatus.COMPLETED.value for run in runs)
    status = (
        TrainingStatus.COMPLETED
        if completed == len(runs)
        else TrainingStatus.PARTIAL_FAILURE
        if completed
        else TrainingStatus.FAILED
    )
    split_config = runs[0].parameters["split"]
    return TrainingResponse(
        experiment_id=experiment.id,
        plan_id=plan_record.id,
        status=status,
        split=SplitSummary(**split_config),
        model_runs=[
            ModelRunResponse(
                model_run_id=run.id,
                model_name=run.model_name,
                status=run.status,
                training_duration_seconds=run.training_duration_seconds,
                failure_information=run.failure_information,
                artifact_filename=run.artifact_filename,
                created_at=run.created_at,
            )
            for run in runs
        ],
    )


def train_experiment(
    experiment_id: str,
    database: Session,
    settings: Settings,
) -> TrainingResponse:
    experiment, plan_record, plan = _load_training_definition(experiment_id, database)
    existing_runs = list(
        database.scalars(
            select(ModelRun)
            .where(ModelRun.experiment_id == experiment.id)
            .order_by(ModelRun.created_at)
        )
    )
    if existing_runs:
        return _stored_response(experiment, plan_record, existing_runs)

    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    features, target, numeric_columns, categorical_columns, text_columns, excluded_columns = _prepare_features(
        dataframe, plan.target_column
    )
    classification = plan.task_type != MLTaskType.REGRESSION
    if not classification:
        target = pd.to_numeric(target, errors="raise")
    splits = split_dataset(
        features,
        target,
        classification=classification,
        validation_size=settings.training_validation_size,
        test_size=settings.training_test_size,
        random_state=settings.training_random_state,
    )
    text_columns = [column for column in text_columns if column in plan.text_columns] if plan.text_columns else text_columns
    preprocessor = build_preprocessor(plan, numeric_columns, categorical_columns, text_columns, settings)
    try:
        transformed_train = preprocessor.fit_transform(splits.x_train)
        preprocessor.transform(splits.x_validation)
        preprocessor.transform(splits.x_test)
    except Exception as error:
        raise TrainingFailureError(
            "Trusted preprocessing could not prepare this dataset."
        ) from error

    target_encoder: LabelEncoder | None = None
    y_train: Any = splits.y_train
    if classification:
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(splits.y_train)

    if classification:
        weighted_models = {item.value for item in plan.balanced_class_weight_models}
        # Preserve the original one-argument factory path when no adaptive
        # weighting is configured. Besides keeping default training unchanged,
        # this maintains compatibility with integrations that wrap the
        # established factory signature.
        model_factories = (
            classification_model_factories(settings.training_random_state, weighted_models)
            if weighted_models
            else classification_model_factories(settings.training_random_state)
        )
    else:
        model_factories = regression_model_factories(settings.training_random_state)
    split_summary = _split_summary(splits, settings)
    runs: list[ModelRun] = []
    for model_name in plan.models:
        run_id = str(uuid4())
        started = perf_counter()
        status = ModelRunStatus.COMPLETED
        failure_information: str | None = None
        artifact_filename: str | None = None
        try:
            model = trusted_factory(model_factories, model_name.value, "model")
            model.fit(transformed_train, y_train)
            artifact_filename = f"{run_id}.joblib"
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "target_encoder": target_encoder,
                    "feature_columns": list(features.columns),
                    "target_column": plan.target_column,
                    "task_type": plan.task_type.value,
                    "train_indices": splits.x_train.index.tolist(),
                    "validation_indices": splits.x_validation.index.tolist(),
                    "test_indices": splits.x_test.index.tolist(),
                    "random_state": settings.training_random_state,
                },
                _artifact_path(settings, artifact_filename),
            )
        except Exception as error:
            status = ModelRunStatus.FAILED
            failure_information = f"Model training failed: {type(error).__name__}."
            artifact_filename = None

        run = ModelRun(
            id=run_id,
            experiment_id=experiment.id,
            pipeline_plan_id=plan_record.id,
            model_name=model_name.value,
            status=status.value,
            parameters={
                "random_state": settings.training_random_state,
                "plan_schema_version": plan_record.schema_version,
                "split": split_summary.model_dump(mode="json"),
                "numeric_imputation": plan.numeric_imputation.value,
                "categorical_imputation": plan.categorical_imputation.value,
                "categorical_encoding": plan.categorical_encoding.value,
                "numeric_scaling": plan.numeric_scaling.value,
                "text_columns": text_columns,
                "excluded_columns": excluded_columns,
                "class_weight": "balanced" if model_name.value in {item.value for item in plan.balanced_class_weight_models} else None,
            },
            training_duration_seconds=round(perf_counter() - started, 6),
            failure_information=failure_information,
            artifact_filename=artifact_filename,
        )
        database.add(run)
        runs.append(run)

    completed_count = sum(run.status == ModelRunStatus.COMPLETED.value for run in runs)
    experiment.status = (
        ExperimentStatus.TRAINED.value
        if completed_count == len(runs)
        else ExperimentStatus.TRAINING_PARTIAL_FAILURE.value
        if completed_count
        else ExperimentStatus.TRAINING_FAILED.value
    )
    database.commit()
    for run in runs:
        database.refresh(run)
    return _stored_response(experiment, plan_record, runs)
