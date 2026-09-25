import json
from pydantic import ValidationError

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ExperimentNotFoundError, ExperimentValidationError, PlanningFailureError, PlanningNotReadyError
from app.llm.router import LLMRouter, RoutedPlan
from app.models.experiment import Experiment
from app.models.pipeline_plan import PipelinePlanRecord
from app.models.model_run import ModelRun
from app.models.job import Job
from app.schemas.experiment import ExperimentStatus
from app.schemas.pipeline_plan import (
    PlanResponse,
    PlanOverrideRequest,
    PipelinePlan,
    PlanningColumnContext,
    PlanningContext,
    TextVectorization,
)
from app.services.dataset_service import get_dataset_metadata, get_dataset_profile, load_dataset_dataframe
from app.services.adaptive_pipeline_service import adapt_pipeline

PLAN_SCHEMA_VERSION = "1.0"


def _deterministic_fallback_plan(context: PlanningContext) -> RoutedPlan:
    classification = context.confirmed_task_type.value != "regression"
    payload = {
        "task_type": context.confirmed_task_type.value,
        "target_column": context.confirmed_target_column,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot",
        "numeric_scaling": "standard",
        "models": ["logistic_regression", "random_forest", "xgboost"] if classification else ["ridge_regression", "random_forest", "xgboost"],
        "primary_metric": "f1" if classification else "rmse",
    }
    plan = PipelinePlan.validate_provider_output(payload, confirmed_task_type=context.confirmed_task_type.value, confirmed_target_column=context.confirmed_target_column)
    return RoutedPlan(plan=plan, provider_used="trusted_fallback")


def build_planning_prompt(context: PlanningContext) -> str:
    supported_values = {
        "numeric_imputation": ["mean", "median"],
        "categorical_imputation": ["most_frequent"],
        "categorical_encoding": ["one_hot"],
        "numeric_scaling": ["standard", "none"],
        "classification_models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ],
        "regression_models": ["ridge_regression", "random_forest", "xgboost"],
        "classification_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "regression_metrics": ["mae", "rmse", "r2"],
    }
    return (
        "Act only as an ML pipeline planner. Return structured JSON only and never "
        "generate Python, shell commands, SQL, or prose. Use only the allowlisted values. "
        "The confirmed task and target are authoritative and must be copied exactly. "
        "Never invent columns. Avoid data leakage and select sensible baseline models "
        "and one compatible primary metric.\n\n"
        f"ALLOWLISTS:\n{json.dumps(supported_values, sort_keys=True)}\n\n"
        f"SAFE_DATASET_CONTEXT:\n{context.model_dump_json()}"
    )


def _get_experiment(experiment_id: str, database: Session) -> Experiment:
    experiment = database.get(Experiment, experiment_id)
    if experiment is None:
        raise ExperimentNotFoundError("Experiment not found.")
    if (
        not experiment.user_objective
        or not experiment.confirmed_task_type
        or not experiment.confirmed_target_column
        or experiment.status == ExperimentStatus.AWAITING_CONFIRMATION.value
    ):
        raise PlanningNotReadyError(
            "Confirm the experiment task and target before generating a plan."
        )
    return experiment


def _planning_context(
    experiment: Experiment,
    database: Session,
    settings: Settings,
) -> PlanningContext:
    profile = get_dataset_profile(experiment.dataset_id, database, settings)
    return PlanningContext(
        objective=experiment.user_objective,
        confirmed_task_type=experiment.confirmed_task_type,
        confirmed_target_column=experiment.confirmed_target_column,
        row_count=profile.summary.row_count,
        column_count=profile.summary.column_count,
        columns=[
            PlanningColumnContext(
                name=column.name,
                logical_type=column.logical_type,
                missing_count=column.missing_count,
                missing_percentage=column.missing_percentage,
                unique_count=column.unique_count,
                is_constant=column.is_constant,
                is_possible_id=column.is_possible_id,
            )
            for column in profile.columns
        ],
    )


def create_pipeline_plan(
    experiment_id: str,
    database: Session,
    settings: Settings,
    llm_router: LLMRouter,
) -> PlanResponse:
    experiment = _get_experiment(experiment_id, database)
    existing = database.scalar(
        select(PipelinePlanRecord).where(
            PipelinePlanRecord.experiment_id == experiment.id
        )
    )
    dataframe = load_dataset_dataframe(get_dataset_metadata(experiment.dataset_id, database, settings), settings)
    if existing is not None:
        existing_plan = PipelinePlan.validate_provider_output(existing.plan, confirmed_task_type=experiment.confirmed_task_type, confirmed_target_column=experiment.confirmed_target_column)
        has_runs = database.scalar(select(ModelRun.id).where(ModelRun.experiment_id == experiment.id).limit(1)) is not None
        if not has_runs:
            adapted, _, _ = adapt_pipeline(dataframe, existing_plan)
            existing.plan = adapted.model_dump(mode="json")
            database.commit(); database.refresh(existing)
        return _plan_response(existing, experiment, dataframe)

    context = _planning_context(experiment, database, settings)
    try:
        routed = llm_router.generate_plan(
            build_planning_prompt(context),
            confirmed_task_type=experiment.confirmed_task_type,
            confirmed_target_column=experiment.confirmed_target_column,
        )
    except PlanningFailureError:
        routed = _deterministic_fallback_plan(context)
    # Feature roles are deterministic profiler evidence, never provider authority.
    routed = RoutedPlan(plan=routed.plan.model_copy(update={
        "excluded_columns": [column.name for column in context.columns if column.is_possible_id and column.name != experiment.confirmed_target_column],
        "text_columns": [column.name for column in context.columns if column.logical_type == "text" and column.name != experiment.confirmed_target_column],
        "text_vectorization": TextVectorization.TFIDF if any(column.logical_type == "text" and column.name != experiment.confirmed_target_column for column in context.columns) else None,
    }), provider_used=routed.provider_used)
    adaptive_plan, _, _ = adapt_pipeline(dataframe, routed.plan)
    record = PipelinePlanRecord(
        experiment_id=experiment.id,
        plan=adaptive_plan.model_dump(mode="json"),
        provider_used=routed.provider_used,
        schema_version=PLAN_SCHEMA_VERSION,
    )
    database.add(record)
    experiment.status = ExperimentStatus.PLAN_READY.value
    database.commit()
    database.refresh(record)
    return _plan_response(record, experiment, dataframe)


def update_pipeline_overrides(experiment_id: str, request: PlanOverrideRequest, database: Session, settings: Settings) -> PlanResponse:
    experiment = _get_experiment(experiment_id, database)
    record = database.scalar(select(PipelinePlanRecord).where(PipelinePlanRecord.experiment_id == experiment.id))
    if record is None:
        raise PlanningNotReadyError("Generate a pipeline plan before changing adaptive settings.")
    trained = database.scalar(select(ModelRun.id).where(ModelRun.experiment_id == experiment.id).limit(1)) is not None
    training_job = database.scalar(select(Job.id).where(Job.experiment_id == experiment.id, Job.job_type == "train", Job.status.in_({"PENDING", "RUNNING"})).limit(1)) is not None
    if trained or training_job:
        raise PlanningNotReadyError("Adaptive overrides are locked after model training has started.")
    plan = PipelinePlan.validate_provider_output(record.plan, confirmed_task_type=experiment.confirmed_task_type or "", confirmed_target_column=experiment.confirmed_target_column or "")
    updated = plan.model_copy(update={"balanced_class_weight_models": request.balanced_class_weight_models})
    # Re-validate the override against the same task/model allowlists.
    try:
        updated = PipelinePlan.validate_provider_output(updated.model_dump(mode="json"), confirmed_task_type=experiment.confirmed_task_type or "", confirmed_target_column=experiment.confirmed_target_column or "")
    except ValidationError as error:
        raise ExperimentValidationError("Balanced weighting is allowed only for planned logistic regression or random forest classification models.") from error
    record.plan = updated.model_dump(mode="json")
    database.commit(); database.refresh(record)
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    return _plan_response(record, experiment, load_dataset_dataframe(dataset, settings))


def _plan_response(record: PipelinePlanRecord, experiment: Experiment, dataframe=None) -> PlanResponse:
    plan = PipelinePlan.validate_provider_output(
        record.plan,
        confirmed_task_type=experiment.confirmed_task_type or "",
        confirmed_target_column=experiment.confirmed_target_column or "",
    )
    adaptive_decisions, model_recommendations = ([], [])
    if dataframe is not None:
        _, adaptive_decisions, model_recommendations = adapt_pipeline(dataframe, plan, apply_recommendations=False)
    return PlanResponse(
        plan_id=record.id,
        experiment_id=record.experiment_id,
        plan=plan,
        provider_used=record.provider_used,
        schema_version=record.schema_version,
        created_at=record.created_at,
        adaptive_decisions=adaptive_decisions,
        model_recommendations=model_recommendations,
    )
