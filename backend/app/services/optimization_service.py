from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

import joblib
import optuna
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import TrainingNotReadyError
from app.ml.registry import tunable_model_factory
from app.models.model_run import ModelRun
from app.models.optimization_result import OptimizationResult
from app.schemas.evaluation import OptimizationResponse
from app.schemas.experiment import ExperimentStatus, MLTaskType
from app.schemas.training import ModelRunStatus
from app.services.evaluation_service import (
    HIGHER_IS_BETTER,
    calculate_metrics,
    evaluate_experiment,
    get_evaluation,
    load_model_artifact,
    model_artifact_path,
)
from app.services.training_service import _load_training_definition, _prepare_features
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe


def _parameters_for_trial(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
    if model_name == "ridge_regression":
        return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise TrainingNotReadyError("No trusted optimization search space exists for this model.")


def _response(record: OptimizationResult) -> OptimizationResponse:
    return OptimizationResponse(
        optimization_id=record.id,
        experiment_id=record.experiment_id,
        baseline_model_run_id=record.baseline_model_run_id,
        optimized_model_run_id=record.optimized_model_run_id,
        model_name=record.model_name,
        status=record.status,
        trial_count=record.trial_count,
        best_parameters=record.best_parameters,
        best_validation_score=record.best_validation_score,
        duration_seconds=record.duration_seconds,
        cost_profile=record.cost_profile or {},
        failure_information=record.failure_information,
    )


def _cost_profile(study: optuna.Study, metric: str, trial_budget: int, timeout_seconds: int = 120) -> dict[str, Any]:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    profile: dict[str, Any] = {"cost_basis": "wall_clock_seconds", "trial_budget": trial_budget, "timeout_seconds": timeout_seconds, "budget_seconds": timeout_seconds, "completed_trial_count": len(completed)}
    if not completed:
        return profile
    higher = metric in HIGHER_IS_BETTER
    best = float(study.best_value)
    def regret(trial: optuna.trial.FrozenTrial) -> float:
        gap = best - float(trial.value) if higher else float(trial.value) - best
        return max(0.0, gap / max(abs(best), 1e-12))
    profile.update({
        "mean_trial_seconds": round(sum(float(t.user_attrs.get("duration_seconds", 0)) for t in completed) / len(completed), 4),
        "quality_best_trial_seconds": round(float(study.best_trial.user_attrs.get("duration_seconds", 0)), 4),
        "near_optimal_threshold": "1% relative validation-score difference",
    })
    near_best = [t for t in completed if regret(t) <= 0.01 and t.user_attrs.get("duration_seconds") is not None]
    if near_best:
        fastest = min(near_best, key=lambda t: float(t.user_attrs["duration_seconds"]))
        profile["fastest_near_optimal"] = {"duration_seconds": round(float(fastest.user_attrs["duration_seconds"]), 4), "validation_score": float(fastest.value), "relative_quality_regret": round(regret(fastest), 6)}
    return profile


def optimize_experiment(experiment_id: str, database: Session, settings: Settings) -> OptimizationResponse:
    existing = database.scalar(select(OptimizationResult).where(OptimizationResult.experiment_id == experiment_id))
    if existing:
        return _response(existing)
    experiment, plan_record, plan = _load_training_definition(experiment_id, database)
    evaluation = get_evaluation(experiment.id, database)
    baseline = database.get(ModelRun, evaluation.selected_model_run_id)
    if baseline is None or baseline.status != ModelRunStatus.COMPLETED.value:
        raise TrainingNotReadyError("The selected baseline model is unavailable for optimization.")
    artifact = load_model_artifact(baseline, settings)
    raw_artifact = joblib.load(model_artifact_path(settings, baseline.artifact_filename))
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    features, target, _, _, _, _ = _prepare_features(dataframe, plan.target_column)
    x_train = features.loc[raw_artifact["train_indices"], artifact.feature_columns]
    y_train = target.loc[raw_artifact["train_indices"]]
    x_validation = features.loc[raw_artifact["validation_indices"], artifact.feature_columns]
    y_validation = target.loc[raw_artifact["validation_indices"]]
    transformed_train = artifact.preprocessor.transform(x_train)
    transformed_validation = artifact.preprocessor.transform(x_validation)
    classification = plan.task_type != MLTaskType.REGRESSION
    balanced_class_weight = baseline.model_name in plan.balanced_class_weight_models
    encoded_train = artifact.target_encoder.transform(y_train) if classification else y_train
    direction = "maximize" if plan.primary_metric in HIGHER_IS_BETTER else "minimize"
    if len(dataframe) < 500:
        trial_budget = min(settings.optuna_n_trials, 3)
    elif len(dataframe) > 50_000:
        trial_budget = min(settings.optuna_n_trials, 5)
    else:
        trial_budget = settings.optuna_n_trials

    def objective(trial: optuna.Trial) -> float:
        trial_started = perf_counter()
        params = _parameters_for_trial(trial, baseline.model_name)
        model = tunable_model_factory(
            baseline.model_name, classification=classification,
            random_state=settings.training_random_state, parameters=params,
            balanced_class_weight=balanced_class_weight,
        )
        model.fit(transformed_train, encoded_train)
        metrics, _ = calculate_metrics(
            model, transformed_validation, y_validation, task_type=plan.task_type,
            target_encoder=artifact.target_encoder,
        )
        score = metrics.get(plan.primary_metric.value)
        if score is None:
            raise optuna.TrialPruned("Primary metric is not available for this candidate.")
        trial.set_user_attr("duration_seconds", round(perf_counter() - trial_started, 6))
        return float(score)

    started = perf_counter()
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=settings.training_random_state))
    try:
        study.optimize(objective, n_trials=trial_budget, timeout=settings.optuna_timeout_seconds)
        if not study.best_trials:
            raise TrainingNotReadyError("Optimization did not produce a valid validation score.")
        cost_profile = _cost_profile(study, plan.primary_metric.value, trial_budget, settings.optuna_timeout_seconds)
        best_parameters = dict(study.best_params)
        best_score = float(study.best_value)
        model = tunable_model_factory(
            baseline.model_name, classification=classification,
            random_state=settings.training_random_state, parameters=best_parameters,
            balanced_class_weight=balanced_class_weight,
        )
        model.fit(transformed_train, encoded_train)
        run_id = str(uuid4())
        filename = f"{run_id}.joblib"
        joblib.dump({
            **raw_artifact,
            "model": model,
            "optimization": {"best_parameters": best_parameters, "trial_count": len(study.trials)},
        }, model_artifact_path(settings, filename))
        run = ModelRun(
            id=run_id, experiment_id=experiment.id, pipeline_plan_id=plan_record.id,
            model_name=baseline.model_name, status=ModelRunStatus.COMPLETED.value,
            parameters={**baseline.parameters, "run_kind": "optimized", "optimization": best_parameters},
            training_duration_seconds=round(perf_counter() - started, 6),
            failure_information=None, artifact_filename=filename,
        )
        database.add(run)
        result = OptimizationResult(
            experiment_id=experiment.id, baseline_model_run_id=baseline.id,
            optimized_model_run_id=run.id, model_name=baseline.model_name, status="COMPLETED",
            trial_count=len(study.trials), best_parameters=best_parameters,
            best_validation_score=best_score, duration_seconds=round(perf_counter() - started, 6),
            cost_profile=cost_profile,
            failure_information=None,
        )
        database.add(result)
        experiment.status = ExperimentStatus.OPTIMIZED.value
        database.commit()
        # Baseline selection and its untouched test evaluation are immutable.
        # The optimized artifact is represented separately by OptimizationResult.
        database.refresh(result)
        return _response(result)
    except Exception as error:
        result = OptimizationResult(
            experiment_id=experiment.id, baseline_model_run_id=baseline.id,
            optimized_model_run_id=None, model_name=baseline.model_name, status="FAILED",
            trial_count=len(study.trials), best_parameters={}, best_validation_score=None,
            duration_seconds=round(perf_counter() - started, 6),
            cost_profile=_cost_profile(study, plan.primary_metric.value, trial_budget, settings.optuna_timeout_seconds),
            failure_information=f"Optimization failed: {type(error).__name__}.",
        )
        database.add(result)
        database.commit()
        return _response(result)
