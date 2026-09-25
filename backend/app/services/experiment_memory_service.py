"""Bounded, verified experiment retrieval. PostgreSQL remains the source of truth."""
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.dataset import Dataset
from app.models.evaluation_result import EvaluationResult
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.optimization_result import OptimizationResult
from app.models.pipeline_plan import PipelinePlanRecord
from app.models.prediction_run import PredictionRun
from app.models.report import Report


_OBJECTIVE_STOPWORDS = {"a", "an", "and", "for", "from", "how", "in", "is", "of", "on", "or", "predict", "the", "to", "using", "with", "my", "model", "data", "dataset"}


def _objective_terms(value: str | None) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(word) > 2 and word not in _OBJECTIVE_STOPWORDS}


def retrieve_verified_memory(database: Session, user_id: str, settings: Settings, experiment_id: str | None = None) -> list[dict[str, Any]]:
    statement = select(Experiment).where(Experiment.user_id == user_id)
    if experiment_id:
        statement = statement.where(Experiment.id == experiment_id)
    experiments = database.scalars(statement.order_by(Experiment.updated_at.desc()).limit(settings.assistant_max_experiments)).all()
    return [_memory_for_experiment(database, experiment) for experiment in experiments]


def retrieve_similar_experiments(database: Session, user_id: str, current: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """Find bounded, same-user historical results by dataset/task/target/objective."""
    statement = (
        select(Experiment)
        .join(EvaluationResult, EvaluationResult.experiment_id == Experiment.id)
        .where(Experiment.user_id == user_id, Experiment.id != current.get("experiment_id"))
        .where(Experiment.confirmed_task_type == current.get("task_type"))
        .where(Experiment.confirmed_target_column == current.get("target_column"))
    )
    candidates = database.scalars(statement.order_by(Experiment.updated_at.desc()).limit(30)).all()
    current_tokens = _objective_terms(current.get("objective"))
    ranked = []
    for experiment in candidates:
        memory = _memory_for_experiment(database, experiment)
        evaluation = memory.get("evaluation") or {}
        if not evaluation.get("final_test_metrics"):
            continue
        prior_tokens = _objective_terms(memory.get("objective"))
        overlap = len(current_tokens & prior_tokens) / max(1, len(current_tokens | prior_tokens))
        same_dataset = bool(current.get("dataset", {}).get("id")) and current["dataset"]["id"] == memory.get("dataset", {}).get("id")
        if not same_dataset and overlap < 0.2:
            continue
        metric = evaluation.get("primary_metric")
        validation = memory.get("selected_model_validation_metrics") or {}
        final = evaluation.get("final_test_metrics") or {}
        ranked.append((int(same_dataset) * 2 + overlap, {
            "experiment_id": memory["experiment_id"], "dataset_filename": memory.get("dataset", {}).get("filename"),
            "objective": memory.get("objective"), "task_type": memory.get("task_type"), "target_column": memory.get("target_column"), "selected_model": memory.get("selected_model"),
            "primary_metric": metric, "validation_score": validation.get(metric), "test_score": final.get(metric),
            "match_reason": "Same saved dataset, task and target" if same_dataset else f"Same task/target with {overlap:.0%} objective-term overlap",
        }))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in ranked[:max(1, min(limit, 5))]]


def _memory_for_experiment(database: Session, experiment: Experiment) -> dict[str, Any]:
    dataset = database.get(Dataset, experiment.dataset_id)
    plan = database.scalar(select(PipelinePlanRecord).where(PipelinePlanRecord.experiment_id == experiment.id))
    evaluation = database.scalar(select(EvaluationResult).where(EvaluationResult.experiment_id == experiment.id))
    optimization = database.scalar(select(OptimizationResult).where(OptimizationResult.experiment_id == experiment.id))
    runs = database.scalars(select(ModelRun).where(ModelRun.experiment_id == experiment.id)).all()
    selected = next((run for run in runs if evaluation and run.id == evaluation.selected_model_run_id), None)
    selected_entry = next(
        (entry for entry in (evaluation.validation_comparison if evaluation else []) if selected and entry.get("model_run_id") == selected.id),
        None,
    )
    return {
        "experiment_id": experiment.id,
        "dataset": {"id": dataset.id if dataset else None, "filename": dataset.original_filename if dataset else None, "rows": dataset.row_count if dataset else None, "columns": dataset.column_count if dataset else None},
        "objective": experiment.user_objective,
        "status": experiment.status,
        "task_type": experiment.confirmed_task_type,
        "target_column": experiment.confirmed_target_column,
        "pipeline_plan": {**plan.plan, "provider_used": plan.provider_used, "schema_version": plan.schema_version} if plan else None,
        "trained_models": [run.model_name for run in runs],
        "selected_model": selected.model_name if selected else None,
        "selected_model_validation_metrics": selected_entry.get("metrics") if selected_entry else None,
        "evaluation": {"primary_metric": evaluation.primary_metric, "validation_comparison": evaluation.validation_comparison, "final_test_metrics": evaluation.final_test_metrics, "confusion_matrix": evaluation.final_test_confusion_matrix} if evaluation else None,
        "optimization": {"status": optimization.status, "model_name": optimization.model_name, "trial_count": optimization.trial_count, "best_parameters": optimization.best_parameters, "best_validation_score": optimization.best_validation_score} if optimization else None,
        "prediction_count": database.query(PredictionRun).filter(PredictionRun.experiment_id == experiment.id).count(),
        "report_count": database.query(Report).filter(Report.experiment_id == experiment.id).count(),
        "updated_at": experiment.updated_at.isoformat(),
    }


def bounded_context(memories: list[dict[str, Any]], max_chars: int) -> str:
    """Human-readable verified evidence without identifiers or implementation labels."""
    def label(value: str | None) -> str:
        return (value or "Not recorded").replace("_", " ").title()

    lines = ["VERIFIED EXPERIMENT EVIDENCE:"]
    for index, memory in enumerate(memories, start=1):
        evaluation = memory.get("evaluation") or {}
        lines += [f"Experiment {index}", f"Dataset: {memory.get('dataset', {}).get('filename') or 'Not recorded'}", f"Task: {label(memory.get('task_type'))}", f"Target: {memory.get('target_column') or 'Not recorded'}", f"Selected model: {label(memory.get('selected_model'))}", f"Primary selection metric: {label(evaluation.get('primary_metric') or (memory.get('pipeline_plan') or {}).get('primary_metric'))}"]
        for row in evaluation.get("validation_comparison") or []:
            lines.append(f"Validation baseline {label(row.get('model_name'))}: {row.get('metrics') or {}}")
        lines.append(f"Selected model validation metrics: {memory.get('selected_model_validation_metrics') or {}}")
        lines.append(f"Final test metrics: {evaluation.get('final_test_metrics') or {}}")
        if memory.get("optimization"):
            lines.append(f"Optimization: {memory['optimization']}")
        lines.append("")
    context = "\n".join(lines)
    return context[:max_chars]
