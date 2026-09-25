from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models.job import Job
from app.models.experiment import Experiment
from app.core.exceptions import ExperimentNotFoundError, JobNotFoundError, TrainingNotReadyError
from app.schemas.job import JobResponse, ExperimentHistoryItem, ExperimentHistoryResponse
from app.schemas.job import ExperimentDetailResponse
from app.models.pipeline_plan import PipelinePlanRecord
from app.models.model_run import ModelRun
from app.models.evaluation_result import EvaluationResult
from app.models.optimization_result import OptimizationResult
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.tasks import execute_job

def as_response(job: Job) -> JobResponse:
    return JobResponse(job_id=job.id, experiment_id=job.experiment_id, job_type=job.job_type, status=job.status, stage=job.stage, error_information=job.error_information, result_reference=job.result_reference, created_at=job.created_at, started_at=job.started_at, completed_at=job.completed_at)

def create_job(experiment_id: str, job_type: str, database: Session) -> JobResponse:
    if database.get(Experiment, experiment_id) is None: raise ExperimentNotFoundError("Experiment not found.")
    if job_type not in {"train", "evaluate", "optimize", "explain", "report"}: raise TrainingNotReadyError("Unsupported background job.")
    job = Job(experiment_id=experiment_id, job_type=job_type, status="PENDING", stage="PENDING")
    database.add(job); database.commit(); database.refresh(job)
    execute_job.delay(job.id)
    return as_response(job)

def get_job(job_id: str, database: Session) -> JobResponse:
    job = database.get(Job, job_id)
    if job is None:
        raise JobNotFoundError("Job not found.")
    return as_response(job)

def history(page: int, page_size: int, database: Session, user_id: str) -> ExperimentHistoryResponse:
    scope = select(Experiment).where(Experiment.user_id == user_id)
    total = database.scalar(select(func.count()).select_from(scope.subquery())) or 0
    rows = database.scalars(scope.order_by(Experiment.updated_at.desc()).offset((page-1)*page_size).limit(page_size)).all()
    plans = {
        plan.experiment_id: plan.plan.get("primary_metric")
        for plan in database.scalars(
            select(PipelinePlanRecord).where(PipelinePlanRecord.experiment_id.in_([row.id for row in rows]))
        ).all()
    } if rows else {}
    return ExperimentHistoryResponse(
        items=[
            ExperimentHistoryItem(
                experiment_id=x.id, dataset_id=x.dataset_id, objective=x.user_objective,
                task_type=x.confirmed_task_type, target_column=x.confirmed_target_column,
                status=x.status, selected_model_run_id=x.selected_model_run_id,
                primary_metric=plans.get(x.id), created_at=x.created_at, updated_at=x.updated_at,
            )
            for x in rows
        ],
        page=page, page_size=page_size, total=total,
    )

def experiment_detail(experiment_id: str, database: Session) -> ExperimentDetailResponse:
    experiment = database.get(Experiment, experiment_id)
    if experiment is None: raise ExperimentNotFoundError("Experiment not found.")
    plan = database.scalar(select(PipelinePlanRecord).where(PipelinePlanRecord.experiment_id == experiment_id))
    evaluation = database.scalar(select(EvaluationResult).where(EvaluationResult.experiment_id == experiment_id))
    optimization = database.scalar(select(OptimizationResult).where(OptimizationResult.experiment_id == experiment_id))
    runs = database.scalars(select(ModelRun).where(ModelRun.experiment_id == experiment_id)).all()
    predictions = database.scalars(select(PredictionRun).where(PredictionRun.experiment_id == experiment_id)).all()
    reports = database.scalars(select(Report).where(Report.experiment_id == experiment_id)).all()
    jobs = database.scalars(select(Job).where(Job.experiment_id == experiment_id).order_by(Job.created_at.desc()).limit(20)).all()
    return ExperimentDetailResponse(
        experiment_id=experiment.id, dataset_id=experiment.dataset_id, objective=experiment.user_objective,
        task_type=experiment.confirmed_task_type, target_column=experiment.confirmed_target_column,
        status=experiment.status, selected_model_run_id=experiment.selected_model_run_id,
        pipeline_plan=plan.plan if plan else None,
        model_runs=[{"id": x.id, "model_name": x.model_name, "status": x.status, "duration": x.training_duration_seconds} for x in runs],
        evaluation={
            "primary_metric": evaluation.primary_metric,
            "selected_model_run_id": evaluation.selected_model_run_id,
            "validation_comparison": evaluation.validation_comparison,
            "final_test_metrics": evaluation.final_test_metrics,
            "final_test_confusion_matrix": evaluation.final_test_confusion_matrix,
            "created_at": evaluation.created_at,
        } if evaluation else None,
        optimization={
            "status": optimization.status, "model_name": optimization.model_name,
            "trial_count": optimization.trial_count, "best_parameters": optimization.best_parameters,
            "best_validation_score": optimization.best_validation_score,
            "failure_information": optimization.failure_information,
        } if optimization else None,
        prediction_runs=[{"id": x.id, "row_count": x.row_count, "created_at": x.created_at} for x in predictions],
        reports=[{"id": x.id, "created_at": x.created_at} for x in reports], jobs=[as_response(x) for x in jobs],
    )
