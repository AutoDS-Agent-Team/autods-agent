from datetime import datetime, timezone
import logging
from app.worker import celery_app
from app.db.session import SessionLocal
from app.core.config import get_settings
from app.models.job import Job
from app.services.training_service import train_experiment
from app.services.evaluation_service import evaluate_experiment
from app.services.optimization_service import optimize_experiment
from app.services.report_service import create_report
from app.services.explainability_service import get_explainability

SERVICES = {"train": ("TRAINING", train_experiment), "evaluate": ("EVALUATING", evaluate_experiment), "optimize": ("OPTIMIZING", optimize_experiment), "explain": ("EXPLAINING", get_explainability), "report": ("REPORTING", create_report)}
logger = logging.getLogger(__name__)

@celery_app.task(name="autods.execute_job", bind=True)
def execute_job(task, job_id: str) -> None:
    database = SessionLocal()
    try:
        job = database.get(Job, job_id)
        if job is None: return
        stage, service = SERVICES[job.job_type]
        job.status, job.stage, job.started_at = "RUNNING", stage, datetime.now(timezone.utc)
        job.error_information = None
        database.commit()
        try:
            result = service(job.experiment_id, database, get_settings())
            job.status, job.stage, job.completed_at = "COMPLETED", "COMPLETED", datetime.now(timezone.utc)
            job.error_information = None
            job.result_reference = getattr(result, "evaluation_id", None) or getattr(result, "optimization_id", None) or getattr(result, "report_id", None)
        except Exception as error:
            logger.exception("Background %s job failed for experiment %s", job.job_type, job.experiment_id)
            settings = get_settings()
            if not task.request.called_directly and task.request.retries < settings.celery_task_max_retries:
                job.status, job.stage = "PENDING", "PENDING"
                job.error_information = f"Retrying {job.job_type}: {type(error).__name__}."
                database.commit()
                raise task.retry(exc=error, countdown=min(60, 2 ** (task.request.retries + 1)))
            job.status, job.stage, job.completed_at = "FAILED", "FAILED", datetime.now(timezone.utc)
            job.error_information = f"{job.job_type} failed: {type(error).__name__}."
        database.commit()
    finally:
        database.close()
