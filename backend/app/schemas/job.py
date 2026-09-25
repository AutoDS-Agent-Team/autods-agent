from datetime import datetime
from pydantic import BaseModel
from typing import Any

class JobResponse(BaseModel):
    job_id: str
    experiment_id: str
    job_type: str
    status: str
    stage: str
    error_information: str | None
    result_reference: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

class ExperimentHistoryItem(BaseModel):
    experiment_id: str
    dataset_id: str
    objective: str
    task_type: str | None
    target_column: str | None
    status: str
    selected_model_run_id: str | None
    primary_metric: str | None = None
    created_at: datetime
    updated_at: datetime

class ExperimentHistoryResponse(BaseModel):
    items: list[ExperimentHistoryItem]
    page: int
    page_size: int
    total: int

class ExperimentDetailResponse(BaseModel):
    experiment_id: str
    dataset_id: str
    objective: str
    task_type: str | None
    target_column: str | None
    status: str
    selected_model_run_id: str | None
    pipeline_plan: dict[str, Any] | None
    model_runs: list[dict[str, Any]]
    evaluation: dict[str, Any] | None
    optimization: dict[str, Any] | None
    prediction_runs: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    jobs: list[JobResponse]
