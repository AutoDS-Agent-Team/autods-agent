from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from app.schemas.pipeline_plan import ModelName


class ModelRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"


class SplitSummary(BaseModel):
    training_rows: int
    validation_rows: int
    test_rows: int
    random_state: int
    stratified: bool


class ModelRunResponse(BaseModel):
    model_run_id: str
    model_name: ModelName
    status: ModelRunStatus
    training_duration_seconds: float
    failure_information: str | None
    artifact_filename: str | None
    created_at: datetime


class TrainingResponse(BaseModel):
    experiment_id: str
    plan_id: str
    status: TrainingStatus
    split: SplitSummary
    model_runs: list[ModelRunResponse]
