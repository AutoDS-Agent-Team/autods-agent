from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class MLTaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class ExperimentStatus(StrEnum):
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    READY_FOR_PLANNING = "READY_FOR_PLANNING"
    PLAN_READY = "PLAN_READY"
    TRAINED = "TRAINED"
    TRAINING_PARTIAL_FAILURE = "TRAINING_PARTIAL_FAILURE"
    TRAINING_FAILED = "TRAINING_FAILED"
    EVALUATED = "EVALUATED"
    OPTIMIZED = "OPTIMIZED"


class InferenceConfidence(StrEnum):
    HIGH = "high"
    LOW = "low"
    NONE = "none"


class ExperimentCreateRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=36)
    objective: str = Field(min_length=1, max_length=4000)

    @field_validator("dataset_id", "objective")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must not be blank.")
        return normalized


class ExperimentConfirmRequest(BaseModel):
    target_column: str = Field(min_length=1, max_length=255)
    task_type: MLTaskType

    @field_validator("target_column")
    @classmethod
    def reject_blank_target(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Target column must not be blank.")
        return normalized


class TargetCandidate(BaseModel):
    name: str
    logical_type: str
    unique_count: int
    non_null_count: int
    is_possible_id: bool
    compatible_task_types: list[MLTaskType]
    suggested_task_type: MLTaskType | None
    confidence: InferenceConfidence
    is_ambiguous: bool
    ambiguity_reason: str | None = None


class ExperimentAnalysisResponse(BaseModel):
    experiment_id: str
    dataset_id: str
    objective: str
    supported_task_types: list[MLTaskType]
    target_candidates: list[TargetCandidate]
    suggested_target_column: str | None
    suggested_task_type: MLTaskType | None
    confidence: InferenceConfidence
    is_ambiguous: bool
    ambiguity_reason: str | None
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime


class ExperimentResponse(BaseModel):
    experiment_id: str
    dataset_id: str
    objective: str
    detected_task_type: MLTaskType | None
    suggested_target_column: str | None
    confirmed_task_type: MLTaskType | None
    confirmed_target_column: str | None
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
