from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.pipeline_plan import MetricName, ModelName


class ConfusionMatrixResponse(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ModelComparisonItem(BaseModel):
    model_run_id: str
    model_name: ModelName
    metrics: dict[str, float | None]
    training_seconds: float | None = None
    inference_seconds_per_row: float | None = None
    complexity_summary: str | None = None
    generalization_change: float | None = None
    generalization_status: str | None = None


class EvaluationResponse(BaseModel):
    experiment_id: str
    evaluation_id: str
    primary_metric: MetricName
    validation_comparison: list[ModelComparisonItem]
    selected_model_run_id: str
    selected_model_name: ModelName
    validation_metrics: dict[str, float | None]
    final_test_metrics: dict[str, float | None]
    final_test_confusion_matrix: ConfusionMatrixResponse | None
    status: str
    created_at: datetime


class OptimizationResponse(BaseModel):
    optimization_id: str
    experiment_id: str
    baseline_model_run_id: str
    optimized_model_run_id: str | None
    model_name: ModelName
    status: str
    trial_count: int
    best_parameters: dict[str, Any]
    best_validation_score: float | None
    duration_seconds: float
    cost_profile: dict[str, Any] = Field(default_factory=dict)
    failure_information: str | None


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float
    direction: str | None = None


class ExplainabilityResponse(BaseModel):
    experiment_id: str
    selected_model_run_id: str
    model_name: ModelName
    method: str
    sample_size: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    global_feature_importance: list[FeatureImportance]


class PredictionResponse(BaseModel):
    prediction_run_id: str
    experiment_id: str
    model_run_id: str
    row_count: int
    status: str
    download_url: str
    created_at: datetime


class ReportResponse(BaseModel):
    report_id: str
    experiment_id: str
    status: str
    download_url: str
    created_at: datetime


class ReportHistoryItem(BaseModel):
    report_id: str
    experiment_id: str
    objective: str
    format: str
    download_url: str
    created_at: datetime


class ReportHistoryResponse(BaseModel):
    items: list[ReportHistoryItem] = Field(default_factory=list)
