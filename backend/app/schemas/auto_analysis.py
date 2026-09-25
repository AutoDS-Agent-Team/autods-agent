from enum import StrEnum

from pydantic import BaseModel, Field


class FeatureRole(StrEnum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"


class AnalysisTask(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class FeatureRoleResult(BaseModel):
    column: str
    role: FeatureRole
    excluded_by_default: bool = False
    reason: str


class TargetAlternative(BaseModel):
    column: str
    score: float = Field(ge=0, le=1)
    suggested_task: AnalysisTask | None = None


class TargetRecommendation(BaseModel):
    recommended_target: str | None
    recommended_task: AnalysisTask | None
    confidence: ConfidenceLevel
    reason: str
    alternatives: list[TargetAlternative] = Field(default_factory=list, max_length=5)


class AlgorithmRecommendation(BaseModel):
    algorithm_id: str
    display_name: str
    reason: str


class DatasetAutoAnalysisResponse(BaseModel):
    dataset_id: str
    feature_roles: list[FeatureRoleResult]
    target_recommendation: TargetRecommendation
    applicable_tasks: list[AnalysisTask]
    recommended_algorithms: list[AlgorithmRecommendation]
    warnings: list[str] = Field(default_factory=list)

