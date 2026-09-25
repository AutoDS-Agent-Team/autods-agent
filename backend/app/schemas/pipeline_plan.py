from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from app.schemas.experiment import MLTaskType


class NumericImputation(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"


class CategoricalImputation(StrEnum):
    MOST_FREQUENT = "most_frequent"


class CategoricalEncoding(StrEnum):
    ONE_HOT = "one_hot"


class NumericScaling(StrEnum):
    STANDARD = "standard"
    NONE = "none"

class TextVectorization(StrEnum):
    TFIDF = "tfidf"


class ModelName(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RIDGE_REGRESSION = "ridge_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class MetricName(StrEnum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"


CLASSIFICATION_MODELS = {
    ModelName.LOGISTIC_REGRESSION,
    ModelName.RANDOM_FOREST,
    ModelName.XGBOOST,
}
REGRESSION_MODELS = {
    ModelName.RIDGE_REGRESSION,
    ModelName.RANDOM_FOREST,
    ModelName.XGBOOST,
}
CLASSIFICATION_METRICS = {
    MetricName.ACCURACY,
    MetricName.PRECISION,
    MetricName.RECALL,
    MetricName.F1,
    MetricName.ROC_AUC,
}
REGRESSION_METRICS = {MetricName.MAE, MetricName.RMSE, MetricName.R2}


class PipelinePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: MLTaskType
    target_column: str = Field(min_length=1)
    numeric_imputation: NumericImputation
    categorical_imputation: CategoricalImputation
    categorical_encoding: CategoricalEncoding
    numeric_scaling: NumericScaling
    models: list[ModelName] = Field(min_length=1, max_length=3)
    primary_metric: MetricName
    excluded_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    text_vectorization: TextVectorization | None = None
    balanced_class_weight_models: list[ModelName] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self, info: ValidationInfo) -> Self:
        if len(set(self.models)) != len(self.models):
            raise ValueError("Pipeline models must be unique.")

        is_classification = self.task_type in {
            MLTaskType.BINARY_CLASSIFICATION,
            MLTaskType.MULTICLASS_CLASSIFICATION,
        }
        allowed_models = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS
        allowed_metrics = (
            CLASSIFICATION_METRICS if is_classification else REGRESSION_METRICS
        )
        if any(model not in allowed_models for model in self.models):
            raise ValueError("The plan contains a model incompatible with its task type.")
        if self.primary_metric not in allowed_metrics:
            raise ValueError("The primary metric is incompatible with the task type.")
        if (
            self.primary_metric == MetricName.ROC_AUC
            and self.task_type != MLTaskType.BINARY_CLASSIFICATION
        ):
            raise ValueError("roc_auc is supported only for binary classification.")
        if self.balanced_class_weight_models:
            if not is_classification:
                raise ValueError("Balanced class weighting is supported only for classification.")
            supported_weighting = {ModelName.LOGISTIC_REGRESSION, ModelName.RANDOM_FOREST}
            if any(model not in self.models or model not in supported_weighting for model in self.balanced_class_weight_models):
                raise ValueError("Class weighting is not supported for one or more planned models.")

        context = info.context or {}
        confirmed_task = context.get("confirmed_task_type")
        confirmed_target = context.get("confirmed_target_column")
        if confirmed_task is not None and self.task_type.value != str(confirmed_task):
            raise ValueError("Plan task type must match the confirmed experiment task.")
        if confirmed_target is not None and self.target_column != confirmed_target:
            raise ValueError("Plan target must match the confirmed experiment target.")
        return self

    @classmethod
    def validate_provider_output(
        cls,
        payload: str | bytes | dict[str, Any],
        *,
        confirmed_task_type: str,
        confirmed_target_column: str,
    ) -> "PipelinePlan":
        context = {
            "confirmed_task_type": confirmed_task_type,
            "confirmed_target_column": confirmed_target_column,
        }
        if isinstance(payload, dict):
            return cls.model_validate(payload, context=context)
        return cls.model_validate_json(payload, context=context)


class AdaptiveDecision(BaseModel):
    decision: str
    reason: str
    evidence: str
    action_taken: str
    applied: bool = False


class ModelRecommendationReason(BaseModel):
    model_name: ModelName
    reason: str
    dataset_evidence: str


class PlanOverrideRequest(BaseModel):
    balanced_class_weight_models: list[ModelName] = Field(default_factory=list, max_length=3)


class PlanResponse(BaseModel):
    plan_id: str
    experiment_id: str
    plan: PipelinePlan
    provider_used: str
    schema_version: str
    created_at: datetime
    adaptive_decisions: list[AdaptiveDecision] = Field(default_factory=list)
    model_recommendations: list[ModelRecommendationReason] = Field(default_factory=list)


class PlanningColumnContext(BaseModel):
    name: str
    logical_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_constant: bool
    is_possible_id: bool


class PlanningContext(BaseModel):
    objective: str
    confirmed_task_type: MLTaskType
    confirmed_target_column: str
    row_count: int
    column_count: int
    columns: list[PlanningColumnContext]
