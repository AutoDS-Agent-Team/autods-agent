from typing import Literal

from pydantic import BaseModel, Field


class AnomalyRequest(BaseModel):
    algorithm: Literal["isolation_forest", "local_outlier_factor"] = "isolation_forest"
    contamination: float = Field(default=0.05, gt=0, le=0.2)


class AnomalyExample(BaseModel):
    row_index: int
    score: float | None = None


class AnomalyResponse(BaseModel):
    dataset_id: str
    algorithm: str
    anomaly_count: int
    anomaly_percentage: float
    analyzed_rows: int
    feature_columns: list[str]
    excluded_columns: list[str]
    examples: list[AnomalyExample] = Field(max_length=20)


class ForecastRequest(BaseModel):
    time_column: str = Field(min_length=1, max_length=255)
    target_column: str = Field(min_length=1, max_length=255)
    algorithm: Literal["naive_forecast", "seasonal_naive", "exponential_smoothing", "arima"] = "naive_forecast"
    seasonal_period: int = Field(default=1, ge=1, le=365)
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.4)


class ForecastMetric(BaseModel):
    mae: float
    rmse: float
    mape: float | None


class ForecastResponse(BaseModel):
    dataset_id: str
    algorithm: str
    training_rows: int
    test_rows: int
    chronological_split: bool = True
    metrics: ForecastMetric
    timestamps: list[str] = Field(max_length=1000)
    actual: list[float] = Field(max_length=1000)
    predicted: list[float] = Field(max_length=1000)

