import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import DatasetValidationError, TrainingFailureError
from app.schemas.specialized_analysis import (
    AnomalyExample,
    AnomalyRequest,
    AnomalyResponse,
    ForecastMetric,
    ForecastRequest,
    ForecastResponse,
)
from app.services.profiling_service import profile_dataframe


def detect_anomalies(dataset_id: str, frame: pd.DataFrame, request: AnomalyRequest, random_state: int) -> AnomalyResponse:
    _, profiles, _ = profile_dataframe(frame)
    excluded = [item.name for item in profiles if item.is_possible_id or item.is_constant or item.logical_type != "numerical"]
    columns = [str(column) for column in frame.columns if str(column) not in excluded]
    if not columns or len(frame) < 10:
        raise TrainingFailureError("Anomaly detection needs at least ten rows and one usable numerical feature.")
    matrix = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(frame[columns]))
    if request.algorithm == "isolation_forest":
        model = IsolationForest(contamination=request.contamination, random_state=random_state, n_jobs=1)
        labels = model.fit_predict(matrix)
        scores = model.decision_function(matrix)
    else:
        neighbors = min(20, len(frame) - 1)
        model = LocalOutlierFactor(n_neighbors=neighbors, contamination=request.contamination)
        labels = model.fit_predict(matrix)
        scores = model.negative_outlier_factor_
    indices = np.flatnonzero(labels == -1)
    ordered = sorted(indices, key=lambda index: float(scores[index]))[:20]
    return AnomalyResponse(dataset_id=dataset_id, algorithm=request.algorithm, anomaly_count=int(len(indices)), anomaly_percentage=round(len(indices) / len(frame) * 100, 4), analyzed_rows=len(frame), feature_columns=columns, excluded_columns=excluded, examples=[AnomalyExample(row_index=int(index), score=round(float(scores[index]), 6)) for index in ordered])


def forecast_series(dataset_id: str, frame: pd.DataFrame, request: ForecastRequest) -> ForecastResponse:
    if request.time_column not in frame or request.target_column not in frame:
        raise DatasetValidationError("The selected time or target column does not exist.")
    prepared = pd.DataFrame({"time": pd.to_datetime(frame[request.time_column], errors="coerce"), "target": pd.to_numeric(frame[request.target_column], errors="coerce")}).dropna().sort_values("time")
    if len(prepared) < 12 or prepared["time"].nunique() != len(prepared):
        raise DatasetValidationError("Forecasting requires at least 12 uniquely ordered temporal observations with a numerical target.")
    split = max(2, int(len(prepared) * (1 - request.test_fraction)))
    train, test = prepared.iloc[:split], prepared.iloc[split:]
    if len(test) < 2:
        raise DatasetValidationError("The chronological test split is too small.")
    if request.algorithm == "naive_forecast":
        predicted = np.repeat(float(train["target"].iloc[-1]), len(test))
    elif request.algorithm == "seasonal_naive":
        period = request.seasonal_period
        if period >= len(train):
            raise DatasetValidationError("Seasonal period must be smaller than the training history.")
        history = train["target"].iloc[-period:].to_numpy()
        predicted = np.resize(history, len(test))
    elif request.algorithm == "exponential_smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        predicted = np.asarray(ExponentialSmoothing(train["target"], trend="add", initialization_method="estimated").fit(optimized=True).forecast(len(test)))
    else:
        from statsmodels.tsa.arima.model import ARIMA
        predicted = np.asarray(ARIMA(train["target"], order=(1, 1, 1)).fit().forecast(len(test)))
    actual = test["target"].to_numpy(dtype=float)
    errors = actual - predicted
    non_zero = actual != 0
    mape = float(np.mean(np.abs(errors[non_zero] / actual[non_zero])) * 100) if non_zero.all() else None
    return ForecastResponse(dataset_id=dataset_id, algorithm=request.algorithm, training_rows=len(train), test_rows=len(test), metrics=ForecastMetric(mae=round(float(np.mean(np.abs(errors))), 6), rmse=round(float(np.sqrt(np.mean(errors ** 2))), 6), mape=round(mape, 6) if mape is not None else None), timestamps=[value.isoformat() for value in test["time"]], actual=[round(float(value), 6) for value in actual], predicted=[round(float(value), 6) for value in predicted])
