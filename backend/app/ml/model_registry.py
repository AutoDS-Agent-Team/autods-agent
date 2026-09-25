from dataclasses import dataclass

from app.schemas.auto_analysis import AnalysisTask


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm_id: str
    display_name: str
    tasks: frozenset[AnalysisTask]
    sparse_support: bool
    scaling_required: bool
    prediction_support: bool
    probability_support: bool
    max_rows: int | None = None


def _tasks(*values: AnalysisTask) -> frozenset[AnalysisTask]:
    return frozenset(values)


MODEL_REGISTRY: dict[str, AlgorithmSpec] = {
    "logistic_regression": AlgorithmSpec("logistic_regression", "Logistic Regression", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), True, True, True, True),
    "random_forest_classifier": AlgorithmSpec("random_forest_classifier", "Random Forest Classifier", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), False, False, True, True),
    "xgboost_classifier": AlgorithmSpec("xgboost_classifier", "XGBoost Classifier", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), True, False, True, True),
    "decision_tree_classifier": AlgorithmSpec("decision_tree_classifier", "Decision Tree Classifier", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), False, False, True, True),
    "knn_classifier": AlgorithmSpec("knn_classifier", "K-Nearest Neighbors", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), False, True, True, True, 50000),
    "svc": AlgorithmSpec("svc", "Support Vector Classifier", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), True, True, True, True, 20000),
    "gaussian_nb": AlgorithmSpec("gaussian_nb", "Gaussian Naive Bayes", _tasks(AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION), False, False, True, True),
    "linear_regression": AlgorithmSpec("linear_regression", "Linear Regression", _tasks(AnalysisTask.REGRESSION), True, True, True, False),
    "ridge_regression": AlgorithmSpec("ridge_regression", "Ridge", _tasks(AnalysisTask.REGRESSION), True, True, True, False),
    "lasso_regression": AlgorithmSpec("lasso_regression", "Lasso", _tasks(AnalysisTask.REGRESSION), False, True, True, False),
    "random_forest_regressor": AlgorithmSpec("random_forest_regressor", "Random Forest Regressor", _tasks(AnalysisTask.REGRESSION), False, False, True, False),
    "xgboost_regressor": AlgorithmSpec("xgboost_regressor", "XGBoost Regressor", _tasks(AnalysisTask.REGRESSION), True, False, True, False),
    "decision_tree_regressor": AlgorithmSpec("decision_tree_regressor", "Decision Tree Regressor", _tasks(AnalysisTask.REGRESSION), False, False, True, False),
    "knn_regressor": AlgorithmSpec("knn_regressor", "KNN Regressor", _tasks(AnalysisTask.REGRESSION), False, True, True, False, 50000),
    "svr": AlgorithmSpec("svr", "Support Vector Regressor", _tasks(AnalysisTask.REGRESSION), True, True, True, False, 20000),
    "kmeans": AlgorithmSpec("kmeans", "K-Means", _tasks(AnalysisTask.CLUSTERING), False, True, False, False),
    "dbscan": AlgorithmSpec("dbscan", "DBSCAN", _tasks(AnalysisTask.CLUSTERING), False, True, False, False, 50000),
    "agglomerative": AlgorithmSpec("agglomerative", "Agglomerative Clustering", _tasks(AnalysisTask.CLUSTERING), False, True, False, False, 20000),
    "isolation_forest": AlgorithmSpec("isolation_forest", "Isolation Forest", _tasks(AnalysisTask.ANOMALY_DETECTION), False, True, True, False),
    "local_outlier_factor": AlgorithmSpec("local_outlier_factor", "Local Outlier Factor", _tasks(AnalysisTask.ANOMALY_DETECTION), False, True, False, False, 50000),
    "naive_forecast": AlgorithmSpec("naive_forecast", "Naive Baseline", _tasks(AnalysisTask.TIME_SERIES_FORECASTING), False, False, True, False),
    "seasonal_naive": AlgorithmSpec("seasonal_naive", "Seasonal Naive", _tasks(AnalysisTask.TIME_SERIES_FORECASTING), False, False, True, False),
    "exponential_smoothing": AlgorithmSpec("exponential_smoothing", "Exponential Smoothing", _tasks(AnalysisTask.TIME_SERIES_FORECASTING), False, False, True, False),
}


def algorithms_for_task(task: AnalysisTask) -> list[AlgorithmSpec]:
    return [spec for spec in MODEL_REGISTRY.values() if task in spec.tasks]


def require_registered(algorithm_id: str, task: AnalysisTask) -> AlgorithmSpec:
    spec = MODEL_REGISTRY.get(algorithm_id)
    if spec is None or task not in spec.tasks:
        raise ValueError(f"Unsupported algorithm '{algorithm_id}' for task '{task.value}'.")
    return spec

