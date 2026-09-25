from collections.abc import Callable
from typing import Any, TypeVar

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from app.core.exceptions import TrainingFailureError

Factory = Callable[[], Any]
T = TypeVar("T")

NUMERIC_IMPUTERS: dict[str, Factory] = {
    "mean": lambda: SimpleImputer(strategy="mean", keep_empty_features=True),
    "median": lambda: SimpleImputer(strategy="median", keep_empty_features=True),
}
CATEGORICAL_IMPUTERS: dict[str, Factory] = {
    "most_frequent": lambda: SimpleImputer(
        strategy="most_frequent", keep_empty_features=True
    ),
}
ENCODERS: dict[str, Factory] = {
    "one_hot": lambda: OneHotEncoder(handle_unknown="ignore"),
}
SCALERS: dict[str, Factory] = {
    "standard": StandardScaler,
    "none": lambda: "passthrough",
}


def classification_model_factories(random_state: int, balanced_class_weight_models: set[str] | None = None) -> dict[str, Callable[[], BaseEstimator]]:
    balanced = balanced_class_weight_models or set()
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=500, random_state=random_state,
            class_weight="balanced" if "logistic_regression" in balanced else None,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=1,
            random_state=random_state,
            class_weight="balanced" if "random_forest" in balanced else None,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=1,
            random_state=random_state,
            tree_method="hist",
        ),
    }


def regression_model_factories(random_state: int) -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "ridge_regression": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            n_jobs=1,
            random_state=random_state,
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=1,
            random_state=random_state,
            tree_method="hist",
        ),
    }


def trusted_factory(registry: dict[str, Callable[[], T]], key: str, category: str) -> T:
    factory = registry.get(key)
    if factory is None:
        raise TrainingFailureError(f"Unsupported trusted {category}: {key}.")
    return factory()


def tunable_model_factory(
    model_name: str, *, classification: bool, random_state: int, parameters: dict[str, Any], balanced_class_weight: bool = False
) -> BaseEstimator:
    """Create only an allowlisted model with trusted, bounded optimization values."""
    if classification:
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=500, random_state=random_state, C=parameters["C"], class_weight="balanced" if balanced_class_weight else None)
        if model_name == "random_forest":
            return RandomForestClassifier(random_state=random_state, n_jobs=1, class_weight="balanced" if balanced_class_weight else None, **parameters)
        if model_name == "xgboost":
            return XGBClassifier(random_state=random_state, n_jobs=1, tree_method="hist", **parameters)
    else:
        if model_name == "ridge_regression":
            return Ridge(alpha=parameters["alpha"])
        if model_name == "random_forest":
            return RandomForestRegressor(random_state=random_state, n_jobs=1, **parameters)
        if model_name == "xgboost":
            return XGBRegressor(random_state=random_state, n_jobs=1, tree_method="hist", **parameters)
    raise TrainingFailureError(f"Unsupported trusted model optimization: {model_name}.")
