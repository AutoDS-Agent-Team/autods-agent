import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.ml.registry import classification_model_factories
from app.schemas.pipeline_plan import ModelName, PipelinePlan
from app.services.adaptive_pipeline_service import adapt_pipeline


def binary_plan(**overrides):
    values = {
        "task_type": "binary_classification", "target_column": "target",
        "numeric_imputation": "median", "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot", "numeric_scaling": "standard",
        "models": ["logistic_regression", "random_forest", "xgboost"], "primary_metric": "f1",
    }
    values.update(overrides)
    return PipelinePlan.model_validate(values)


def test_verified_imbalance_enables_only_supported_balanced_estimators():
    size = 240
    frame = pd.DataFrame({
        "age": np.r_[np.arange(size - 12, dtype=float), [10000.0] * 12],
        "segment": [f"segment-{index}" for index in range(size)],
        "target": ["major"] * (size - 12) + ["minor"] * 12,
    })
    frame.attrs["categorical_columns"] = {"segment"}
    frame.loc[0:10, "age"] = np.nan
    updated, decisions, recommendations = adapt_pipeline(frame, binary_plan())
    assert updated.balanced_class_weight_models == [ModelName.LOGISTIC_REGRESSION, ModelName.RANDOM_FOREST]
    assert ModelName.XGBOOST not in updated.balanced_class_weight_models
    assert "Class imbalance detected" in {item.decision for item in decisions}
    assert any("high-cardinality" in item.decision.lower() for item in decisions)
    assert any("Skewed numerical" in item.decision for item in decisions)
    assert {item.model_name for item in recommendations} == set(updated.models)
    factories = classification_model_factories(42, {name.value for name in updated.balanced_class_weight_models})
    assert factories["logistic_regression"]().class_weight == "balanced"
    assert factories["random_forest"]().class_weight == "balanced"
    assert factories["xgboost"]().get_params().get("scale_pos_weight") in (None, 1)


def test_manual_override_can_disable_class_weight_and_schema_rejects_unsupported_model():
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": ["a", "a", "a", "a", "a", "b"]})
    plan, decisions, _ = adapt_pipeline(frame, binary_plan(), apply_recommendations=False)
    assert not plan.balanced_class_weight_models
    imbalance = next(item for item in decisions if item.decision == "Class imbalance detected")
    assert "Manual override" in imbalance.action_taken
    with pytest.raises(ValidationError):
        binary_plan(balanced_class_weight_models=["xgboost"])
