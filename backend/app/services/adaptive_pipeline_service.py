"""Deterministic, profile-grounded pipeline guidance and allowlisted adaptations."""
from math import ceil, sqrt
from typing import Any

import pandas as pd

from app.schemas.experiment import MLTaskType
from app.schemas.pipeline_plan import (
    AdaptiveDecision,
    ModelName,
    ModelRecommendationReason,
    PipelinePlan,
)
from app.services.profiling_service import profile_dataframe


def adapt_pipeline(dataframe: pd.DataFrame, plan: PipelinePlan, *, apply_recommendations: bool = True) -> tuple[PipelinePlan, list[AdaptiveDecision], list[ModelRecommendationReason]]:
    _, columns, _ = profile_dataframe(dataframe)
    target = dataframe[plan.target_column].dropna()
    numeric = [item for item in columns if item.logical_type == "numerical" and item.name != plan.target_column]
    categorical = [item for item in columns if item.logical_type == "categorical" and item.name != plan.target_column]
    text = [item for item in columns if item.logical_type == "text" and item.name != plan.target_column]
    datetimes = [item for item in columns if item.logical_type == "datetime" and item.name != plan.target_column]
    decisions: list[AdaptiveDecision] = []
    rows = len(dataframe)

    numeric_missing = sum(item.missing_count for item in numeric)
    categorical_missing = sum(item.missing_count for item in categorical)
    if numeric_missing or categorical_missing:
        decisions.append(AdaptiveDecision(
            decision="Use configured imputers for missing predictors",
            reason="Observed missing values require a deterministic fill policy during training.",
            evidence=f"{numeric_missing:,} missing numeric and {categorical_missing:,} missing categorical predictor values across {rows:,} rows.",
            action_taken=f"Trusted preprocessing applies {plan.numeric_imputation.value} numeric and {plan.categorical_imputation.value} categorical imputation, fitted on training data only.",
            applied=True,
        ))
    else:
        decisions.append(AdaptiveDecision(
            decision="No missing predictor values detected",
            reason="No fill operation is needed for the observed feature values.",
            evidence=f"0 missing values across {len(numeric)} numeric and {len(categorical)} categorical predictors.",
            action_taken="The allowlisted imputers remain in the pipeline as a safe no-op for this dataset.",
            applied=True,
        ))

    target_distribution = target.value_counts(normalize=True, dropna=True)
    class_weight_models = []
    if plan.task_type in {MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION} and len(target_distribution) > 1:
        minority_share = float(target_distribution.min())
        if minority_share < 0.30:
            supported_weight_models = [name for name in plan.models if name in {ModelName.LOGISTIC_REGRESSION, ModelName.RANDOM_FOREST}]
            class_weight_models = supported_weight_models if apply_recommendations else list(plan.balanced_class_weight_models)
            decisions.append(AdaptiveDecision(
                decision="Class imbalance detected",
                reason="A minority class below 30% can be underrepresented by unweighted loss.",
                evidence=f"{len(target_distribution)} target classes; minority class share {minority_share:.1%}.",
                action_taken=(f"Balanced class weights enabled for {', '.join(item.value for item in class_weight_models)}." if class_weight_models else ("Manual override leaves class weights disabled." if not apply_recommendations else "No weighted estimator is in the current plan; recommendation only.")),
                applied=bool(class_weight_models),
            ))
        else:
            decisions.append(AdaptiveDecision(
                decision="No material class imbalance detected",
                reason="Observed target proportions do not cross the conservative imbalance trigger.",
                evidence=f"Minority class share {minority_share:.1%} across {len(target_distribution)} classes.",
                action_taken="No class weighting added.", applied=True,
            ))
    updated = plan.model_copy(update={"balanced_class_weight_models": class_weight_models}) if apply_recommendations else plan

    high_cardinality = [item for item in categorical if item.unique_count > max(20, ceil(sqrt(max(rows, 1)))) and item.unique_count / max(rows, 1) >= 0.05]
    if high_cardinality:
        descriptions = ", ".join(f"{item.name} ({item.unique_count:,} distinct)" for item in high_cardinality[:5])
        decisions.append(AdaptiveDecision(
            decision="Review high-cardinality categorical features",
            reason="One-hot expansion can increase memory and model cost when a categorical feature has many observed values.",
            evidence=f"{descriptions}; one-hot expansion remains the only validated categorical encoder in the current allowlist.",
            action_taken="No unsupported encoding was introduced; the validated one-hot encoder remains active. Consider an explicit manual feature redesign before training.",
            applied=False,
        ))

    skewed = []
    outlier_columns = []
    for item in numeric:
        values = pd.to_numeric(dataframe[item.name], errors="coerce").dropna()
        if len(values) >= 8 and abs(float(values.skew())) >= 1.0:
            skewed.append((item.name, float(values.skew())))
        if len(values) >= 8:
            q1, q3 = values.quantile([0.25, 0.75])
            spread = float(q3 - q1)
            if spread > 0:
                count = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
                if count:
                    outlier_columns.append((item.name, count))
    if skewed:
        decisions.append(AdaptiveDecision(
            decision="Skewed numerical predictors identified",
            reason="Strong skew can affect scale-sensitive models; transformations should be validated rather than assumed beneficial.",
            evidence="; ".join(f"{name} skew={value:.2f}" for name, value in skewed[:5]),
            action_taken="No log/power transform is applied because the current validated preprocessing allowlist has no such transform; retained for manual review.",
            applied=False,
        ))
    if outlier_columns:
        decisions.append(AdaptiveDecision(
            decision="Potential numerical outliers identified",
            reason="IQR fences flag observations for review; they do not establish data errors.",
            evidence="; ".join(f"{name}: {count:,} IQR-flagged rows" for name, count in outlier_columns[:5]),
            action_taken="Rows are retained; no unsupported clipping or removal is performed.", applied=False,
        ))
    if datetimes:
        decisions.append(AdaptiveDecision(
            decision="Datetime features detected",
            reason="Chronological structure may matter for forecasting or leakage-safe evaluation.",
            evidence=f"{len(datetimes)} datetime predictor(s): {', '.join(item.name for item in datetimes[:5])}.",
            action_taken="No time-aware split/feature extraction is introduced into this tabular ML workflow; review chronology before interpreting results.", applied=False,
        ))
    if rows < 500:
        decisions.append(AdaptiveDecision(
            decision="Small-data optimization budget",
            reason="A bounded search avoids spending disproportionate compute on a small sample.",
            evidence=f"{rows:,} dataset rows.",
            action_taken="If optimization is requested, trusted code caps the search at 3 trials.", applied=True,
        ))
    elif rows > 50_000:
        decisions.append(AdaptiveDecision(
            decision="Large-data optimization budget",
            reason="Limit repeated fits on a large dataset while preserving the configured primary metric.",
            evidence=f"{rows:,} dataset rows.",
            action_taken="If optimization is requested, trusted code caps the search at 5 trials.", applied=True,
        ))

    evidence = f"{rows:,} rows; {len(numeric)} numeric, {len(categorical)} categorical, {len(text)} text, {len(datetimes)} datetime predictors."
    reasons: list[ModelRecommendationReason] = []
    for model in updated.models:
        if model == ModelName.LOGISTIC_REGRESSION:
            why = "Provides a compact linear classification baseline; balanced class weights are enabled when the verified target profile triggers the imbalance rule." 
        elif model == ModelName.RIDGE_REGRESSION:
            why = "Provides a regularized linear regression baseline for comparison against nonlinear estimators."
        elif model == ModelName.RANDOM_FOREST:
            why = "Provides a tree-ensemble baseline for nonlinear effects and mixed engineered features without requiring feature scaling." 
        else:
            why = "Provides a gradient-boosted tree baseline to compare nonlinear predictive performance with the other planned candidates." 
        reasons.append(ModelRecommendationReason(model_name=model, reason=why.strip(), dataset_evidence=evidence))
    return updated, decisions, reasons
