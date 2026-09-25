import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app.ml.model_registry import algorithms_for_task
from app.schemas.auto_analysis import (
    AlgorithmRecommendation,
    AnalysisTask,
    ConfidenceLevel,
    DatasetAutoAnalysisResponse,
    FeatureRole,
    FeatureRoleResult,
    TargetAlternative,
    TargetRecommendation,
)
from app.services.profiling_service import profile_dataframe

TARGET_NAME_TOKENS = {
    "target", "label", "class", "outcome", "result", "survived", "churn",
    "fraud", "rating", "price", "salary", "revenue", "sales", "variety",
    "species", "diagnosis", "status", "quality",
}


def _words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower()))


def detect_feature_roles(dataframe: pd.DataFrame) -> list[FeatureRoleResult]:
    _, profiles, _ = profile_dataframe(dataframe)
    results: list[FeatureRoleResult] = []
    for profile in profiles:
        series = dataframe[profile.name]
        if profile.is_possible_id:
            role, reason = FeatureRole.IDENTIFIER, "High uniqueness or identifier-like column name."
        elif is_bool_dtype(series.dtype) or set(series.dropna().astype(str).str.lower().unique()) <= {"true", "false", "yes", "no"}:
            role, reason = FeatureRole.BOOLEAN, "Contains boolean values."
        elif profile.logical_type == "datetime":
            role, reason = FeatureRole.DATETIME, "Parsed as temporal values."
        elif profile.logical_type == "text":
            role, reason = FeatureRole.TEXT, "Contains varied multi-word text."
        elif profile.logical_type == "numerical":
            role, reason = FeatureRole.NUMERICAL, "Contains numeric measurements."
        else:
            raw = series.dropna().astype(str)
            looks_temporal = bool(len(raw)) and float(raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$").mean()) >= 0.9
            candidate = pd.to_datetime(raw, errors="coerce") if looks_temporal else pd.Series(dtype="datetime64[ns]")
            if len(candidate) >= 3 and float(candidate.notna().mean()) >= 0.9:
                role, reason = FeatureRole.DATETIME, "Values consistently parse as dates/times."
            else:
                role, reason = FeatureRole.CATEGORICAL, "Contains a bounded set of labels."
        results.append(FeatureRoleResult(column=profile.name, role=role, excluded_by_default=role == FeatureRole.IDENTIFIER, reason=reason))
    return results


def _infer_task(series: pd.Series, role: FeatureRole) -> tuple[AnalysisTask | None, str]:
    values = series.dropna()
    unique = int(values.nunique())
    if unique == 2:
        return AnalysisTask.BINARY_CLASSIFICATION, "It contains exactly two outcome classes."
    if role in {FeatureRole.CATEGORICAL, FeatureRole.BOOLEAN} or (role == FeatureRole.NUMERICAL and unique <= 20 and bool((pd.to_numeric(values, errors="coerce") % 1 == 0).all())):
        return AnalysisTask.MULTICLASS_CLASSIFICATION, f"It contains {unique} discrete outcome values."
    if role == FeatureRole.NUMERICAL:
        return AnalysisTask.REGRESSION, "It is a continuous numerical outcome."
    return None, "Its values are not a supported supervised outcome."


def recommend_target(dataframe: pd.DataFrame, roles: list[FeatureRoleResult]) -> TargetRecommendation:
    role_by_column = {item.column: item.role for item in roles}
    scored: list[tuple[float, str, AnalysisTask | None, str]] = []
    row_count = max(len(dataframe), 1)
    for index, column in enumerate(dataframe.columns):
        name = str(column)
        role = role_by_column[name]
        series = dataframe[column]
        non_null = series.dropna()
        unique = int(non_null.nunique())
        if role == FeatureRole.IDENTIFIER or unique < 2 or not len(non_null):
            continue
        task, task_reason = _infer_task(series, role)
        if task is None or (task == AnalysisTask.MULTICLASS_CLASSIFICATION and unique > 100):
            continue
        tokens = _words(name)
        normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
        semantic = 0.55 if tokens & TARGET_NAME_TOKENS or any(token in normalized_name for token in TARGET_NAME_TOKENS) else 0.0
        cardinality = 0.18 if 2 <= unique <= 20 else (0.08 if role == FeatureRole.NUMERICAL else 0.0)
        position = 0.1 if index == len(dataframe.columns) - 1 else 0.0
        completeness = 0.1 * (len(non_null) / row_count)
        score = min(1.0, semantic + cardinality + position + completeness)
        scored.append((score, name, task, task_reason))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.4:
        return TargetRecommendation(recommended_target=None, recommended_task=None, confidence=ConfidenceLevel.NONE, reason="No column has enough deterministic evidence to be selected as a supervised target. Use EDA, clustering, or anomaly detection, or choose a target manually.", alternatives=[TargetAlternative(column=name, score=round(score, 3), suggested_task=task) for score, name, task, _ in scored[:5]])
    score, name, task, task_reason = scored[0]
    confidence = ConfidenceLevel.HIGH if score >= 0.8 else ConfidenceLevel.MEDIUM if score >= 0.6 else ConfidenceLevel.LOW
    ambiguity = " Low-cardinality integer outcomes may be ordinal; confirm the task." if is_numeric_dtype(dataframe[name]) and 3 <= dataframe[name].nunique() <= 20 else ""
    return TargetRecommendation(recommended_target=name, recommended_task=task, confidence=confidence, reason=f"{name} scored highest from its name, cardinality, position, completeness, and identifier checks. {task_reason}{ambiguity}", alternatives=[TargetAlternative(column=alt_name, score=round(alt_score, 3), suggested_task=alt_task) for alt_score, alt_name, alt_task, _ in scored[1:6]])


def select_algorithms(task: AnalysisTask | None, dataframe: pd.DataFrame, roles: list[FeatureRoleResult]) -> list[AlgorithmRecommendation]:
    if task is None:
        return []
    row_count, column_count = dataframe.shape
    has_text = any(role.role == FeatureRole.TEXT for role in roles)
    selected = []
    for spec in algorithms_for_task(task):
        if spec.max_rows is not None and row_count > spec.max_rows:
            continue
        if has_text and not spec.sparse_support:
            continue
        if task in {AnalysisTask.BINARY_CLASSIFICATION, AnalysisTask.MULTICLASS_CLASSIFICATION} and spec.algorithm_id not in {"logistic_regression", "random_forest_classifier", "xgboost_classifier", "decision_tree_classifier"}:
            continue
        if task == AnalysisTask.REGRESSION and spec.algorithm_id not in {"ridge_regression", "random_forest_regressor", "xgboost_regressor"}:
            continue
        reason = "Sparse-compatible candidate for text features." if has_text else ("Bounded linear baseline." if "regression" in spec.algorithm_id and "forest" not in spec.algorithm_id else "Nonlinear tree-based candidate.")
        selected.append(AlgorithmRecommendation(algorithm_id=spec.algorithm_id, display_name=spec.display_name, reason=reason))
        if len(selected) >= 3:
            break
    return selected


def analyze_dataset(dataset_id: str, dataframe: pd.DataFrame) -> DatasetAutoAnalysisResponse:
    roles = detect_feature_roles(dataframe)
    target = recommend_target(dataframe, roles)
    applicable = [AnalysisTask.CLUSTERING, AnalysisTask.ANOMALY_DETECTION]
    if target.recommended_task:
        applicable.insert(0, target.recommended_task)
    if any(role.role == FeatureRole.DATETIME for role in roles) and any(role.role == FeatureRole.NUMERICAL for role in roles):
        applicable.append(AnalysisTask.TIME_SERIES_FORECASTING)
    return DatasetAutoAnalysisResponse(dataset_id=dataset_id, feature_roles=roles, target_recommendation=target, applicable_tasks=applicable, recommended_algorithms=select_algorithms(target.recommended_task, dataframe, roles), warnings=["Identifiers are excluded by default."])
