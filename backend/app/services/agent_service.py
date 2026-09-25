"""Small, bounded LLM agents that advise around trusted deterministic services."""
import json
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMProvider, ProviderFailure
from app.schemas.agents import AgentSeverity, DataAnalystResponse, DataQualityIssue
from app.schemas.dataset import DatasetProfileResponse
from app.schemas.experiment import InferenceConfidence, MLTaskType


def _profile_context(profile: DatasetProfileResponse) -> dict[str, Any]:
    return {
        "rows": profile.summary.row_count,
        "columns": profile.summary.column_count,
        "duplicates": profile.summary.duplicate_row_count,
        "total_missing": profile.summary.total_missing_values,
        "column_profiles": [
            {"name": item.name, "logical_type": item.logical_type, "missing_count": item.missing_count,
             "missing_percentage": item.missing_percentage, "unique_count": item.unique_count,
             "is_possible_id": item.is_possible_id, "is_constant": item.is_constant}
            for item in profile.columns
        ],
    }


def _fallback_profile_analysis(profile: DatasetProfileResponse) -> DataAnalystResponse:
    issues, observations, preprocessing = [], [], []
    for column in profile.columns:
        if column.missing_count:
            severity = AgentSeverity.HIGH if column.missing_percentage >= 50 else AgentSeverity.MEDIUM
            issues.append(DataQualityIssue(column=column.name, issue="missing_values", severity=severity, explanation=f"{column.name} has verified missing values and needs the configured imputation strategy."))
            preprocessing.append(f"Review missing values in {column.name} before training.")
        if column.is_possible_id:
            observations.append(f"{column.name} appears identifier-like and should be reviewed before use as a feature.")
    return DataAnalystResponse(summary=f"Verified profile contains {profile.summary.row_count} rows and {profile.summary.column_count} columns.", quality_issues=issues, feature_observations=observations, possible_targets=[], suggested_target=None, suggested_task=None, confidence=InferenceConfidence.NONE, reasoning_summary="AI interpretation is unavailable; deterministic profile evidence remains available for target confirmation.", preprocessing_observations=preprocessing, provider_used="verified_profile", status="FALLBACK")


def analyze_profile(profile: DatasetProfileResponse, providers: tuple[LLMProvider, LLMProvider], max_retries: int) -> DataAnalystResponse:
    context = _profile_context(profile)
    allowed_columns = {item.name for item in profile.columns}
    prompt = ("You are the Data Analyst Agent. Interpret only this bounded deterministic profile metadata. "
              "Return JSON matching the schema. Do not calculate or claim statistics not present. Do not include IDs, paths, raw data, code, SQL, or commands. "
              f"VERIFIED_PROFILE={json.dumps(context, sort_keys=True)}")
    for provider in providers:
        for _ in range(max_retries + 1):
            try:
                raw = provider.generate_structured(prompt, DataAnalystResponse.model_json_schema())
                result = DataAnalystResponse.model_validate_json(raw) if isinstance(raw, str) else DataAnalystResponse.model_validate(raw)
                referenced = {issue.column for issue in result.quality_issues} | set(result.possible_targets)
                if result.suggested_target:
                    referenced.add(result.suggested_target)
                if not referenced.issubset(allowed_columns):
                    continue
                return result.model_copy(update={"provider_used": provider.name, "status": "COMPLETED"})
            except (ProviderFailure, ValidationError):
                continue
    return _fallback_profile_analysis(profile)
