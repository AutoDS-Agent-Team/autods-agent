from datetime import datetime, timezone

from app.llm.base import ProviderFailure
from app.schemas.dataset import ColumnProfile, DatasetProfileResponse, DatasetProfileSummary, DatasetResponse
from app.services.agent_service import analyze_profile


def profile() -> DatasetProfileResponse:
    return DatasetProfileResponse(
        dataset=DatasetResponse(id="dataset", original_filename="sample.csv", stored_filename="internal.csv", file_size=10, row_count=12, column_count=2, has_header=True, created_at=datetime.now(timezone.utc)),
        summary=DatasetProfileSummary(row_count=12, column_count=2, duplicate_row_count=0, total_missing_values=3, numerical_columns=["Age"], categorical_columns=["Survived"], constant_columns=[], possible_id_columns=[]),
        columns=[ColumnProfile(name="Age", dtype="float", logical_type="numerical", missing_count=3, missing_percentage=25.0, unique_count=9, is_constant=False, is_possible_id=False), ColumnProfile(name="Survived", dtype="int", logical_type="numerical", missing_count=0, missing_percentage=0.0, unique_count=2, is_constant=False, is_possible_id=False)],
        correlations=[],
    )


class Provider:
    name = "gemini"
    def __init__(self, output, fails=False): self.output, self.fails, self.prompts = output, fails, []
    def generate_structured(self, prompt, _schema):
        self.prompts.append(prompt)
        if self.fails: raise ProviderFailure(self.name, "unavailable", "unavailable")
        return self.output


def test_data_analyst_receives_bounded_profile_and_validates_output() -> None:
    provider = Provider({"summary": "Age needs imputation.", "quality_issues": [{"column": "Age", "issue": "missing_values", "severity": "medium", "explanation": "Verified missing values exist."}], "feature_observations": [], "possible_targets": ["Survived"], "suggested_target": "Survived", "suggested_task": "binary_classification", "confidence": "high", "reasoning_summary": "Two target values are verified.", "preprocessing_observations": []})
    result = analyze_profile(profile(), (provider, provider), 0)
    assert result.provider_used == "gemini" and result.suggested_target == "Survived"
    assert "sample.csv" not in provider.prompts[0] and "VERIFIED_PROFILE" in provider.prompts[0]


def test_invalid_agent_output_is_rejected_and_failure_preserves_profile() -> None:
    invalid = Provider({"summary": "x", "quality_issues": [{"column": "not-a-column", "issue": "missing_values", "severity": "high", "explanation": "x"}], "feature_observations": [], "possible_targets": [], "suggested_target": None, "suggested_task": None, "confidence": "low", "reasoning_summary": "x", "preprocessing_observations": []})
    fallback = Provider({}, fails=True)
    result = analyze_profile(profile(), (invalid, fallback), 0)
    assert result.provider_used == "verified_profile" and result.quality_issues
