from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.schemas.experiment import InferenceConfidence, MLTaskType


class AgentSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataQualityIssue(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    issue: str = Field(min_length=1, max_length=64)
    severity: AgentSeverity
    explanation: str = Field(min_length=1, max_length=500)


class DataAnalystResponse(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    quality_issues: list[DataQualityIssue] = Field(default_factory=list, max_length=30)
    feature_observations: list[str] = Field(default_factory=list, max_length=30)
    possible_targets: list[str] = Field(default_factory=list, max_length=30)
    suggested_target: str | None = None
    suggested_task: MLTaskType | None = None
    confidence: InferenceConfidence
    reasoning_summary: str = Field(min_length=1, max_length=1200)
    preprocessing_observations: list[str] = Field(default_factory=list, max_length=30)
    provider_used: str = ""
    status: str = ""

    @field_validator("possible_targets", "feature_observations", "preprocessing_observations")
    @classmethod
    def strip_values(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class ResearchCitation(BaseModel):
    title: str
    source_url: str
    year: int | None = None


class InsightRecommendation(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=600)
    suggested_action: str = Field(min_length=1, max_length=300)
    requires_retraining: bool
    research_sources: list[ResearchCitation] = Field(default_factory=list)


class InsightFeature(BaseModel):
    feature: str = Field(min_length=1, max_length=255)
    explanation: str = Field(min_length=1, max_length=500)


class ResearchEvidence(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    source_url: str
    evidence_summary: str
    relevance_score: float


class SimilarExperimentEvidence(BaseModel):
    experiment_id: str
    dataset_filename: str | None = None
    objective: str | None = None
    selected_model: str | None = None
    primary_metric: str | None = None
    validation_score: float | None = None
    test_score: float | None = None
    match_reason: str


class DecisionExplanation(BaseModel):
    stage: str
    decision: str
    rationale: str
    evidence_source: str = "persisted experiment data"


class InsightReflectionResponse(BaseModel):
    executive_summary: str = Field(min_length=1, max_length=1200)
    model_assessment: str = Field(min_length=1, max_length=1200)
    strengths: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    important_features: list[InsightFeature] = Field(default_factory=list, max_length=20)
    recommendations: list[InsightRecommendation] = Field(default_factory=list)
    research_evidence: list[ResearchEvidence] = Field(default_factory=list)
    research_status: str = "NO_RELEVANT_EVIDENCE"
    similar_experiments: list[SimilarExperimentEvidence] = Field(default_factory=list)
    decision_explanations: list[DecisionExplanation] = Field(default_factory=list)
    provider_used: str = ""
    status: str = ""
