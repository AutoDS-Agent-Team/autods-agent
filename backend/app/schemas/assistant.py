from typing import Literal
from pydantic import BaseModel, Field, model_validator

EvidenceType = Literal["dataset", "experiment", "dataset_experiment", "conversation", "general", "research", "ambiguous", "unsupported"]


class EvidenceRoutingDecision(BaseModel):
    evidence_type: EvidenceType
    needs_dataset: bool = False
    needs_experiment: bool = False
    uses_conversation: bool = False
    uses_research: bool = False
    reason: str = Field(max_length=300)

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> "EvidenceRoutingDecision":
        required = {
            "dataset": (True, False),
            "experiment": (False, True),
            "dataset_experiment": (True, True),
        }
        if self.evidence_type in required:
            needs_dataset, needs_experiment = required[self.evidence_type]
            if (self.needs_dataset, self.needs_experiment) != (needs_dataset, needs_experiment):
                raise ValueError("Evidence type and required evidence flags must agree.")
        elif self.evidence_type == "conversation":
            if not self.uses_conversation or not (self.needs_dataset or self.needs_experiment):
                raise ValueError("Conversation evidence requires conversation context and a source evidence type.")
        elif self.evidence_type == "research" and not self.uses_research:
            raise ValueError("Research evidence must enable research retrieval.")
        elif self.evidence_type in {"general", "research", "ambiguous", "unsupported"} and (self.needs_dataset or self.needs_experiment):
            raise ValueError("This evidence type cannot require dataset or experiment evidence.")
        return self

class AssistantQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    context_type: Literal["auto", "dataset", "experiment"] | None = "auto"
    experiment_id: str | None = None
    dataset_id: str | None = None
    conversation_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @model_validator(mode="after")
    def validate_context_ids(self) -> "AssistantQueryRequest":
        if self.context_type == "dataset" and self.experiment_id and not self.dataset_id:
            raise ValueError("Dataset context requires a dataset_id when experiment_id is provided.")
        if self.context_type == "experiment" and self.dataset_id and not self.experiment_id:
            raise ValueError("Experiment context requires an experiment_id when dataset_id is provided.")
        return self

class AssistantSource(BaseModel):
    experiment_id: str | None = None
    dataset_id: str | None = None
    document_type: str = "verified_experiment_summary"


class GeneralExplanationClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=500)
    basis: Literal["general_knowledge", "curated_research"]


class GeneralExplanationResponse(BaseModel):
    explanation: str = Field(min_length=1, max_length=3000)
    claims: list[GeneralExplanationClaim] = Field(default_factory=list, max_length=8)

class AssistantQueryResponse(BaseModel):
    answer: str
    sources: list[AssistantSource]
    provider_used: str = "verified_retrieval"
    structured_result: dict | None = None
    evidence_type: EvidenceType = "experiment"
    provenance_label: str = "Verified from experiment"
    conversation_id: str = "default"
