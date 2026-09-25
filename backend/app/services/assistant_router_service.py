"""Evidence routing for Ask AutoDS; routing never supplies analytical answers."""
import json
import re
from typing import Iterable

from app.llm.base import LLMProvider, ProviderFailure
from app.schemas.assistant import EvidenceRoutingDecision


_ANALYTICS_TERMS = re.compile(
    r"\b(?:average|mean|median|sum|total|count|many|rate|share|proportion|distribution|"
    r"correlation|associated|outlier|highest|lowest|compare|group|rows?|columns?|missing|duplicate)\b",
    re.I,
)
_EXPERIMENT_TERMS = re.compile(
    r"\b(?:model|validation|test metric|accuracy|precision|recall|f1|roc.?auc|rmse|mae|r.?squared|"
    r"pipeline|optimization|parameter|feature importance|shap|prediction|report|trained|overfitting)\b",
    re.I,
)
_RESEARCH_TERMS = re.compile(r"\b(?:research|paper|literature|citation|published|study|studies)\b", re.I)
_GENERAL_FORM = re.compile(r"^\s*(?:what (?:is|does)|define|explain|why (?:use|is)|how does)\b", re.I)
_FOLLOW_UP = re.compile(r"^\s*(?:what about|how about|and\b|now\b|only\b|what if|for those|among those|why did you|why was that)", re.I)


def _column_mentioned(question: str, columns: Iterable[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower())
    for column in columns:
        label = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        if label and (re.search(rf"\b{re.escape(label)}s?\b", normalized) or any(len(token) >= 4 and token in normalized.split() for token in label.split())):
            return True
    return False


def deterministic_route(question: str, columns: Iterable[str] = (), has_conversation: bool = False) -> EvidenceRoutingDecision | None:
    """Route common evidence needs generically; never return an answer or value."""
    column_signal = _column_mentioned(question, columns)
    dataset_signal = column_signal or bool(_ANALYTICS_TERMS.search(question))
    experiment_signal = bool(_EXPERIMENT_TERMS.search(question))
    research_signal = bool(_RESEARCH_TERMS.search(question))
    follow_up = has_conversation and bool(_FOLLOW_UP.search(question))
    if research_signal and not dataset_signal and not experiment_signal:
        return EvidenceRoutingDecision(evidence_type="research", uses_research=True, reason="The question explicitly asks for research evidence.")
    if _GENERAL_FORM.search(question) and not column_signal and not re.search(r"\b(?:my|selected|was|result|score|performed|experiment)\b", question, re.I):
        return EvidenceRoutingDecision(evidence_type="general", reason="The question asks for a general data-science explanation.")
    if dataset_signal and experiment_signal:
        return EvidenceRoutingDecision(evidence_type="dataset_experiment", needs_dataset=True, needs_experiment=True, uses_conversation=follow_up, reason="The question combines dataset statistics with experiment evidence.")
    if experiment_signal:
        return EvidenceRoutingDecision(evidence_type="experiment", needs_experiment=True, uses_conversation=follow_up, reason="The question asks about a trained experiment or model evidence.")
    if dataset_signal:
        return EvidenceRoutingDecision(evidence_type="conversation" if follow_up else "dataset", needs_dataset=True, uses_conversation=follow_up, reason="The question requests a calculation from dataset records.")
    if follow_up:
        return EvidenceRoutingDecision(evidence_type="conversation", uses_conversation=True, reason="The question depends on the previous analytical topic.")
    if _GENERAL_FORM.search(question):
        return EvidenceRoutingDecision(evidence_type="general", reason="The question asks for a general data-science explanation.")
    return None


def route_question(
    question: str,
    columns: Iterable[str],
    providers: list[LLMProvider],
    conversation_summary: str = "",
) -> tuple[EvidenceRoutingDecision, str]:
    local = deterministic_route(question, columns, bool(conversation_summary))
    if local is not None:
        return local, "deterministic_router"
    schema = EvidenceRoutingDecision.model_json_schema()
    prompt = (
        "Classify which evidence is required to answer the question. Return JSON only. "
        "Dataset evidence means a calculation over rows. Experiment evidence means persisted pipeline/model/evaluation artifacts. "
        "dataset_experiment requires values from both. Conversation is intent context only, never numeric truth. "
        "Research means source-backed literature. General means conceptual data-science knowledge. Unsupported means unsafe or unrelated. "
        "Do not answer the question and do not invent available evidence.\n"
        f"AVAILABLE_DATASET_COLUMNS={json.dumps([str(item) for item in columns])}\n"
        f"{conversation_summary}\nQUESTION={question}\nSCHEMA={json.dumps(schema)}"
    )
    for provider in providers:
        try:
            raw = provider.generate_structured(prompt, schema)
            decision = EvidenceRoutingDecision.model_validate_json(raw) if isinstance(raw, str) else EvidenceRoutingDecision.model_validate(raw)
            return decision, provider.name
        except (ProviderFailure, ValueError):
            continue
    return EvidenceRoutingDecision(evidence_type="unsupported", reason="No safe evidence route could be established."), "deterministic_router"
