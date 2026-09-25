"""Context-aware Ask AutoDS orchestration with verified evidence boundaries."""
import re
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import DatasetNotFoundError, ExperimentNotFoundError
from app.llm.base import LLMProvider, ProviderFailure
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.user import User
from app.schemas.analytics import Aggregation, AnalyticsPlanStep, DatasetAnalyticsPlan, PlanStepOperation
from app.schemas.assistant import AssistantQueryRequest, AssistantQueryResponse, AssistantSource, EvidenceRoutingDecision
from app.services.analytics_service import execute_analytics_plan
from app.services.assistant_conversation_service import (
    conversation_context,
    recent_turns,
    remembered_context,
    resolve_follow_up,
    save_turn,
)
from app.services.assistant_router_service import route_question
from app.services.assistant_service import answer_question
from app.services.dataset_answer_presentation_service import enrich_verified_dataset_result, format_verified_dataset_answer
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.experiment_memory_service import retrieve_similar_experiments, retrieve_verified_memory
from app.services.question_planner_service import UNSUPPORTED_QUESTION, _column, plan_open_ended_dataset_question
from app.services.research_rag_service import retrieve_research


_METRIC_ALIASES = {
    "roc_auc": ("roc auc", "roc-auc", "auc"),
    "r2": ("r2", "r²", "r squared"),
    "rmse": ("rmse",), "mae": ("mae",), "accuracy": ("accuracy",),
    "precision": ("precision",), "recall": ("recall",), "f1": ("f1", "f1 score"),
}
_PERCENT_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}


def _metric_label(name: str) -> str:
    return {"roc_auc": "ROC-AUC", "r2": "R²", "rmse": "RMSE", "mae": "MAE", "f1": "F1 score"}.get(name, name.replace("_", " ").title())


def _format_metric(name: str, value: float) -> str:
    return f"{value * 100:,.2f}%" if name in _PERCENT_METRICS else f"{value:,.3f}"


def _providers(settings: Settings, providers: list[LLMProvider] | None) -> list[LLMProvider]:
    return providers or []


def _owned_experiment(database: Session, user: User, experiment_id: str | None) -> Experiment | None:
    if not experiment_id:
        return None
    experiment = database.get(Experiment, experiment_id)
    if experiment is None or experiment.user_id != user.id:
        raise ExperimentNotFoundError("Experiment not found.")
    return experiment


def _linked_experiments(database: Session, user: User, dataset_id: str) -> list[Experiment]:
    return database.scalars(
        select(Experiment).where(Experiment.user_id == user.id, Experiment.dataset_id == dataset_id).order_by(Experiment.updated_at.desc()).limit(3)
    ).all()


def _ambiguous_response(request: AssistantQueryRequest, message: str, dataset_id: str | None, experiment_id: str | None) -> AssistantQueryResponse:
    return AssistantQueryResponse(
        answer=message,
        provider_used="evidence_router",
        sources=[AssistantSource(dataset_id=dataset_id, experiment_id=experiment_id, document_type="context_selection")],
        evidence_type="ambiguous",
        provenance_label="Context selection required",
        conversation_id=request.conversation_id,
    )


def _general_answer(question: str, providers: list[LLMProvider], research: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    research = research or []
    evidence = "\n".join(f"- {item['title']} ({item['year']}, {item['venue']}): {item['evidence_summary']}" for item in research)
    prompt = (
        "Give a concise general data-science explanation. Do not claim any dataset, experiment, model metric, or citation that is not supplied. "
        "This is general guidance, not a verified AutoDS result."
        + (f"\nCurated research evidence:\n{evidence}" if evidence else "")
        + f"\nQuestion: {question}"
    )
    for provider in providers:
        try:
            text = provider.generate_text(prompt).strip()
            if text:
                sources = "\n\n### Retrieved sources\n" + "\n".join(f"- [{item['title']}]({item['source_url']})" for item in research) if research else ""
                return f"### AI Explanation\n\n{text[:4000]}{sources}", provider.name
        except ProviderFailure:
            continue
    if research:
        sources = "\n".join(f"- [{item['title']}]({item['source_url']})" for item in research)
        return f"### Retrieved research evidence\n\n{sources}\n\nAI explanation is currently unavailable.", "retrieved_research"
    return "### AI Explanation\n\nAI explanation is currently unavailable. Please try again when Gemini or Ollama is available.", "unavailable"


def _mixed_answer(question: str, frame: pd.DataFrame, dataset: Dataset, memory: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Support verified experiment-metric versus target-statistic comparisons only."""
    final = (memory.get("evaluation") or {}).get("final_test_metrics") or {}
    lower = question.lower()
    metrics = [name for name, aliases in _METRIC_ALIASES.items() if name in final and any(alias in lower for alias in aliases)]
    target = memory.get("target_column")
    if not metrics or not target or target not in frame.columns:
        return None
    requested_statistics = []
    for match in re.finditer(r"\b(average|mean|median|sum|total)\s+(?:of\s+)?([a-z0-9 _-]+?)(?=\s+(?:and|with|versus|vs|against|compared)|[,?]|$)", lower):
        column = _column(frame, match.group(2).strip())
        if column and pd.api.types.is_numeric_dtype(frame[column]):
            aggregation = Aggregation.MEDIAN if match.group(1) == "median" else Aggregation.SUM if match.group(1) in {"sum", "total"} else Aggregation.MEAN
            requested_statistics.append((column, aggregation))
    if not requested_statistics:
        requested_statistics = [(target, Aggregation.MEDIAN if "median" in lower else Aggregation.MEAN)]
    dataset_statistics = []
    for column, statistic in requested_statistics:
        plan = DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.AGGREGATE, column=column, aggregation=statistic)])
        dataset_result = execute_analytics_plan(frame, plan, dataset.original_filename)
        if not isinstance(dataset_result.value, (int, float)):
            return None
        dataset_statistics.append((column, statistic, dataset_result))
    target, statistic = dataset_statistics[0][0], dataset_statistics[0][1]
    baseline = dataset_statistics[0][2].value
    metric_values = {metric: final.get(metric) for metric in metrics}
    if any(not isinstance(value, (int, float)) for value in metric_values.values()) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    if len(metrics) > 1 or len(dataset_statistics) > 1:
        answer = "### Verified dataset + experiment comparison\n\n"
        answer += "\n".join(f"- **Final test {_metric_label(metric)} (experiment):** **{_format_metric(metric, float(metric_values[metric]))}**" for metric in metrics)
        answer += "\n" + "\n".join(f"- **{statistic.value.title()} {column} (dataset):** **{float(result.value):,.3f}**" for column, statistic, result in dataset_statistics)
        answer += "\n\nVerified persisted experiment metrics were compared with trusted dataset statistics."
        return answer, {
            "answer_type": "mixed",
            "title": f"Experiment metrics compared with {statistic.value} {target}",
            "experiment_metrics": [{"name": metric, "value": metric_values[metric]} for metric in metrics],
            "dataset_statistics": [{"column": column, "aggregation": statistic.value, "value": result.value} for column, statistic, result in dataset_statistics],
            "calculation": dataset_result.calculation,
        }
    metric = metrics[0]
    metric_value = metric_values[metric]
    ratio = abs(float(metric_value)) / abs(float(baseline))
    statistic_label = statistic.value.title()
    answer = (
        "### Verified dataset + experiment comparison\n\n"
        f"- **Final test {_metric_label(metric)} (experiment):** **{_format_metric(metric, float(metric_value))}**\n"
        f"- **{statistic_label} {target} (dataset):** **{float(baseline):,.3f}**\n"
        f"- **Relative size:** the {_metric_label(metric)} is **{ratio * 100:,.2f}%** of the dataset {statistic.value} target value.\n\n"
        "### How calculated\n\n"
        f"Read the persisted final-test {_metric_label(metric)} from the selected experiment, then calculated the {statistic.value} of **{target}** from the linked dataset using the trusted local analytics executor."
    )
    return answer, {
        "answer_type": "mixed",
        "title": f"{_metric_label(metric)} compared with {statistic.value} {target}",
        "experiment_metric": {"name": metric, "value": metric_value},
        "dataset_statistic": {"column": target, "aggregation": statistic.value, "value": baseline},
        "relative_ratio": ratio,
        "calculation": dataset_result.calculation,
    }


def _dataset_response(
    request: AssistantQueryRequest,
    question: str,
    dataset: Dataset,
    frame: pd.DataFrame,
    providers: list[LLMProvider],
    used_conversation: bool,
) -> tuple[AssistantQueryResponse, list[str], str]:
    plan = plan_open_ended_dataset_question(question, frame, providers)
    if plan is None:
        response = AssistantQueryResponse(
            answer=UNSUPPORTED_QUESTION,
            provider_used="trusted_analytics",
            sources=[AssistantSource(dataset_id=dataset.id, document_type="verified_dataset_calculation")],
            evidence_type="conversation" if used_conversation else "unsupported",
            provenance_label="Conversation context used; no safe dataset operation" if used_conversation else "No verified dataset evidence",
            conversation_id=request.conversation_id,
        )
        return response, [], "dataset analysis"
    result = enrich_verified_dataset_result(execute_analytics_plan(frame, plan, dataset.original_filename), plan, frame)
    payload = result.model_dump(mode="json")
    payload["plan"] = plan.model_dump(mode="json")
    if used_conversation:
        payload["conversation_context_used"] = True
    response = AssistantQueryResponse(
        answer=format_verified_dataset_answer(result, plan),
        provider_used="trusted_analytics",
        sources=[
            AssistantSource(dataset_id=dataset.id, document_type="verified_dataset_calculation"),
            *([AssistantSource(dataset_id=dataset.id, document_type="conversation_context")] if used_conversation else []),
        ],
        structured_result=payload,
        evidence_type="conversation" if used_conversation else "dataset",
        provenance_label="Verified from dataset · conversation context" if used_conversation else "Verified from dataset",
        conversation_id=request.conversation_id,
    )
    columns = [step.column for step in plan.steps if step.column] + [name for step in plan.steps for name in step.group_by] + [item.column for step in plan.steps for item in step.filters]
    return response, list(dict.fromkeys(columns)), "dataset analysis"


def answer_context_aware_question(
    request: AssistantQueryRequest,
    database: Session,
    user: User,
    settings: Settings,
    providers: list[LLMProvider] | None = None,
    experiment_answerer: Any = None,
) -> AssistantQueryResponse:
    providers = _providers(settings, providers)
    turns = recent_turns(database, user.id, request.conversation_id, settings.assistant_max_turns)
    remembered_dataset_id, remembered_experiment_id = remembered_context(turns)
    dataset_id = request.dataset_id or remembered_dataset_id
    experiment_id = request.experiment_id or remembered_experiment_id
    experiment = _owned_experiment(database, user, experiment_id)
    dataset = get_dataset_metadata(dataset_id, database, settings) if dataset_id else None
    if dataset and dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    if experiment and dataset and experiment.dataset_id != dataset.id:
        return _ambiguous_response(request, "The selected dataset and experiment are not linked. Select a matching context before asking a combined question.", dataset.id, experiment.id)
    if experiment and not dataset:
        dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    frame = load_dataset_dataframe(dataset, settings) if dataset else None
    resolved_question, is_follow_up = resolve_follow_up(request.question, turns)
    decision, router_provider = route_question(
        resolved_question,
        frame.columns if frame is not None else (),
        providers,
        conversation_context(turns, settings.assistant_max_context_chars // 4) if turns else "",
    )
    if request.context_type == "experiment" and decision.evidence_type == "unsupported":
        decision = EvidenceRoutingDecision(evidence_type="experiment", needs_experiment=True, reason="The selected experiment context is an explicit routing hint.")
    elif request.experiment_id and decision.evidence_type == "unsupported":
        decision = EvidenceRoutingDecision(evidence_type="experiment", needs_experiment=True, reason="The selected experiment is an explicit routing hint.")
    elif request.context_type == "dataset" and decision.evidence_type == "unsupported":
        decision = EvidenceRoutingDecision(evidence_type="dataset", needs_dataset=True, reason="The selected dataset context is an explicit routing hint.")
    used_conversation = bool(decision.uses_conversation or is_follow_up)

    if decision.needs_experiment and experiment is None and dataset is not None:
        candidates = _linked_experiments(database, user, dataset.id)
        if len(candidates) == 1:
            experiment = candidates[0]
        elif len(candidates) > 1:
            response = _ambiguous_response(request, "More than one saved experiment is linked to this dataset. Select the experiment you want to use.", dataset.id, None)
            save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id, experiment_id=None, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type="ambiguous", source_types=["context_selection"])
            return response
    if decision.needs_dataset and dataset is None and experiment is not None:
        dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
        frame = load_dataset_dataframe(dataset, settings)

    if decision.evidence_type in {"general", "research"}:
        research = retrieve_research(resolved_question) if decision.evidence_type == "research" else []
        answer, provider = _general_answer(resolved_question, providers, research)
        response = AssistantQueryResponse(
            answer=answer,
            provider_used=provider,
            sources=[AssistantSource(document_type="curated_research") for _ in research],
            evidence_type="research" if research else "general",
            provenance_label="Retrieved research evidence" if research else "AI Explanation",
            conversation_id=request.conversation_id,
        )
        save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id if dataset else None, experiment_id=experiment.id if experiment else None, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type=response.evidence_type, source_types=[source.document_type for source in response.sources])
        return response

    if decision.needs_dataset and dataset is None:
        response = _ambiguous_response(request, "Select a dataset before asking a dataset question.", None, experiment.id if experiment else None)
        return response
    if decision.needs_experiment and experiment is None:
        response = _ambiguous_response(request, "Select an evaluated experiment before asking an experiment question.", dataset.id if dataset else None, None)
        return response

    if decision.evidence_type == "dataset_experiment":
        assert dataset is not None and frame is not None and experiment is not None
        memories = retrieve_verified_memory(database, user.id, settings, experiment.id)
        mixed = _mixed_answer(resolved_question, frame, dataset, memories[0]) if memories else None
        if mixed is None:
            response = AssistantQueryResponse(answer="I can't answer that reliably from the selected dataset and experiment with the currently supported mixed analytical operations.", provider_used="trusted_analytics", sources=[AssistantSource(dataset_id=dataset.id, document_type="verified_dataset_calculation"), AssistantSource(experiment_id=experiment.id, document_type="verified_experiment_summary")], evidence_type="unsupported", provenance_label="Verified evidence required", conversation_id=request.conversation_id)
            columns: list[str] = []
        else:
            answer, structured = mixed
            response = AssistantQueryResponse(answer=answer, provider_used="trusted_analytics", sources=[AssistantSource(dataset_id=dataset.id, document_type="verified_dataset_calculation"), AssistantSource(experiment_id=experiment.id, document_type="verified_experiment_summary")], structured_result=structured, evidence_type="dataset_experiment", provenance_label="Verified from dataset + experiment", conversation_id=request.conversation_id)
            statistic = structured.get("dataset_statistic")
            statistics = structured.get("dataset_statistics", [])
            columns = ([statistic["column"]] if statistic else []) + [item["column"] for item in statistics]
        save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id, experiment_id=experiment.id, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type=response.evidence_type, referenced_columns=columns, analytical_topic="mixed comparison", source_types=[source.document_type for source in response.sources])
        return response

    if decision.needs_dataset:
        assert dataset is not None and frame is not None
        response, columns, topic = _dataset_response(request, resolved_question, dataset, frame, providers, used_conversation)
        save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id, experiment_id=experiment.id if experiment else None, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type=response.evidence_type, referenced_columns=columns, analytical_topic=topic, source_types=[source.document_type for source in response.sources])
        return response

    if decision.needs_experiment:
        assert experiment is not None
        memories = retrieve_verified_memory(database, user.id, settings, experiment.id)
        similar = retrieve_similar_experiments(database, user.id, memories[0], settings.assistant_max_experiments) if memories else []
        # Existing service keeps numerical statements deterministic even if an LLM is available.
        primary = providers[0] if providers else _UnavailableProvider("gemini")
        fallback = providers[1] if len(providers) > 1 else _UnavailableProvider("ollama")
        result = (experiment_answerer or answer_question)(resolved_question, memories, primary, fallback, settings.assistant_max_context_chars, similar)
        sources = [AssistantSource(experiment_id=item["experiment_id"], document_type="verified_experiment_summary") for item in memories]
        sources.extend(AssistantSource(experiment_id=item["experiment_id"], document_type="similar_experiment_context") for item in similar)
        if used_conversation:
            sources.append(AssistantSource(experiment_id=experiment.id, document_type="conversation_context"))
        response = AssistantQueryResponse(answer=result.answer, provider_used=result.provider_used, sources=sources, evidence_type="conversation" if used_conversation else "experiment", provenance_label="Verified from experiment · conversation context" if used_conversation else "Verified from experiment", conversation_id=request.conversation_id)
        save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id if dataset else None, experiment_id=experiment.id, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type=response.evidence_type, analytical_topic="experiment retrieval", source_types=[source.document_type for source in sources])
        return response

    response = AssistantQueryResponse(answer=UNSUPPORTED_QUESTION, provider_used=router_provider, sources=[], evidence_type="unsupported", provenance_label="No verified evidence route", conversation_id=request.conversation_id)
    save_turn(database, user_id=user.id, conversation_id=request.conversation_id, dataset_id=dataset.id if dataset else None, experiment_id=experiment.id if experiment else None, question=request.question, resolved_question=resolved_question, answer=response.answer, evidence_type="unsupported", source_types=[])
    return response


class _UnavailableProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate_text(self, _prompt: str) -> str:
        raise ProviderFailure(self.name, "unavailable", "Provider is unavailable.")
