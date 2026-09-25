"""Bounded post-experiment interpretation; deterministic code remains authoritative."""
import re
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import TrainingNotReadyError
from app.llm.base import LLMProvider, ProviderFailure
from app.schemas.agents import DecisionExplanation, InsightFeature, InsightRecommendation, InsightReflectionResponse, ResearchCitation, ResearchEvidence, SimilarExperimentEvidence
from app.services.experiment_memory_service import bounded_context, retrieve_similar_experiments, retrieve_verified_memory
from app.services.explainability_service import get_explainability
from app.services.research_rag_service import retrieve_research
from app.services.dataset_service import get_dataset_profile


PERCENT_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


def _label(value: str | None) -> str:
    return (value or "Not recorded").replace("_", " ").title()


def _metric_label(value: str) -> str:
    return {"f1": "F1 Score", "roc_auc": "ROC-AUC"}.get(value, value.replace("_", " ").title())


def _metric(value: str, score: Any) -> str:
    number = float(score)
    return f"{number * 100:.2f}%" if value in PERCENT_METRICS else f"{number:.3f}"


def _is_higher_better(metric: str) -> bool:
    return metric not in {"mae", "rmse", "mse", "log_loss"}


def _verified_review(memory: dict[str, Any], features: list[str]) -> InsightReflectionResponse:
    evaluation = memory.get("evaluation") or {}
    final = evaluation.get("final_test_metrics") or {}
    selected, primary = memory.get("selected_model"), evaluation.get("primary_metric")
    validation = memory.get("selected_model_validation_metrics") or {}
    if not selected or not primary or validation.get(primary) is None:
        raise TrainingNotReadyError("Evaluate baseline models before requesting an AI review.")
    summary = f"{_label(selected)} is the selected baseline model for {_label(memory.get('task_type'))} predicting {memory.get('target_column') or 'the confirmed target'}."
    validation_score = float(validation[primary])
    assessment = f"Selection used validation {_metric_label(primary)} of {_metric(primary, validation_score)}."
    if final.get(primary) is not None:
        test_score = float(final[primary])
        difference = test_score - validation_score
        assessment += f" Final test {_metric_label(primary)} was {_metric(primary, test_score)}."
        if difference == 0:
            assessment += " Validation and test results were the same on this metric."
        else:
            direction = "higher" if difference > 0 else "lower"
            assessment += f" Test was {_metric(primary, abs(difference))} {direction} than validation."
    strengths, weaknesses = [], []
    for name, score in final.items():
        if score is None:
            continue
        formatted = f"Final test {_metric_label(name)}: {_metric(name, score)}."
        if name in PERCENT_METRICS and float(score) >= 0.75:
            strengths.append(formatted)
    if final.get("precision") is not None and final.get("recall") is not None and float(final["recall"]) + 0.05 < float(final["precision"]):
        weaknesses.append("Recall is weaker than precision, so some positive cases may be missed.")
    if final.get(primary) is not None:
        strengths.append(
            f"Selected by validation {_metric_label(primary)} and verified on the untouched test split."
        )
    if final.get(primary) is not None:
        test_score = float(final[primary])
        gap = abs(test_score - validation_score)
        # A comparison with the observed validation score gives a transparent,
        # data-grounded stability warning without asserting a quality threshold.
        if gap > max(abs(validation_score) * 0.1, 0.01):
            weaknesses.append(
                f"Validation-to-test {_metric_label(primary)} changed by {_metric(primary, gap)}; validate on another split or future data."
            )
    weaknesses.append("Results are based on one persisted train/validation/test split and may not generalize to new data.")
    important = [InsightFeature(feature=name, explanation="This feature is among the verified global importance results.") for name in features[:5]]
    recommendations = [
        InsightRecommendation(
            title="Validate on additional data",
            reason="The reported test result comes from one persisted split.",
            suggested_action="Evaluate the selected workflow on a future or external holdout before operational use.",
            requires_retraining=False,
        ),
        InsightRecommendation(
            title="Review the leading features",
            reason="Feature importance identifies the inputs most used by the selected model.",
            suggested_action="Check the listed important features for stability, leakage, and availability at prediction time.",
            requires_retraining=False,
        ),
    ]
    if weaknesses:
        recommendations.append(InsightRecommendation(title="Compare a supported alternative", reason="The observed validation-to-test difference should be checked before relying on the selected baseline.", suggested_action="Run a supported preprocessing or baseline-model experiment and compare the same primary metric.", requires_retraining=True))
    return InsightReflectionResponse(executive_summary=summary, model_assessment=assessment, strengths=strengths, weaknesses=weaknesses, important_features=important, recommendations=recommendations, provider_used="verified_reflection", status="FALLBACK")


def _safe_provider_recommendations(candidate: InsightReflectionResponse, allowed_features: set[str]) -> list[InsightRecommendation]:
    allowed_terms = ("class weight", "preprocessing", "metric", "baseline model")
    safe = []
    for item in candidate.recommendations:
        text = " ".join((item.title, item.reason, item.suggested_action)).lower()
        if not re.search(r"\d", text) and any(term in text for term in allowed_terms):
            safe.append(item)
    return safe


def _safe_provider_explanations(candidate: InsightReflectionResponse, verified: InsightReflectionResponse, allowed_features: set[str]) -> dict[str, Any]:
    """Keep provider prose only when it cannot introduce unverified facts.

    Deterministic evidence remains the leading statement.  Provider prose may
    add qualitative interpretation, but not numerical claims, code actions, or
    feature names outside the persisted explainability result.
    """
    prohibited = re.compile(r"\b(?:sql|shell|command|import|eval|exec|path|file system)\b", re.I)

    def safe_text(value: str) -> str | None:
        value = value.strip()
        if not value or re.search(r"\d", value) or prohibited.search(value):
            return None
        return value

    summary = safe_text(candidate.executive_summary)
    assessment = safe_text(candidate.model_assessment)
    strengths = [item for item in (safe_text(value) for value in candidate.strengths) if item]
    weaknesses = [item for item in (safe_text(value) for value in candidate.weaknesses) if item]
    important_features = [item for item in candidate.important_features if item.feature in allowed_features and safe_text(item.explanation)]
    return {
        "executive_summary": f"{verified.executive_summary} AI interpretation: {summary}" if summary else verified.executive_summary,
        "model_assessment": f"{verified.model_assessment} AI interpretation: {assessment}" if assessment else verified.model_assessment,
        "strengths": list(dict.fromkeys([*verified.strengths, *strengths])),
        "weaknesses": list(dict.fromkeys([*verified.weaknesses, *weaknesses])),
        "important_features": important_features or verified.important_features,
    }


def _link_research_to_recommendations(recommendations: list[InsightRecommendation]) -> tuple[list[InsightRecommendation], list[ResearchEvidence]]:
    linked, evidence_by_url = [], {}
    for recommendation in recommendations:
        query = " ".join((recommendation.title, recommendation.reason, recommendation.suggested_action))
        hits = retrieve_research(query, limit=1)
        citations = [ResearchCitation(title=item["title"], source_url=item["source_url"], year=item.get("year")) for item in hits]
        for item in hits:
            evidence_by_url[item["source_url"]] = ResearchEvidence.model_validate(item)
        linked.append(recommendation.model_copy(update={"research_sources": citations}))
    return linked, list(evidence_by_url.values())


def _decision_explanations(memory: dict[str, Any], profile: Any | None = None) -> list[dict[str, str]]:
    evaluation = memory.get("evaluation") or {}
    metric = evaluation.get("primary_metric")
    chosen = memory.get("selected_model")
    chosen_row = next((row for row in evaluation.get("validation_comparison", []) if row.get("model_name") == chosen), None)
    explanations = [
        {"stage": "Task and target", "decision": f"{_label(memory.get('task_type'))} · {memory.get('target_column') or 'target not recorded'}", "rationale": "These values were confirmed and persisted for this experiment; the planner was constrained to use them.", "evidence_source": "confirmed experiment configuration"}
    ]
    if metric and chosen_row and chosen_row.get("metrics", {}).get(metric) is not None:
        score = chosen_row["metrics"][metric]
        rows = [row for row in evaluation.get("validation_comparison", []) if row.get("metrics", {}).get(metric) is not None]
        best_score = (max if _is_higher_better(metric) else min)(float(row["metrics"][metric]) for row in rows) if rows else float(score)
        is_best = float(score) == best_score
        rationale = f"The persisted validation comparison ranked {len(rows)} candidate(s) by {_metric_label(metric)} ({'higher' if _is_higher_better(metric) else 'lower'} is better). The selected model scored {_metric(metric, score)}. "
        rationale += "It matched the best validation value." if is_best else "The selected result differs from the best recorded value; review the saved comparison."
        rationale += " Final test results were kept out of selection."
        explanations.append({"stage": "Baseline selection", "decision": f"{_label(chosen)} selected", "rationale": rationale, "evidence_source": "persisted validation comparison"})
    plan = memory.get("pipeline_plan") or {}
    if plan:
        prep = []
        if plan.get("numeric_imputation"):
            prep.append(f"numeric missing values: {_label(plan['numeric_imputation'])} imputation")
        if plan.get("categorical_imputation"):
            prep.append(f"categorical missing values: {_label(plan['categorical_imputation'])} imputation")
        if plan.get("categorical_encoding"):
            prep.append(f"categorical features: {_label(plan['categorical_encoding'])} encoding")
        if plan.get("numeric_scaling"):
            prep.append(f"numeric features: {_label(plan['numeric_scaling'])} scaling")
        if plan.get("text_columns") and plan.get("text_vectorization"):
            prep.append(f"text features: {_label(plan['text_vectorization'])} vectorization")
        if prep:
            explanations.append({"stage": "Preprocessing", "decision": "; ".join(prep), "rationale": "These are the persisted, allowlisted pipeline settings applied by trusted preprocessing code. They are configuration choices, not claims that research or the LLM changed model results.", "evidence_source": "persisted pipeline plan"})
        if plan.get("excluded_columns"):
            explanations.append({"stage": "Feature exclusions", "decision": ", ".join(plan["excluded_columns"]), "rationale": "The dataset profiler flagged these columns as identifier-like; the planner persisted them as excluded to reduce the risk of learning row identifiers.", "evidence_source": "persisted pipeline plan and dataset profile"})
        if plan.get("models"):
            explanations.append({"stage": "Candidate models", "decision": ", ".join(_label(item) for item in plan["models"][:5]), "rationale": f"The saved plan used these task-compatible baseline candidates and primary metric {_label(plan.get('primary_metric'))}; the final selection was made from persisted validation results.", "evidence_source": "persisted allowlisted pipeline plan"})
    if profile is not None:
        summary = profile.summary
        explanations.append({"stage": "Dataset quality", "decision": f"{summary.total_missing_values:,} missing values · {summary.duplicate_row_count:,} duplicate rows", "rationale": "These counts come from the deterministic profiler and are surfaced as review context; no records were changed by this explanation.", "evidence_source": "deterministic dataset profile"})
        target = memory.get("target_column")
        numeric_missing = sum(column.missing_count for column in profile.columns if column.logical_type == "numerical" and column.name != target)
        categorical_missing = sum(column.missing_count for column in profile.columns if column.logical_type != "numerical" and column.name != target)
        if numeric_missing or categorical_missing:
            explanations.append({"stage": "Missing-value preparation context", "decision": f"{numeric_missing:,} numeric and {categorical_missing:,} non-numeric predictor values missing", "rationale": "The pipeline's saved imputation settings correspond to missing values observed by the profiler among predictors; target-column missingness is excluded from these counts.", "evidence_source": "deterministic dataset profile and persisted pipeline plan"})
    return explanations


def create_reflection(experiment_id: str, user_id: str, database: Session, settings: Settings, providers: tuple[LLMProvider, LLMProvider]) -> InsightReflectionResponse:
    memories = retrieve_verified_memory(database, user_id, settings, experiment_id)
    if not memories or not (memories[0].get("evaluation") or {}).get("final_test_metrics"):
        raise TrainingNotReadyError("Evaluate baseline models before requesting an AI review.")
    memory = memories[0]
    profile = None
    dataset_id = (memory.get("dataset") or {}).get("id")
    if dataset_id:
        try:
            profile = get_dataset_profile(dataset_id, database, settings)
        except Exception:
            # Reviews and verified results remain available if the source file
            # is no longer present; profile-dependent explanations are omitted.
            pass
    features: list[str] = []
    try:
        explanation = get_explainability(experiment_id, database, settings)
        features = [item.feature_name for item in explanation.global_feature_importance]
    except TrainingNotReadyError:
        pass
    verified = _verified_review(memory, features)
    plan_context = memory.get("pipeline_plan") or {}
    query = " ".join(str(value) for value in (
        memory.get("objective"), memory.get("task_type"), memory.get("target_column"),
        memory.get("selected_model"), (memory.get("evaluation") or {}).get("primary_metric"),
        " ".join(plan_context.get("models", [])), plan_context.get("primary_metric"),
    ) if value)
    research = retrieve_research(query)
    try:
        similar = retrieve_similar_experiments(database, user_id, memory)
    except Exception:
        similar = []
    research_context = "\nADVISORY OFFLINE RESEARCH (may inform explanations/recommendations only; must not change metrics, results, or model selection):\n" + "\n".join(f"{item['title']} ({item.get('year')}, {item.get('venue')}): {item['evidence_summary']} Source: {item['source_url']}" for item in research) if research else "\nNo relevant curated research evidence was retrieved. Do not invent citations."
    similar_prompt = [{key: item.get(key) for key in ("dataset_filename", "objective", "task_type", "target_column", "selected_model", "primary_metric", "validation_score", "test_score", "match_reason")} for item in similar]
    prompt = ("You are the Insight and Reflection Agent. Return JSON only. Interpret this verified evidence without calculating or inventing metrics, model names, features, IDs, paths, citations, code, commands, or SQL. Recommendations are advisory only and may suggest only class weighting, supported preprocessing, an alternative supported metric, or an existing baseline model. Curated research and previous experiments are advisory only.\n" + bounded_context([memory], settings.assistant_max_context_chars) + f"\nVERIFIED_FEATURES={features[:10]}\nSIMILAR_PERSISTED_EXPERIMENTS={similar_prompt}" + research_context)
    deterministic_context = {
        "research_evidence": [ResearchEvidence.model_validate(item) for item in research],
        "research_status": "EVIDENCE_FOUND" if research else "NO_RELEVANT_EVIDENCE",
        "similar_experiments": [SimilarExperimentEvidence.model_validate(item) for item in similar],
        "decision_explanations": [DecisionExplanation.model_validate(item) for item in _decision_explanations(memory, profile)],
    }
    for provider in providers:
        try:
            raw = provider.generate_structured(prompt, InsightReflectionResponse.model_json_schema())
            candidate = InsightReflectionResponse.model_validate_json(raw) if isinstance(raw, str) else InsightReflectionResponse.model_validate(raw)
            recommendations = _safe_provider_recommendations(candidate, set(features))
            explanations = _safe_provider_explanations(candidate, verified, set(features))
            recommendations, recommendation_evidence = _link_research_to_recommendations(recommendations or verified.recommendations)
            evidence = {item.source_url: item for item in deterministic_context["research_evidence"]}
            evidence.update({item.source_url: item for item in recommendation_evidence})
            return verified.model_copy(update={**explanations, "recommendations": recommendations, **deterministic_context, "research_evidence": list(evidence.values()), "research_status": "EVIDENCE_FOUND" if evidence else "NO_RELEVANT_EVIDENCE", "provider_used": provider.name, "status": "COMPLETED"})
        except (ProviderFailure, ValidationError):
            continue
    recommendations, recommendation_evidence = _link_research_to_recommendations(verified.recommendations)
    evidence = {item.source_url: item for item in deterministic_context["research_evidence"]}
    evidence.update({item.source_url: item for item in recommendation_evidence})
    return verified.model_copy(update={**deterministic_context, "research_evidence": list(evidence.values()), "research_status": "EVIDENCE_FOUND" if evidence else "NO_RELEVANT_EVIDENCE", "recommendations": recommendations})
