from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.exceptions import DatasetValidationError
from app.schemas.analytics import AnalyticsFilter, AnalyticsQuery, AnalyticsOperation, FilterOperator
from app.schemas.assistant import AssistantQueryRequest, EvidenceRoutingDecision
from app.services.analytics_service import execute_analytics, execute_analytics_plan
from app.services.assistant_conversation_service import resolve_follow_up
from app.services.assistant_service import _deduplicate, answer_question
import pandas as pd
from app.services.question_planner_service import plan_open_ended_dataset_question
from app.services.assistant_orchestration_service import _general_answer, _mixed_answer


def test_routing_contract_rejects_contradictory_evidence_flags() -> None:
    with pytest.raises(ValidationError):
        EvidenceRoutingDecision(evidence_type="dataset", needs_experiment=True, reason="invalid")


def test_explicit_context_requires_matching_identifier() -> None:
    with pytest.raises(ValidationError):
        AssistantQueryRequest(question="Show the result", context_type="experiment", dataset_id="dataset-only")


def test_unknown_categorical_filter_is_rejected_before_empty_execution() -> None:
    frame = pd.DataFrame({"group": ["alpha", "beta"], "value": [1, 2]})
    query = AnalyticsQuery(
        operation=AnalyticsOperation.ROW_COUNT,
        filters=[AnalyticsFilter(column="group", operator=FilterOperator.EQ, value="unseen")],
    )
    with pytest.raises(DatasetValidationError):
        execute_analytics(frame, query, "fixture.csv")


def test_multiple_aggregations_fail_safe_instead_of_dropping_one() -> None:
    frame = pd.DataFrame({"group": ["alpha", "beta"], "value": [1, 2]})
    plan = plan_open_ended_dataset_question("Show the average and median value by group.", frame, [])
    assert plan is not None
    result_step = next(step for step in plan.steps if step.operation.value == "group_aggregate")
    assert result_step.aggregations == ["mean", "median"]
    result = execute_analytics_plan(frame, plan, "fixture.csv")
    assert [metric.name for metric in result.metric_results] == ["mean", "median"]


def test_follow_up_reuses_intent_but_not_prior_numeric_answer() -> None:
    turn = SimpleNamespace(resolved_question="Which group has the highest average measure?", referenced_columns=["group", "measure"])
    resolved, is_follow_up = resolve_follow_up("And what about the median?", [turn])
    assert is_follow_up
    assert resolved.startswith(turn.resolved_question.rstrip(" ?"))
    assert "median" in resolved.lower()
    standalone, is_follow_up = resolve_follow_up("Compare group values", [turn])
    assert standalone == "Compare group values"
    assert not is_follow_up


def test_verified_experiment_answer_does_not_echo_provider_number() -> None:
    verified = 0.73
    memory = [{
        "experiment_id": "one",
        "dataset": {"filename": "fixture.csv", "rows": 4, "columns": 2},
        "objective": "predict outcome",
        "task_type": "binary_classification",
        "target_column": "outcome",
        "selected_model": "model_a",
        "selected_model_validation_metrics": {"accuracy": verified},
        "evaluation": {"primary_metric": "accuracy", "validation_comparison": [], "final_test_metrics": {}},
    }]

    class HallucinatingProvider:
        name = "test-provider"

        def generate_text(self, _prompt: str) -> str:
            return "The verified accuracy is 999%."

    answer = answer_question("Which model was selected?", memory, HallucinatingProvider(), HallucinatingProvider(), 1000).answer
    assert "999%" not in answer
    assert f"{verified * 100:.2f}%" in answer


def test_same_metadata_different_experiment_ids_are_not_collapsed() -> None:
    base = {
        "dataset": {"filename": "fixture.csv"},
        "objective": "predict outcome",
        "task_type": "binary_classification",
        "target_column": "outcome",
    }
    memories = [{**base, "experiment_id": "first"}, {**base, "experiment_id": "second"}]
    assert [item["experiment_id"] for item in _deduplicate(memories)] == ["first", "second"]


def test_mixed_answer_preserves_multiple_requested_persisted_metrics() -> None:
    frame = pd.DataFrame({"outcome": [1.0, 2.0, 3.0], "feature": [4.0, 5.0, 6.0]})
    memory = {"target_column": "outcome", "evaluation": {"final_test_metrics": {"accuracy": 0.8, "precision": 0.7}}}
    answer = _mixed_answer("Compare accuracy and precision with the average outcome.", frame, SimpleNamespace(original_filename="fixture.csv"), memory)
    assert answer is not None
    _, structured = answer
    assert [item["name"] for item in structured["experiment_metrics"]] == ["accuracy", "precision"]


def test_general_explanation_requires_structured_claim_basis() -> None:
    class Provider:
        name = "test-provider"

        def generate_structured(self, _prompt, _schema):
            return {"explanation": "RMSE summarizes the typical size of prediction errors.", "claims": [{"claim": "This is general guidance.", "basis": "general_knowledge"}]}

    answer, provider = _general_answer("What does RMSE mean?", [Provider()])
    assert provider == "test-provider"
    assert "AI Explanation" in answer and "RMSE" in answer
