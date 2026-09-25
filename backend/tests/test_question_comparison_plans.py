"""Regression coverage for bounded grouped comparisons and top-group questions."""
import json

import pandas as pd
import pytest

from app.schemas.analytics import PlanStepOperation
from app.services.analytics_service import execute_analytics_plan
from app.services import question_planner_service as planner


@pytest.mark.parametrize("question", [
    "Which Country has the highest average Revenue and compare with others?",
    "Which Country has the highest average Revenue? Include a comparison with other groups.",
    "Compare average Revenue by Country.",
])
def test_numeric_group_comparisons_retain_every_group(question: str) -> None:
    frame = pd.DataFrame({"Country": ["A", "A", "B", "B", "C", "C"], "Revenue": [10, 30, 80, 100, 30, 50]})
    plan = planner.plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    result = execute_analytics_plan(frame, plan, "revenue.csv")
    assert [(row.label, row.value) for row in result.rows] == [("B", 90.0), ("C", 40.0), ("A", 20.0)]
    assert result.chart.labels == ["B", "C", "A"]
    assert result.chart.values == [90.0, 40.0, 20.0]


@pytest.mark.parametrize(("extreme", "expected"), [("highest", "Z"), ("lowest", "A")])
def test_only_extreme_keeps_all_groups_with_correct_group_first(extreme: str, expected: str) -> None:
    # Alphabetical category order must not discard the true winner before ranking.
    frame = pd.DataFrame({"Country": list("ABCDEFGHIJKZ"), "Revenue": list(range(12))})
    question = f"Which Country has the {extreme} average Revenue?"
    plan = planner.plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    result = execute_analytics_plan(frame, plan, "revenue.csv")
    assert len(result.rows) == len(frame)
    assert result.rows[0].label == expected


def test_comparison_preserves_more_than_ten_groups_with_true_highest_first() -> None:
    frame = pd.DataFrame({"Country": [f"Group {index:02}" for index in range(15)], "Revenue": list(range(15))})
    plan = planner.plan_open_ended_dataset_question("Which Country has the highest mean Revenue and compare with others?", frame, [])
    assert plan is not None
    result = execute_analytics_plan(frame, plan, "revenue.csv")
    assert len(result.rows) == 15
    assert result.rows[0].label == "Group 14"
    assert result.rows[-1].value == 0


@pytest.mark.parametrize("group_name", ["PlanType", "Region", "CustomerSegment"])
def test_churn_comparison_uses_the_named_group_generically(group_name: str) -> None:
    frame = pd.DataFrame({group_name: ["A", "A", "B", "B", "C", "C"], "Churn": ["Yes", "No", "Yes", "Yes", "No", "No"]})
    question = f"Which {group_name} has the highest churn rate and compare with others?"
    plan = planner.plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    grouped = next(step for step in plan.steps if step.operation == PlanStepOperation.GROUP_AGGREGATE)
    assert grouped.group_by == [group_name]
    assert grouped.positive_value == "Yes"
    result = execute_analytics_plan(frame, plan, "customers.csv")
    assert [(row.label, row.value) for row in result.rows] == [("B", 1.0), ("A", 0.5), ("C", 0.0)]


def test_plural_group_and_multiple_filters_preserve_comparison_rows() -> None:
    frame = pd.DataFrame({
        "PlanType": ["A", "A", "A", "B", "B", "C"],
        "SupportCalls": [3, 3, 0, 4, 4, 5],
        "SatisfactionScore": [2, 2, 1, 1, 4, 4],
        "Churn": [1, 0, 1, 1, 0, 1],
    })
    question = "Among customers with more than 2 support calls and satisfaction score below 3, compare churn rates across plan types."
    plan = planner.plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    assert len(plan.steps[0].filters) == 2
    result = execute_analytics_plan(frame, plan, "customers.csv")
    assert [(row.label, row.value) for row in result.rows] == [("B", 1.0), ("A", 0.5)]


def test_provider_top_one_is_normalized_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    class TopOneProvider:
        calls = 0

        def generate_structured(self, prompt: str, schema: dict) -> str:
            self.calls += 1
            return json.dumps({"steps": [
                {"operation": "group_aggregate", "column": "Revenue", "group_by": ["Country"], "aggregation": "mean", "limit": 1},
                {"operation": "sort_limit", "sort": "descending", "limit": 1},
                {"operation": "sort_limit", "sort": "descending", "limit": 1},
            ]})

    # Exercise the provider path independently of the offline parser's vocabulary.
    monkeypatch.setattr(planner, "_fallback", lambda question, frame: None)
    frame = pd.DataFrame({"Country": ["A", "B", "C"], "Revenue": [20, 90, 40]})
    provider = TopOneProvider()
    plan = planner.plan_open_ended_dataset_question("Which Country has the highest average Revenue and compare with others?", frame, [provider])
    assert plan is not None
    assert provider.calls == 1
    result = execute_analytics_plan(frame, plan, "revenue.csv")
    assert [(row.label, row.value) for row in result.rows] == [("B", 90.0), ("C", 40.0), ("A", 20.0)]


def test_comparison_remains_bounded_and_sorts_before_truncation() -> None:
    frame = pd.DataFrame({"Country": [f"Group {index:03}" for index in range(120)], "Revenue": list(range(120))})
    plan = planner.plan_open_ended_dataset_question("Compare mean Revenue by Country, including the highest.", frame, [])
    assert plan is not None
    assert all(step.limit <= 100 for step in plan.steps)
    result = execute_analytics_plan(frame, plan, "revenue.csv")
    assert len(result.rows) == 100
    assert result.rows[0].label == "Group 119"
    assert result.rows[-1].value == 20
