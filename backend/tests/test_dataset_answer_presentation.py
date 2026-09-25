"""Focused regression tests for trusted Ask AutoDS presentation metadata."""
import pandas as pd

from app.services.analytics_service import execute_analytics_plan
from app.services.dataset_answer_presentation_service import enrich_verified_dataset_result, format_verified_dataset_answer
from app.services.question_planner_service import plan_open_ended_dataset_question


def _answer(question: str, frame: pd.DataFrame):
    plan = plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    raw = execute_analytics_plan(frame, plan, "customers.csv")
    return enrich_verified_dataset_result(raw, plan, frame), plan


def test_rate_keeps_raw_value_and_formats_percentage() -> None:
    frame = pd.DataFrame({"Churn": ["Yes"] * 74 + ["No"] * 51})
    result, plan = _answer("What is the churn rate?", frame)

    assert result.value == 0.592
    assert result.value_format == "percent"
    assert "59.2%" in format_verified_dataset_answer(result, plan)


def test_group_comparison_has_verified_rate_insights_and_categorical_followups() -> None:
    frame = pd.DataFrame({
        "ContractType": ["Monthly"] * 4 + ["Annual"] * 4 + ["Two year"] * 4,
        "Churn": ["Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No"],
    })
    result, _ = _answer("Which contract type has the highest churn rate and compare with others?", frame)

    assert [(row.label, row.value) for row in result.rows] == [
        ("Monthly", 0.75), ("Annual", 0.5), ("Two year", 0.25),
    ]
    assert result.chart is not None and result.chart.values == [0.75, 0.5, 0.25]
    assert result.value_format == "percent"
    assert result.metric_label == "Churn rate"
    assert result.group_label == "Contract Type"
    assert any("50 percentage points" in insight for insight in result.insights)
    assert any("3.00×" in insight for insight in result.insights)
    assert all("outlier" not in question.lower() for question in result.follow_up_questions)


def test_numeric_followups_include_outliers_without_false_rate_format() -> None:
    frame = pd.DataFrame({"Department": ["A", "A", "B", "B"], "HourlyRate": [0.2, 0.4, 0.6, 0.8]})
    result, _ = _answer("Compare average HourlyRate by Department.", frame)

    assert result.value_format == "number"
    assert any("outliers" in question for question in result.follow_up_questions)
    assert all("%" not in insight for insight in result.insights)
