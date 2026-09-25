import pandas as pd
import pytest

from app.schemas.analytics import Aggregation, AnalyticsFilter, AnalyticsPlanStep, DatasetAnalyticsPlan, FilterOperator, PlanStepOperation
from app.services.analytics_service import execute_analytics_plan
from app.services.dataset_answer_presentation_service import enrich_verified_dataset_result
from app.services.question_planner_service import _semantic_validate_plan, plan_open_ended_dataset_question


def _plan(question: str, frame: pd.DataFrame) -> DatasetAnalyticsPlan:
    plan = plan_open_ended_dataset_question(question, frame, [])
    assert plan is not None
    return plan


def _result(question: str, frame: pd.DataFrame):
    plan = _plan(question, frame)
    return plan, enrich_verified_dataset_result(execute_analytics_plan(frame, plan, "fixture.csv"), plan, frame)


def test_highest_average_numeric_by_categorical_group() -> None:
    frame = pd.DataFrame({"region": ["north", "north", "south", "south"], "spend": [10, 30, 20, 40]})
    plan, result = _result("Which region has the highest average spend?", frame)
    grouped = frame.groupby("region")["spend"].mean()
    assert plan.steps[-2].group_by == ["region"]
    assert plan.steps[-2].aggregation == Aggregation.MEAN
    assert result.rows[0].label == str(grouped.idxmax())
    assert result.rows[0].value == grouped.max()
    assert {row.label for row in result.rows} == {str(label) for label in grouped.index}


def test_lowest_median_numeric_by_group() -> None:
    frame = pd.DataFrame({"segment": ["a", "a", "b", "b"], "amount": [1, 9, 4, 5]})
    plan, result = _result("Which segment has the lowest median amount?", frame)
    grouped = frame.groupby("segment")["amount"].median()
    assert plan.steps[-2].group_by == ["segment"]
    assert plan.steps[-2].aggregation == Aggregation.MEDIAN
    assert result.rows[0].label == str(grouped.idxmin())


def test_binary_rate_by_categorical_group() -> None:
    frame = pd.DataFrame({"channel": ["web", "web", "store", "store"], "converted": ["Yes", "No", "Yes", "Yes"]})
    plan, result = _result("Which channel has the highest conversion rate?", frame)
    expected = frame["converted"].eq("Yes").groupby(frame["channel"]).mean()
    assert plan.steps[-2].group_by == ["channel"]
    assert plan.steps[-2].positive_value == "Yes"
    assert result.rows[0].label == str(expected.idxmax())


def test_count_by_group() -> None:
    frame = pd.DataFrame({"order_id": [1, 2, 3], "channel": ["web", "web", "store"]})
    plan, result = _result("How many orders by channel?", frame)
    result_step = next(step for step in plan.steps if step.operation == PlanStepOperation.GROUP_AGGREGATE)
    assert result_step.group_by == ["channel"]
    assert result_step.aggregation == Aggregation.COUNT
    assert {row.label: row.value for row in result.rows} == frame.groupby("channel").size().to_dict()


def test_multiple_filters_mean_group_and_ranking() -> None:
    frame = pd.DataFrame({
        "region": ["north", "north", "south", "south"],
        "status": ["active", "inactive", "active", "active"],
        "score": [2, 1, 2, 4],
        "value": [10, 99, 20, 30],
    })
    plan, result = _result("Which region has the highest average value among active customers with score below 3?", frame)
    filtered = frame[(frame["status"] == "active") & (frame["score"] < 3)]
    expected = filtered.groupby("region")["value"].mean()
    filters = plan.steps[0].filters
    assert {(item.column, item.operator) for item in filters} == {("status", FilterOperator.EQ), ("score", FilterOperator.LT)}
    assert plan.steps[-2].group_by == ["region"]
    assert result.rows[0].label == str(expected.idxmax())
    assert "status = active" in result.calculation
    assert "score < 3.0" in result.calculation


def test_explicit_group_replaces_identifier_dimension() -> None:
    frame = pd.DataFrame({"customer_id": [101, 102, 103, 104], "region": ["north", "north", "south", "south"], "spend": [1, 3, 2, 4]})
    invalid = DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.GROUP_AGGREGATE, column="spend", group_by=["customer_id"], aggregation=Aggregation.MEAN)])
    repaired = _semantic_validate_plan("Which region has the highest average spend?", frame, invalid)
    assert repaired is not None
    assert repaired.steps[0].group_by == ["region"]


def test_continuous_numeric_column_cannot_become_binary_rate() -> None:
    frame = pd.DataFrame({"region": ["north", "south"], "spend": [10.0, 20.0]})
    assert plan_open_ended_dataset_question("Which region has the highest spend rate?", frame, []) is None
    invalid = DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.GROUP_AGGREGATE, column="spend", group_by=["region"], aggregation=Aggregation.MEAN, positive_value="Yes")])
    assert _semantic_validate_plan("Which region has the highest spend rate?", frame, invalid) is None


@pytest.mark.parametrize(
    ("frame", "question", "group", "measure"),
    [
        (pd.DataFrame({"geography": ["east", "west"], "cost": [4, 8]}), "Which geography has the highest average cost?", "geography", "cost"),
        (pd.DataFrame({"team": ["red", "blue"], "revenue": [7, 3]}), "Which team has the highest average revenue?", "team", "revenue"),
    ],
)
def test_same_semantics_work_with_different_schema_names(frame: pd.DataFrame, question: str, group: str, measure: str) -> None:
    plan = _plan(question, frame)
    result_step = next(step for step in plan.steps if step.operation == PlanStepOperation.GROUP_AGGREGATE)
    assert result_step.group_by == [group]
    assert result_step.column == measure


def test_numeric_measure_followups_do_not_suggest_rates() -> None:
    frame = pd.DataFrame({"region": ["north", "south"], "spend": [10, 20]})
    _, result = _result("Which region has the highest average spend?", frame)
    assert all("rate" not in question.lower() for question in result.follow_up_questions)


def test_missing_categorical_group_key_is_excluded_and_reported() -> None:
    frame = pd.DataFrame({"region": ["north", None, "south"], "value": [10, 100, 20]})
    _, result = _result("Which region has the highest average value?", frame)
    assert {row.label for row in result.rows} == {"north", "south"}
    assert all(row.label.lower() not in {"nan", "none", "null"} for row in result.rows)
    assert result.rows_matched == 3
    assert result.rows_excluded_missing_group == 1
    assert result.rows_used_for_grouped_analysis == 2
    assert "Rows excluded because region was missing: 1" in result.calculation


def test_missing_numeric_aggregation_value_is_excluded_and_reported() -> None:
    frame = pd.DataFrame({"region": ["north", "north", "south"], "value": [10, None, 30]})
    _, result = _result("Which region has the highest average value?", frame)
    assert {row.label: row.value for row in result.rows} == {"north": 10.0, "south": 30.0}
    assert result.rows_excluded_missing_value == 1
    assert result.rows_used_for_grouped_analysis == 2


def test_multiple_group_keys_exclude_any_row_with_missing_key() -> None:
    frame = pd.DataFrame({"region": ["north", "north", None, "south"], "segment": ["a", None, "b", "b"], "value": [10, 20, 90, 30]})
    _, result = _result("Compare average value by region and segment.", frame)
    assert {row.label for row in result.rows} == {"north / a", "south / b"}
    assert result.rows_excluded_missing_group == 2
    assert result.rows_used_for_grouped_analysis == 2


def test_explicit_missing_value_analysis_uses_missing_label() -> None:
    frame = pd.DataFrame({"region": ["north", None, "south"], "value": [10, 100, 20]})
    plan, result = _result("Compare average value by region including missing values.", frame)
    grouped = next(step for step in plan.steps if step.operation == PlanStepOperation.GROUP_AGGREGATE)
    assert grouped.include_missing
    assert "Missing" in {row.label for row in result.rows}
    assert all(row.label.lower() not in {"nan", "none", "null"} for row in result.rows)
    assert result.rows_excluded_missing_group == 0


def test_ranking_never_selects_excluded_missing_group() -> None:
    frame = pd.DataFrame({"region": ["north", None, "south"], "value": [10, 999, 20]})
    _, result = _result("Which region has the highest average value?", frame)
    assert result.rows[0].label == "south"
    assert result.rows[0].label != "Missing"