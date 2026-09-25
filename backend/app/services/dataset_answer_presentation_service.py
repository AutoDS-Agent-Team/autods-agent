"""Deterministic presentation metadata derived only from verified analytics results."""
import re
from typing import Any

import pandas as pd

from app.schemas.analytics import (
    Aggregation,
    AnalyticsPlanStep,
    AnalyticsQueryResponse,
    DatasetAnalyticsPlan,
    PlanStepOperation,
)


def _label(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).replace("_", " ").replace("-", " ")
    return " ".join(spaced.split())


def _number(value: Any, as_rate: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if as_rate:
        rendered = f"{float(value) * 100:,.2f}".rstrip("0").rstrip(".")
        return f"{rendered}%"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.3f}".rstrip("0").rstrip(".")


def _result_step(plan: DatasetAnalyticsPlan) -> AnalyticsPlanStep | None:
    return next(
        (step for step in plan.steps if step.operation not in {PlanStepOperation.FILTER, PlanStepOperation.SORT_LIMIT}),
        None,
    )


def _is_rate(step: AnalyticsPlanStep | None, frame: pd.DataFrame) -> bool:
    if not step or step.aggregation != Aggregation.MEAN or not step.column:
        return False
    if step.positive_value is not None:
        return True
    values = set(pd.to_numeric(frame[step.column].dropna(), errors="coerce").dropna().unique())
    return len(values) == 2 and values == {0, 1}


def _column_kind(step: AnalyticsPlanStep | None, frame: pd.DataFrame, is_rate: bool) -> str | None:
    if not step or not step.column or step.column not in frame:
        return None
    if is_rate:
        return "categorical"
    series = frame[step.column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
        return "numerical"
    return "categorical"


def _comparison_insights(result: AnalyticsQueryResponse, metric_label: str, is_rate: bool) -> list[str]:
    numeric_rows = [row for row in result.rows if isinstance(row.value, (int, float)) and row.value is not None]
    if not numeric_rows:
        return []
    highest = max(numeric_rows, key=lambda row: float(row.value))
    lowest = min(numeric_rows, key=lambda row: float(row.value))
    insights = [f"{highest.label} has the highest {metric_label.lower()} at {_number(highest.value, is_rate)}."]
    # A bounded top-100 response cannot truthfully name the global minimum when
    # additional groups exist outside the displayed verified rows.
    if len(numeric_rows) == 1 or (result.total_groups is not None and result.total_groups > len(numeric_rows)):
        return insights
    insights.append(f"{lowest.label} has the lowest {metric_label.lower()} at {_number(lowest.value, is_rate)}.")
    gap = float(highest.value) - float(lowest.value)
    if is_rate:
        rendered_gap = f"{gap * 100:,.2f}".rstrip("0").rstrip(".")
        insights.append(f"The difference between {highest.label} and {lowest.label} is {rendered_gap} percentage points.")
        if float(lowest.value) > 0:
            insights.append(f"{highest.label}'s rate is {float(highest.value) / float(lowest.value):.2f}× {lowest.label}'s rate.")
    else:
        insights.append(f"The absolute difference between {highest.label} and {lowest.label} is {_number(gap)}.")
    return insights[:5]


def _follow_ups(step: AnalyticsPlanStep | None, column_kind: str | None, is_rate: bool) -> list[str]:
    if not step or not step.column:
        return ["Show the dataset distribution.", "Compare the top groups.", "Which features are associated?"]
    column = step.column
    if column_kind == "numerical":
        return [
            f"Show the distribution of {column}.",
            f"Are there outliers in {column}?",
            f"What are the mean and median of {column}?",
        ]
    group = step.group_by[0] if step.group_by else None
    questions = [f"Show the distribution of {column}.", f"Show counts for each {column} category."]
    if group:
        questions.append(f"Compare {column} rates across {group}." if is_rate else f"Compare {column} across {group}.")
    else:
        questions.append(f"Compare {column} rates across a categorical feature." if is_rate else f"Compare {column} across a categorical feature.")
    return questions


def enrich_verified_dataset_result(
    result: AnalyticsQueryResponse,
    plan: DatasetAnalyticsPlan,
    frame: pd.DataFrame,
) -> AnalyticsQueryResponse:
    """Attach display hints and insights without altering any verified raw value."""
    step = _result_step(plan)
    is_rate = _is_rate(step, frame)
    metric_label = _label(step.column) if step and step.column else result.title
    if is_rate:
        metric_label = f"{metric_label} rate"
    group_label = " / ".join(_label(name) for name in step.group_by) if step and step.group_by else None
    insights = (
        _comparison_insights(result, metric_label, is_rate)
        if step and step.operation == PlanStepOperation.GROUP_AGGREGATE
        else []
    )
    return result.model_copy(update={
        "value_format": "percent" if is_rate else "number",
        "metric_label": metric_label,
        "group_label": group_label,
        "insights": insights,
        "follow_up_questions": _follow_ups(step, _column_kind(step, frame, is_rate), is_rate),
    })


def _meaning(plan: DatasetAnalyticsPlan) -> str:
    result_step = _result_step(plan)
    if not result_step:
        return "A verified result from the selected dataset."
    if result_step.operation == PlanStepOperation.AGGREGATE:
        return f"The {result_step.aggregation.value} of valid **{result_step.column}** records."
    if result_step.operation == PlanStepOperation.GROUP_AGGREGATE:
        return f"A comparison of **{result_step.column}** across **{', '.join(result_step.group_by)}** groups."
    if result_step.operation == PlanStepOperation.DISTRIBUTION:
        return f"How verified values are distributed in **{result_step.column}**."
    if result_step.operation in {PlanStepOperation.CORRELATION, PlanStepOperation.ASSOCIATION}:
        return "A ranked statistical association, not evidence of causation."
    return "A verified result from the selected dataset."


def format_verified_dataset_answer(result: AnalyticsQueryResponse, plan: DatasetAnalyticsPlan) -> str:
    """Build concise Markdown only from the trusted executor response and validated plan."""
    is_rate = result.value_format == "percent"
    lines = [f"### {result.title}", ""]
    if result.answer_type == "scalar":
        lines.append(f"- **{result.title}:** **{_number(result.value, is_rate)}**")
    elif result.rows:
        first = result.rows[0]
        lines.append(f"- **Direct answer:** **{first.label} — {_number(first.value, is_rate)}**")
        for row in result.rows[:10]:
            lines.append(f"- **{row.label}:** **{_number(row.value, is_rate)}**")
    lines.extend([
        f"- **Rows Analyzed:** **{result.rows_analyzed:,}**",
        f"- **Meaning:** {_meaning(plan)}",
        f"- **How calculated:** {result.calculation}",
        f"- **Source:** **{result.source_filename}**",
        "- **Verified from dataset**",
    ])
    if result.rows_matched is not None:
        lines.insert(-4, f"- **Rows matched filters:** **{result.rows_matched:,}**")
    if result.rows_used_for_grouped_analysis is not None:
        lines.insert(-4, f"- **Rows used for grouped analysis:** **{result.rows_used_for_grouped_analysis:,}**")
    if result.rows_excluded_missing_group:
        lines.insert(-4, f"- **Rows excluded because a group key was missing:** **{result.rows_excluded_missing_group:,}**")
    if result.rows_excluded_missing_value:
        lines.insert(-4, f"- **Rows excluded because the aggregated value was missing:** **{result.rows_excluded_missing_value:,}**")
    if result.metric_results:
        lines.extend(["", "### Additional Metrics", ""])
        for metric in result.metric_results[1:]:
            if metric.rows:
                lines.append(f"- **{_label(metric.name)}:** " + "; ".join(f"{row.label} — {_number(row.value, is_rate)}" for row in metric.rows[:10]))
            else:
                lines.append(f"- **{_label(metric.name)}:** **{_number(metric.value, is_rate)}**")
    lines.extend(["", "### Suggested Analysis", ""])
    lines.extend(f"- {question}" for question in result.follow_up_questions)
    return "\n".join(lines)
