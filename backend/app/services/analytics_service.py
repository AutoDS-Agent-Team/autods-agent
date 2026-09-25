import math
from typing import Any

import pandas as pd

from app.core.exceptions import DatasetValidationError
from app.schemas.analytics import (
    Aggregation,
    AnalyticsPlanStep,
    DatasetAnalyticsPlan,
    AnalyticsOperation,
    AnalyticsQuery,
    AnalyticsQueryResponse,
    AnalyticsMetricResult,
    AnalyticsResultRow,
    ChartSpec,
    FilterOperator,
    PlanStepOperation,
)


def _safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _display_group_part(value: Any, include_missing: bool) -> str:
    if pd.isna(value):
        return "Missing" if include_missing else ""
    return str(value)


def _require_columns(frame: pd.DataFrame, query: AnalyticsQuery) -> None:
    requested = [query.column, *query.group_by, *(item.column for item in query.filters)]
    missing = [name for name in requested if name and name not in frame.columns]
    if missing:
        raise DatasetValidationError(f"Unknown dataset column: {missing[0]}.")


def _coerce_filter_value(series: pd.Series, value: Any) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        try:
            if isinstance(value, list):
                return [float(item) for item in value]
            return float(value)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError("Filter value is not compatible with the numeric column.") from error
    values = series.dropna().unique().tolist()
    candidates = value if isinstance(value, list) else [value]
    unknown = [candidate for candidate in candidates if candidate not in values]
    if unknown:
        raise DatasetValidationError("Filter value is not present in the verified categorical column.")
    return value


def _filter_summary(filters: list[Any]) -> str:
    if not filters:
        return "none"
    operators = {FilterOperator.EQ: "=", FilterOperator.NEQ: "!=", FilterOperator.GT: ">", FilterOperator.GTE: ">=", FilterOperator.LT: "<", FilterOperator.LTE: "<=", FilterOperator.IN: "in", FilterOperator.NOT_IN: "not in"}
    return "; ".join(f"{item.column} {operators[item.operator]} {item.value}" for item in filters)


def _step_aggregations(step: AnalyticsPlanStep) -> list[Aggregation]:
    return step.aggregations or ([step.aggregation] if step.aggregation else [])


def _filtered(frame: pd.DataFrame, query: AnalyticsQuery) -> pd.DataFrame:
    result = frame
    for condition in query.filters:
        series = result[condition.column]
        value = _coerce_filter_value(series, condition.value)
        op = condition.operator
        if op in {FilterOperator.IN, FilterOperator.NOT_IN} and not isinstance(value, list):
            raise DatasetValidationError("The in/not_in operators require a list value.")
        mask = {
            FilterOperator.EQ: lambda: series == value,
            FilterOperator.NEQ: lambda: series != value,
            FilterOperator.GT: lambda: series > value,
            FilterOperator.GTE: lambda: series >= value,
            FilterOperator.LT: lambda: series < value,
            FilterOperator.LTE: lambda: series <= value,
            FilterOperator.IN: lambda: series.isin(value),
            FilterOperator.NOT_IN: lambda: ~series.isin(value),
        }[op]()
        result = result.loc[mask.fillna(False)]
    return result


def _aggregate(series: pd.Series, aggregation: Aggregation, quantile: float | None) -> Any:
    if aggregation == Aggregation.COUNT:
        return int(series.count())
    if aggregation == Aggregation.MODE:
        values = series.mode(dropna=True)
        return _safe(values.iloc[0]) if len(values) else None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        raise DatasetValidationError(f"{aggregation.value} requires a numerical column.")
    functions = {
        Aggregation.MEAN: numeric.mean,
        Aggregation.MEDIAN: numeric.median,
        Aggregation.MIN: numeric.min,
        Aggregation.MAX: numeric.max,
        Aggregation.SUM: numeric.sum,
        Aggregation.STD: numeric.std,
        Aggregation.VARIANCE: numeric.var,
        Aggregation.QUANTILE: lambda: numeric.quantile(0.5 if quantile is None else quantile),
    }
    return _safe(functions[aggregation]())


def _planned_aggregate(series: pd.Series, step: AnalyticsPlanStep) -> Any:
    if step.positive_value is not None:
        return _safe(float(series.eq(step.positive_value).mean()))
    return _aggregate(series, step.aggregation, None)


def _table(series: pd.Series, query: AnalyticsQuery, title: str, calculation: str, filename: str, rows: int, proportions: bool = False) -> AnalyticsQueryResponse:
    values = series.value_counts(dropna=False, normalize=proportions)
    values = values.sort_values(ascending=query.sort == "ascending") if query.sort else values
    values = values.head(query.limit)
    result_rows = [AnalyticsResultRow(label="Missing" if pd.isna(label) else str(label), value=round(float(value), 6) if proportions else int(value)) for label, value in values.items()]
    chart = ChartSpec(chart_type="bar", title=title, labels=[row.label for row in result_rows], values=[float(row.value) for row in result_rows])
    return AnalyticsQueryResponse(answer_type="chart", title=title, rows=result_rows, chart=chart, rows_analyzed=rows, source_filename=filename, calculation=calculation)


def execute_analytics(frame: pd.DataFrame, query: AnalyticsQuery, filename: str) -> AnalyticsQueryResponse:
    _require_columns(frame, query)
    filtered = _filtered(frame, query)
    operation = query.operation
    scalar: Any = None
    title = operation.value.replace("_", " ").title()
    if operation == AnalyticsOperation.ROW_COUNT:
        scalar = len(filtered)
    elif operation == AnalyticsOperation.COLUMN_COUNT:
        scalar = len(filtered.columns)
    elif operation == AnalyticsOperation.COLUMN_NAMES:
        return AnalyticsQueryResponse(answer_type="table", title="Column Names", rows=[AnalyticsResultRow(label=str(name), value=str(dtype)) for name, dtype in filtered.dtypes.items()][:query.limit], rows_analyzed=len(filtered), source_filename=filename, calculation="Returned persisted dataset columns and dtypes.")
    elif operation == AnalyticsOperation.COLUMN_TYPES:
        return AnalyticsQueryResponse(answer_type="table", title="Column Types", rows=[AnalyticsResultRow(label=str(name), value=str(dtype)) for name, dtype in filtered.dtypes.items()][:query.limit], rows_analyzed=len(filtered), source_filename=filename, calculation="Read trusted dataframe dtypes.")
    elif operation == AnalyticsOperation.DUPLICATE_COUNT:
        scalar = int(filtered.duplicated().sum())
    elif operation == AnalyticsOperation.MISSING_COUNT:
        scalar = int(filtered[query.column].isna().sum())
    elif operation == AnalyticsOperation.MISSING_PERCENTAGE:
        scalar = round(float(filtered[query.column].isna().mean() * 100), 6) if len(filtered) else 0.0
    elif operation == AnalyticsOperation.UNIQUE_COUNT:
        scalar = int(filtered[query.column].nunique(dropna=True))
    elif operation == AnalyticsOperation.AGGREGATE:
        scalar = _aggregate(filtered[query.column], query.aggregation, query.quantile)
        title = f"{query.aggregation.value.title()} of {query.column}"
    elif operation in {AnalyticsOperation.VALUE_COUNTS, AnalyticsOperation.PROPORTIONS}:
        return _table(filtered[query.column], query, f"Distribution of {query.column}", f"Computed {operation.value} after {len(query.filters)} validated filters.", filename, len(filtered), operation == AnalyticsOperation.PROPORTIONS)
    elif operation in {AnalyticsOperation.GROUP_COUNT, AnalyticsOperation.GROUP_AGGREGATE}:
        if operation == AnalyticsOperation.GROUP_COUNT:
            grouped = filtered.groupby(query.group_by, dropna=False).size()
        else:
            grouped = filtered.groupby(query.group_by, dropna=False)[query.column].apply(lambda series: _aggregate(series, query.aggregation, query.quantile))
        grouped = grouped.sort_values(ascending=query.sort == "ascending") if query.sort else grouped
        result_rows = [AnalyticsResultRow(label=" / ".join(str(part) for part in (key if isinstance(key, tuple) else (key,))), value=_safe(value)) for key, value in grouped.head(query.limit).items()]
        chart = ChartSpec(chart_type="bar", title=title, labels=[row.label for row in result_rows], values=[float(row.value) for row in result_rows if isinstance(row.value, (int, float))])
        return AnalyticsQueryResponse(answer_type="chart", title=title, rows=result_rows, chart=chart if len(chart.values) == len(result_rows) else None, rows_analyzed=len(filtered), source_filename=filename, calculation=f"Applied {len(query.filters)} filters, grouped by {', '.join(query.group_by)}, then computed {query.aggregation.value if query.aggregation else 'count'}.")
    elif operation == AnalyticsOperation.CORRELATION:
        numeric = filtered.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise DatasetValidationError("Correlation requires at least two numerical columns.")
        matrix = numeric.corr(method=query.correlation_method)
        pairs = []
        names = list(matrix.columns)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                value = _safe(matrix.loc[left, right])
                if value is not None:
                    pairs.append((abs(value), left, right, value))
        pairs.sort(reverse=True)
        return AnalyticsQueryResponse(answer_type="table", title=f"{query.correlation_method.title()} Correlations", rows=[AnalyticsResultRow(label=f"{left} ↔ {right}", value=round(value, 6)) for _, left, right, value in pairs[:query.limit]], rows_analyzed=len(filtered), source_filename=filename, calculation=f"Computed {query.correlation_method} correlation on numeric columns.")
    elif operation == AnalyticsOperation.OUTLIER_SUMMARY:
        numeric = pd.to_numeric(filtered[query.column], errors="coerce").dropna()
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        count = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
        scalar, title = count, f"Potential outliers in {query.column}"
    elif operation == AnalyticsOperation.DISTRIBUTION_SUMMARY:
        numeric = pd.to_numeric(filtered[query.column], errors="coerce").dropna()
        if numeric.empty:
            return _table(filtered[query.column], query, f"Distribution of {query.column}", "Computed bounded value counts.", filename, len(filtered))
        counts, bins = pd.cut(numeric, bins=min(10, max(2, int(math.sqrt(len(numeric))))), duplicates="drop", retbins=True)
        return _table(counts.astype(str), query, f"Distribution of {query.column}", "Computed a bounded histogram from verified values.", filename, len(filtered))
    return AnalyticsQueryResponse(answer_type="scalar", title=title, value=_safe(scalar), rows_analyzed=len(filtered), source_filename=filename, calculation=f"Executed allowlisted operation '{operation.value}' after {len(query.filters)} validated filters.")


def _validate_plan_columns(frame: pd.DataFrame, step: AnalyticsPlanStep) -> None:
    requested = [step.column, *step.group_by, *(item.column for item in step.filters)]
    missing = [name for name in requested if name and name not in frame.columns]
    if missing:
        raise DatasetValidationError(f"Unknown dataset column: {missing[0]}.")


def _result_from_series(
    series: pd.Series,
    title: str,
    filename: str,
    rows_analyzed: int,
    calculation: str,
    chart_type: str = "bar",
    total_groups: int | None = None,
    rows_matched: int | None = None,
    rows_excluded_missing_group: int = 0,
    rows_excluded_missing_value: int = 0,
    rows_used_for_grouped_analysis: int | None = None,
) -> AnalyticsQueryResponse:
    result_rows = [AnalyticsResultRow(label=str(label), value=_safe(value)) for label, value in series.head(100).items()]
    numeric = all(isinstance(row.value, (int, float)) and row.value is not None for row in result_rows)
    chart = ChartSpec(chart_type=chart_type, title=title, labels=[row.label for row in result_rows], values=[float(row.value) for row in result_rows]) if numeric else None
    return AnalyticsQueryResponse(
        answer_type="chart" if chart else "table",
        title=title,
        rows=result_rows,
        chart=chart,
        rows_analyzed=rows_analyzed,
        source_filename=filename,
        calculation=calculation,
        total_groups=total_groups,
        rows_matched=rows_matched,
        rows_excluded_missing_group=rows_excluded_missing_group,
        rows_excluded_missing_value=rows_excluded_missing_value,
        rows_used_for_grouped_analysis=rows_used_for_grouped_analysis,
    )


def execute_analytics_plan(frame: pd.DataFrame, plan: DatasetAnalyticsPlan, filename: str) -> AnalyticsQueryResponse:
    """Execute a validated declarative plan without evaluating user/model supplied code."""
    working = frame
    result: AnalyticsQueryResponse | None = None
    result_series: pd.Series | None = None
    for step_index, step in enumerate(plan.steps):
        _validate_plan_columns(working, step)
        if step.operation == PlanStepOperation.FILTER:
            working = _filtered(working, AnalyticsQuery(operation=AnalyticsOperation.ROW_COUNT, filters=step.filters))
            continue
        if step.operation == PlanStepOperation.AGGREGATE:
            aggregations = _step_aggregations(step)
            primary_aggregation = aggregations[0]
            values = {aggregation: _aggregate(working[step.column], aggregation, None) for aggregation in aggregations} if step.positive_value is None else {Aggregation.MEAN: _planned_aggregate(working[step.column], step)}
            value = values[primary_aggregation]
            title = f"Rate of {step.column}" if step.positive_value is not None else f"{primary_aggregation.value.title()} of {step.column}"
            filters = [item for filter_step in plan.steps if filter_step.operation == PlanStepOperation.FILTER for item in filter_step.filters]
            calculation = f"Applied filters: {_filter_summary(filters)}. Then calculated the share equal to the dataset's verified positive value ({step.positive_value})." if step.positive_value is not None else f"Applied filters: {_filter_summary(filters)}. Then calculated {', '.join(aggregation.value for aggregation in aggregations)} of {step.column}."
            metric_results = [AnalyticsMetricResult(name=aggregation.value, value=metric_value) for aggregation, metric_value in values.items()] if len(values) > 1 else []
            result = AnalyticsQueryResponse(answer_type="table" if metric_results else "scalar", title=title, value=value, rows=[AnalyticsResultRow(label=aggregation.value, value=metric_value) for aggregation, metric_value in values.items()] if metric_results else [], metric_results=metric_results, rows_analyzed=len(working), source_filename=filename, calculation=calculation)
        elif step.operation == PlanStepOperation.GROUP_AGGREGATE:
            rows_matched = len(working)
            aggregations = _step_aggregations(step)
            requires_value = any(aggregation != Aggregation.COUNT for aggregation in aggregations)
            missing_group = working[step.group_by].isna().any(axis=1)
            rows_excluded_missing_group = int(missing_group.sum()) if not step.include_missing else 0
            grouped_working = working if step.include_missing else working.loc[~missing_group]
            missing_value = grouped_working[step.column].isna()
            rows_excluded_missing_value = int(missing_value.sum()) if requires_value else 0
            if requires_value:
                grouped_working = grouped_working.loc[~missing_value]
            grouped = grouped_working.groupby(step.group_by, dropna=False)[step.column]
            grouped_results = {aggregation: grouped.apply(lambda values: _aggregate(values, aggregation, None)) for aggregation in aggregations} if step.positive_value is None else {Aggregation.MEAN: grouped.apply(lambda values: _planned_aggregate(values, step))}
            result_series = grouped_results[aggregations[0]]
            result_series = result_series.sort_values(ascending=step.sort == "ascending") if step.sort else result_series
            total_groups = len(result_series)
            has_later_sort = any(item.operation == PlanStepOperation.SORT_LIMIT for item in plan.steps[step_index + 1:])
            display_series = result_series if has_later_sort else result_series.head(step.limit)
            labels = pd.Series(display_series.values, index=[" / ".join(_display_group_part(part, step.include_missing) for part in (key if isinstance(key, tuple) else (key,))) for key in display_series.index])
            result_series.index = [" / ".join(_display_group_part(part, step.include_missing) for part in (key if isinstance(key, tuple) else (key,))) for key in result_series.index]
            title = f"{step.column} rate by {', '.join(step.group_by)}" if step.positive_value is not None else f"{aggregations[0].value.title()} {step.column} by {', '.join(step.group_by)}"
            filters = [item for filter_step in plan.steps if filter_step.operation == PlanStepOperation.FILTER for item in filter_step.filters]
            calculation = f"Applied filters: {_filter_summary(filters)}. Rows matched filters: {rows_matched}. Rows excluded because {', '.join(step.group_by)} was missing: {rows_excluded_missing_group}. Rows excluded because {step.column} was missing: {rows_excluded_missing_value}. Rows used for grouped analysis: {len(grouped_working)}. "
            calculation += f"Grouped verified rows by {', '.join(step.group_by)} and calculated the share equal to the dataset's verified positive value ({step.positive_value})." if step.positive_value is not None else f"Grouped verified rows by {', '.join(step.group_by)} and calculated {', '.join(aggregation.value for aggregation in aggregations)} of {step.column}."
            metric_results = []
            if len(grouped_results) > 1:
                for aggregation, values in grouped_results.items():
                    values = values.sort_values(ascending=step.sort == "ascending") if step.sort else values
                    metric_results.append(AnalyticsMetricResult(name=aggregation.value, rows=[AnalyticsResultRow(label=" / ".join(_display_group_part(part, step.include_missing) for part in (key if isinstance(key, tuple) else (key,))), value=_safe(value)) for key, value in values.items()]))
            result = _result_from_series(labels, title, filename, len(grouped_working), calculation, total_groups=total_groups, rows_matched=rows_matched, rows_excluded_missing_group=rows_excluded_missing_group, rows_excluded_missing_value=rows_excluded_missing_value, rows_used_for_grouped_analysis=len(grouped_working))
            if metric_results:
                result = result.model_copy(update={"answer_type": "table", "metric_results": metric_results})
        elif step.operation == PlanStepOperation.VALUE_COUNTS:
            result_series = working[step.column].value_counts(dropna=False).head(step.limit)
            result = _result_from_series(result_series, f"Distribution of {step.column}", filename, len(working), f"Counted values in {step.column} from verified rows.", "bar")
        elif step.operation == PlanStepOperation.CORRELATION:
            numeric = working.select_dtypes(include="number")
            if numeric.shape[1] < 2:
                raise DatasetValidationError("Correlation requires at least two numerical columns.")
            matrix = numeric.corr(method=step.correlation_method)
            pairs = [(abs(value), f"{left} ↔ {right}", value) for index, left in enumerate(matrix.columns) for right, value in matrix.iloc[index + 1:, index].items() if pd.notna(value)]
            result_series = pd.Series({label: round(float(value), 6) for _, label, value in sorted(pairs, reverse=True)[:step.limit]})
            result = _result_from_series(result_series, f"{step.correlation_method.title()} correlations", filename, len(working), f"Calculated {step.correlation_method} correlations across compatible numeric columns.", "heatmap")
        elif step.operation == PlanStepOperation.ASSOCIATION:
            target = pd.to_numeric(working[step.column], errors="coerce")
            if target.notna().sum() < 2:
                raise DatasetValidationError("Association analysis currently requires a numerical target column.")
            rows = {}
            for name in working.select_dtypes(include="number").columns:
                if name == step.column:
                    continue
                coefficient = target.corr(pd.to_numeric(working[name], errors="coerce"), method=step.correlation_method)
                if pd.notna(coefficient):
                    rows[name] = round(float(coefficient), 6)
            if not rows:
                raise DatasetValidationError("No compatible numerical features are available for association analysis.")
            result_series = pd.Series(rows).sort_values(key=lambda values: values.abs(), ascending=False).head(step.limit)
            result = _result_from_series(result_series, f"Features associated with {step.column}", filename, len(working), f"Ranked compatible numerical features by absolute {step.correlation_method} correlation with {step.column}.", "bar")
        elif step.operation == PlanStepOperation.DISTRIBUTION:
            result = execute_analytics(working, AnalyticsQuery(operation=AnalyticsOperation.DISTRIBUTION_SUMMARY, column=step.column, limit=step.limit), filename)
            if result.chart:
                result.chart.chart_type = "histogram"
        elif step.operation == PlanStepOperation.OUTLIER_SUMMARY:
            result = execute_analytics(working, AnalyticsQuery(operation=AnalyticsOperation.OUTLIER_SUMMARY, column=step.column), filename)
        elif step.operation == PlanStepOperation.SORT_LIMIT:
            if result_series is None or result is None:
                raise DatasetValidationError("sort_limit requires a tabular analytical result.")
            result_series = result_series.sort_values(ascending=step.sort == "ascending").head(step.limit)
            result = _result_from_series(
                result_series,
                result.title,
                filename,
                result.rows_analyzed,
                result.calculation + f" Sorted {step.sort} and limited to {step.limit}.",
                total_groups=result.total_groups if result.total_groups is not None else len(result_series),
                rows_matched=result.rows_matched,
                rows_excluded_missing_group=result.rows_excluded_missing_group,
                rows_excluded_missing_value=result.rows_excluded_missing_value,
                rows_used_for_grouped_analysis=result.rows_used_for_grouped_analysis,
            )
    if result is None:
        raise DatasetValidationError("I can't answer that reliably from the available dataset with the currently supported analytical operations.")
    return result
