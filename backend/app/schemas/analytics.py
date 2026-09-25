from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AnalyticsOperation(StrEnum):
    ROW_COUNT = "row_count"
    COLUMN_COUNT = "column_count"
    COLUMN_NAMES = "column_names"
    COLUMN_TYPES = "column_types"
    MISSING_COUNT = "missing_count"
    MISSING_PERCENTAGE = "missing_percentage"
    DUPLICATE_COUNT = "duplicate_count"
    UNIQUE_COUNT = "unique_count"
    AGGREGATE = "aggregate"
    VALUE_COUNTS = "value_counts"
    PROPORTIONS = "proportions"
    GROUP_COUNT = "group_count"
    GROUP_AGGREGATE = "group_aggregate"
    CORRELATION = "correlation"
    OUTLIER_SUMMARY = "outlier_summary"
    DISTRIBUTION_SUMMARY = "distribution_summary"


class FilterOperator(StrEnum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"


class Aggregation(StrEnum):
    COUNT = "count"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    STD = "std"
    VARIANCE = "variance"
    QUANTILE = "quantile"


class PlanStepOperation(StrEnum):
    """The complete, intentionally small language understood by the dataset executor."""

    FILTER = "filter"
    AGGREGATE = "aggregate"
    GROUP_AGGREGATE = "group_aggregate"
    VALUE_COUNTS = "value_counts"
    CORRELATION = "correlation"
    ASSOCIATION = "association"
    DISTRIBUTION = "distribution"
    OUTLIER_SUMMARY = "outlier_summary"
    SORT_LIMIT = "sort_limit"


class AnalyticsPlanStep(BaseModel):
    """One allowlisted operation.  No expressions, code, paths, or function names exist here."""

    operation: PlanStepOperation
    filters: list["AnalyticsFilter"] = Field(default_factory=list, max_length=5)
    column: str | None = Field(default=None, max_length=255)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    aggregation: Aggregation | None = None
    aggregations: list[Aggregation] = Field(default_factory=list, max_length=3)
    positive_value: Any = None
    include_missing: bool = False
    sort: Literal["ascending", "descending"] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    correlation_method: Literal["pearson", "spearman"] = "pearson"

    @model_validator(mode="after")
    def validate_step(self) -> "AnalyticsPlanStep":
        if self.operation == PlanStepOperation.FILTER:
            if not self.filters:
                raise ValueError("Filter steps require at least one condition.")
            if self.column or self.group_by or self.aggregation or self.aggregations:
                raise ValueError("Filter steps only accept filter conditions.")
        elif self.filters:
            raise ValueError("Filters must be expressed in a preceding filter step.")
        if self.operation == PlanStepOperation.AGGREGATE and (not self.column or not (self.aggregation or self.aggregations)):
            raise ValueError("Aggregate steps require a column and aggregation.")
        if self.operation == PlanStepOperation.GROUP_AGGREGATE and (not self.column or not self.group_by or not (self.aggregation or self.aggregations)):
            raise ValueError("Grouped aggregate steps require column, group_by, and aggregation.")
        if self.aggregations and self.aggregation and self.aggregations[0] != self.aggregation:
            raise ValueError("The primary aggregation must be the first aggregation.")
        if self.aggregations and self.aggregation is None:
            self.aggregation = self.aggregations[0]
        if len(self.aggregations) != len(set(self.aggregations)):
            raise ValueError("Aggregations must be unique.")
        if self.positive_value is not None and (
            self.operation not in {PlanStepOperation.AGGREGATE, PlanStepOperation.GROUP_AGGREGATE}
            or (self.aggregations and self.aggregations != [Aggregation.MEAN])
            or (not self.aggregations and self.aggregation != Aggregation.MEAN)
        ):
            raise ValueError("A positive value is supported only for a mean aggregation.")
        if self.operation in {PlanStepOperation.VALUE_COUNTS, PlanStepOperation.DISTRIBUTION, PlanStepOperation.OUTLIER_SUMMARY, PlanStepOperation.ASSOCIATION} and not self.column:
            raise ValueError("This step requires a column.")
        return self


class DatasetAnalyticsPlan(BaseModel):
    """A bounded declarative plan generated from a natural-language question."""

    steps: list[AnalyticsPlanStep] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_order(self) -> "DatasetAnalyticsPlan":
        result_steps = {PlanStepOperation.AGGREGATE, PlanStepOperation.GROUP_AGGREGATE, PlanStepOperation.VALUE_COUNTS, PlanStepOperation.CORRELATION, PlanStepOperation.ASSOCIATION, PlanStepOperation.DISTRIBUTION, PlanStepOperation.OUTLIER_SUMMARY}
        seen_result = False
        for step in self.steps:
            if seen_result and step.operation != PlanStepOperation.SORT_LIMIT:
                raise ValueError("Only sort_limit may follow a result-producing step.")
            if step.operation in result_steps:
                seen_result = True
            if step.operation == PlanStepOperation.SORT_LIMIT and not seen_result:
                raise ValueError("sort_limit must follow a result-producing step.")
        if not seen_result:
            raise ValueError("A plan must contain a result-producing step.")
        return self


class AnalyticsFilter(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    operator: FilterOperator
    value: Any


class AnalyticsQuery(BaseModel):
    operation: AnalyticsOperation
    column: str | None = Field(default=None, max_length=255)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    aggregation: Aggregation | None = None
    filters: list[AnalyticsFilter] = Field(default_factory=list, max_length=5)
    sort: Literal["ascending", "descending"] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    quantile: float | None = Field(default=None, ge=0, le=1)
    correlation_method: Literal["pearson", "spearman"] = "pearson"

    @field_validator("group_by")
    @classmethod
    def unique_group_columns(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Group-by columns must be unique.")
        return values

    @model_validator(mode="after")
    def validate_shape(self) -> "AnalyticsQuery":
        column_ops = {AnalyticsOperation.MISSING_COUNT, AnalyticsOperation.MISSING_PERCENTAGE, AnalyticsOperation.UNIQUE_COUNT, AnalyticsOperation.AGGREGATE, AnalyticsOperation.VALUE_COUNTS, AnalyticsOperation.PROPORTIONS, AnalyticsOperation.OUTLIER_SUMMARY, AnalyticsOperation.DISTRIBUTION_SUMMARY}
        if self.operation in column_ops and not self.column:
            raise ValueError("This operation requires a column.")
        if self.operation == AnalyticsOperation.AGGREGATE and not self.aggregation:
            raise ValueError("Aggregate requires an aggregation.")
        if self.operation in {AnalyticsOperation.GROUP_COUNT, AnalyticsOperation.GROUP_AGGREGATE} and not self.group_by:
            raise ValueError("Grouping requires at least one group-by column.")
        if self.operation == AnalyticsOperation.GROUP_AGGREGATE and (not self.column or not self.aggregation):
            raise ValueError("Grouped aggregate requires a column and aggregation.")
        return self


class AnalyticsResultRow(BaseModel):
    label: str
    value: Any


class AnalyticsMetricResult(BaseModel):
    name: str
    value: Any = None
    rows: list[AnalyticsResultRow] = Field(default_factory=list, max_length=100)


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "histogram", "pie", "donut", "scatter", "heatmap"]
    title: str
    labels: list[str] = Field(default_factory=list, max_length=100)
    values: list[float] = Field(default_factory=list, max_length=100)


class AnalyticsQueryResponse(BaseModel):
    answer_type: Literal["scalar", "table", "chart"]
    title: str
    value: Any = None
    rows: list[AnalyticsResultRow] = Field(default_factory=list, max_length=100)
    metric_results: list[AnalyticsMetricResult] = Field(default_factory=list, max_length=3)
    chart: ChartSpec | None = None
    rows_analyzed: int
    source_filename: str
    calculation: str
    value_format: Literal["number", "percent"] = "number"
    metric_label: str | None = None
    group_label: str | None = None
    total_groups: int | None = Field(default=None, ge=0)
    rows_matched: int | None = Field(default=None, ge=0)
    rows_excluded_missing_group: int = Field(default=0, ge=0)
    rows_excluded_missing_value: int = Field(default=0, ge=0)
    rows_used_for_grouped_analysis: int | None = Field(default=None, ge=0)
    insights: list[str] = Field(default_factory=list, max_length=5)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)

