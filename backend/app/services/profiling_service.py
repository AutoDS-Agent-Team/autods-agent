import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from app.schemas.dataset import (
    ColumnProfile,
    CorrelationEntry,
    DatasetProfileSummary,
    NumericStatistics,
    ValueFrequency,
)

MAX_CORRELATION_PAIRS = 50
HEADER_SAMPLE_ROWS = 25
HEADER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]*$")
NUMERIC_INFERENCE_THRESHOLD = 0.95
TEXT_MIN_AVERAGE_LENGTH = 24
TEXT_MIN_MULTIWORD_RATIO = 0.35
NUMERIC_CODE_NAME_TOKENS = ("zip", "zipcode", "postal", "postcode", "pclass", "class_code", "category_code")
DATETIME_NAME_PATTERN = re.compile(r"(?:^|_)(date|datetime|timestamp|time)(?:$|_)")
DATETIME_NAME_TOKENS = ("date", "time", "timestamp", "datetime")
ISO_DATE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$")


@dataclass(frozen=True)
class ParsedCsv:
    dataframe: pd.DataFrame
    has_header: bool
    has_metadata_preamble: bool


def _sample_csv_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file, strict=True)
        for row in reader:
            if any(value.strip() for value in row):
                rows.append(row)
            if len(rows) >= HEADER_SAMPLE_ROWS:
                break
    return rows


def _looks_like_column_names(row: list[str]) -> bool:
    normalized = [value.strip() for value in row]
    return bool(normalized) and len(set(normalized)) == len(normalized) and all(
        HEADER_NAME_PATTERN.fullmatch(value) for value in normalized
    )


def _value_kind(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return "empty"
    if normalized.lower() in {"true", "false"}:
        return "boolean"
    try:
        converted = float(normalized)
        return "number" if math.isfinite(converted) else "string"
    except ValueError:
        return "string"


def detect_csv_header(path: Path) -> bool:
    rows = _sample_csv_rows(path)
    if not rows or not _looks_like_column_names(rows[0]):
        return False
    if len(rows) == 1:
        return True

    evidence_count = 0
    for column_index in range(len(rows[0])):
        observed_kinds = [
            _value_kind(row[column_index])
            for row in rows[1:]
            if column_index < len(row) and row[column_index].strip()
        ]
        if not observed_kinds:
            continue
        dominant_kind = max(set(observed_kinds), key=observed_kinds.count)
        dominant_ratio = observed_kinds.count(dominant_kind) / len(observed_kinds)
        if dominant_kind in {"number", "boolean"} and dominant_ratio >= 0.8:
            evidence_count += 1

    return evidence_count > 0


def _has_metadata_preamble(path: Path) -> bool:
    rows = _sample_csv_rows(path)
    if len(rows) < 2 or len(rows[0]) < 3:
        return False

    first_row = rows[0]
    try:
        sample_count = int(first_row[0])
        feature_count = int(first_row[1])
    except ValueError:
        return False

    if sample_count < len(rows) - 1 or feature_count != len(first_row) - 1:
        return False
    if not all(_value_kind(value) == "string" for value in first_row[2:]):
        return False

    for column_index in range(feature_count):
        observed = [
            _value_kind(row[column_index])
            for row in rows[1:]
            if len(row) == len(first_row) and row[column_index].strip()
        ]
        if not observed or observed.count("number") / len(observed) < 0.8:
            return False
    return True


def read_csv_dataset(path: Path, has_header: bool | None = None) -> ParsedCsv:
    detected_header = detect_csv_header(path) if has_header is None else has_header
    has_metadata_preamble = not detected_header and _has_metadata_preamble(path)
    dataframe = pd.read_csv(
        path,
        header=0 if detected_header else None,
        skiprows=1 if has_metadata_preamble else None,
        on_bad_lines="error",
    )
    if not detected_header:
        dataframe.columns = [
            f"column_{index}" for index in range(1, dataframe.shape[1] + 1)
        ]
    if has_metadata_preamble and dataframe.shape[1]:
        last_column = dataframe.columns[-1]
        dataframe[last_column] = dataframe[last_column].astype("string")
        dataframe.attrs["categorical_columns"] = {str(last_column)}
    return ParsedCsv(
        dataframe=dataframe,
        has_header=detected_header,
        has_metadata_preamble=has_metadata_preamble,
    )


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _logical_type(series: pd.Series) -> str:
    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_numeric_dtype(series.dtype):
        return "numerical"
    if is_datetime64_any_dtype(series.dtype):
        return "datetime"
    non_null_count = int(series.notna().sum())
    if non_null_count:
        numeric_count = int(pd.to_numeric(series, errors="coerce").notna().sum())
        if numeric_count / non_null_count >= NUMERIC_INFERENCE_THRESHOLD:
            return "numerical"
    values = series.dropna().astype(str)
    if len(values) >= 10 and values.str.len().mean() >= TEXT_MIN_AVERAGE_LENGTH and values.str.contains(r"\s+").mean() >= TEXT_MIN_MULTIWORD_RATIO and values.nunique() / len(values) >= 0.3:
        return "text"
    return "categorical"


def _logical_type_for_column(name: str, series: pd.Series, categorical_hints: set[str]) -> str:
    """Keep numeric measurement columns separate from numeric identifiers and location codes."""
    if name in categorical_hints:
        return "categorical"
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if not is_numeric_dtype(series.dtype) and DATETIME_NAME_PATTERN.search(normalized):
        non_null = series.dropna()
        if len(non_null) and pd.to_datetime(non_null, errors="coerce").notna().mean() >= NUMERIC_INFERENCE_THRESHOLD:
            return "datetime"
    if is_numeric_dtype(series.dtype) and any(token in normalized for token in NUMERIC_CODE_NAME_TOKENS):
        return "categorical"
    if any(token in normalized for token in ("text", "description", "comment", "message", "body")) and not is_numeric_dtype(series.dtype):
        return "text"
    # CSV date columns are usually loaded as strings.  Require both a temporal
    # name and consistently ISO-like values so ordinary categorical strings do
    # not get coerced into dates by an overly-permissive parser.
    values = series.dropna().astype(str).str.strip()
    if (
        not is_numeric_dtype(series.dtype)
        and len(values) >= 3
        and any(token in normalized for token in DATETIME_NAME_TOKENS)
        and float(values.str.match(ISO_DATE_PATTERN).mean()) >= 0.9
        and float(pd.to_datetime(values, errors="coerce").notna().mean()) >= 0.9
    ):
        return "datetime"
    return _logical_type(series)


def _is_possible_id(name: str, series: pd.Series) -> bool:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return False

    unique_ratio = float(series.nunique(dropna=True)) / non_null_count
    normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")
    name_suggests_text = any(token in normalized_name for token in ("text", "description", "comment", "message", "body"))
    name_suggests_id = normalized_name == "id" or normalized_name.endswith("_id") or normalized_name.endswith("id") or any(token in normalized_name for token in ("uuid", "email", "phone", "index"))

    if name_suggests_text:
        return False

    if name_suggests_id and unique_ratio >= 0.8:
        return True
    # Unique numeric measurements (age, price, dimensions) are often valid
    # features. Only infer an unnamed identifier from high-cardinality text.
    if is_numeric_dtype(series.dtype):
        return False
    values = series.dropna().astype(str)
    return non_null_count >= 20 and unique_ratio >= 0.98 and float(values.str.len().mean()) >= 8


def _numeric_statistics(series: pd.Series) -> NumericStatistics:
    description = series.describe(percentiles=[0.25, 0.5, 0.75])
    return NumericStatistics(
        count=int(description["count"]),
        mean=_safe_float(description.get("mean")),
        std=_safe_float(description.get("std")),
        min=_safe_float(description.get("min")),
        percentile_25=_safe_float(description.get("25%")),
        median=_safe_float(description.get("50%")),
        percentile_75=_safe_float(description.get("75%")),
        max=_safe_float(description.get("max")),
    )


def _top_values(series: pd.Series, limit: int) -> list[ValueFrequency]:
    frequencies = series.value_counts(dropna=False).head(limit)
    return [
        ValueFrequency(
            value=None if pd.isna(value) else str(value),
            count=int(count),
        )
        for value, count in frequencies.items()
    ]


def _correlations(
    dataframe: pd.DataFrame,
    numerical_columns: list[str],
) -> list[CorrelationEntry]:
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(dataframe[column], errors="coerce")
            for column in numerical_columns
        }
    )
    if numeric.shape[1] < 2:
        return []

    matrix = numeric.corr()
    entries: list[CorrelationEntry] = []
    names = list(matrix.columns)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            coefficient = _safe_float(matrix.loc[left_name, right_name])
            if coefficient is not None:
                entries.append(
                    CorrelationEntry(
                        column_x=str(left_name),
                        column_y=str(right_name),
                        coefficient=coefficient,
                    )
                )

    entries.sort(key=lambda entry: abs(entry.coefficient), reverse=True)
    return entries[:MAX_CORRELATION_PAIRS]


def profile_dataframe(
    dataframe: pd.DataFrame,
    top_values_limit: int = 10,
) -> tuple[DatasetProfileSummary, list[ColumnProfile], list[CorrelationEntry]]:
    row_count, column_count = dataframe.shape
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    constant_columns: list[str] = []
    possible_id_columns: list[str] = []
    column_profiles: list[ColumnProfile] = []
    categorical_hints = set(dataframe.attrs.get("categorical_columns", set()))

    for column_name in dataframe.columns:
        name = str(column_name)
        series = dataframe[column_name]
        normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        logical_type = _logical_type_for_column(name, series, categorical_hints)
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_possible_id = _is_possible_id(name, series)

        if logical_type == "numerical":
            numerical_columns.append(name)
        else:
            categorical_columns.append(name)
        if is_constant:
            constant_columns.append(name)
        if is_possible_id:
            possible_id_columns.append(name)

        column_profiles.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                logical_type=logical_type,
                missing_count=missing_count,
                missing_percentage=round((missing_count / row_count) * 100, 4),
                unique_count=unique_count,
                is_constant=is_constant,
                is_possible_id=is_possible_id,
                numeric_statistics=(
                    _numeric_statistics(pd.to_numeric(series, errors="coerce"))
                    if logical_type == "numerical"
                    else None
                ),
                top_values=(
                    _top_values(series, top_values_limit)
                    if logical_type != "numerical"
                    else None
                ),
            )
        )

    summary = DatasetProfileSummary(
        row_count=int(row_count),
        column_count=int(column_count),
        duplicate_row_count=int(dataframe.duplicated().sum()),
        total_missing_values=int(dataframe.isna().sum().sum()),
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        constant_columns=constant_columns,
        possible_id_columns=possible_id_columns,
    )
    return summary, column_profiles, _correlations(dataframe, numerical_columns)
