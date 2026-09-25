from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    stored_filename: str
    file_size: int
    row_count: int
    column_count: int
    has_header: bool
    source_format: str = "csv"
    worksheet_name: str | None = None
    worksheet_names: list[str] = Field(default_factory=list)
    status: str = "READY"
    created_at: datetime
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def generated_column_names(self) -> bool:
        return not self.has_header


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse] = Field(default_factory=list)


class DatasetDeleteResponse(BaseModel):
    dataset_id: str
    deleted: bool = True


class NumericStatistics(BaseModel):
    count: int
    mean: float | None
    std: float | None
    min: float | None
    percentile_25: float | None
    median: float | None
    percentile_75: float | None
    max: float | None


class ValueFrequency(BaseModel):
    value: str | None
    count: int


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    logical_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_constant: bool
    is_possible_id: bool
    numeric_statistics: NumericStatistics | None = None
    top_values: list[ValueFrequency] | None = None


class DatasetProfileSummary(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    total_missing_values: int
    numerical_columns: list[str]
    categorical_columns: list[str]
    constant_columns: list[str]
    possible_id_columns: list[str]


class CorrelationEntry(BaseModel):
    column_x: str
    column_y: str
    coefficient: float


class DatasetProfileResponse(BaseModel):
    dataset: DatasetResponse
    summary: DatasetProfileSummary
    columns: list[ColumnProfile]
    correlations: list[CorrelationEntry]


class WorksheetSelectionRequest(BaseModel):
    worksheet_name: str = Field(min_length=1, max_length=255)
