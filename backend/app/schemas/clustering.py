from pydantic import BaseModel, Field


class ClusterSummary(BaseModel):
    cluster: int
    row_count: int
    percentage: float


class ClusteringResponse(BaseModel):
    dataset_id: str
    algorithm: str = "kmeans"
    selected_cluster_count: int
    silhouette_score: float | None
    cluster_summaries: list[ClusterSummary]
    excluded_columns: list[str]
    text_columns: list[str]
    feature_count: int
    status: str = "COMPLETED"

