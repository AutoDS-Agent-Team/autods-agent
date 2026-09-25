"""Trusted, bounded unsupervised clustering for structured datasets."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import TrainingFailureError
from app.schemas.clustering import ClusterSummary, ClusteringResponse
from app.services.dataset_service import get_dataset_metadata, load_dataset_dataframe
from app.services.profiling_service import profile_dataframe


def cluster_dataset(dataset_id: str, database: Session, settings: Settings) -> ClusteringResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    _, profiles, _ = profile_dataframe(dataframe)
    excluded = [item.name for item in profiles if item.is_possible_id or item.is_constant]
    numeric = [item.name for item in profiles if item.logical_type == "numerical" and item.name not in excluded]
    text = [item.name for item in profiles if item.logical_type == "text" and item.name not in excluded]
    categorical = [item.name for item in profiles if item.logical_type not in {"numerical", "text"} and item.name not in excluded]
    features = dataframe.drop(columns=excluded, errors="ignore").copy()
    for column in text: features[column] = features[column].fillna("").astype(str)
    transformers = []
    if numeric: transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical: transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    for column in text: transformers.append((f"text_{column}", TfidfVectorizer(max_features=settings.text_tfidf_max_features, ngram_range=(1, 2), min_df=2), column))
    if not transformers or len(features) < 4:
        raise TrainingFailureError("Clustering needs at least four rows and one usable non-identifier feature.")
    matrix = ColumnTransformer(transformers).fit_transform(features)
    maximum = min(8, len(features) - 1)
    if maximum < 2: raise TrainingFailureError("Clustering needs at least two candidate clusters.")
    best_k, best_score, best_labels = 2, -1.0, None
    for clusters in range(2, maximum + 1):
        labels = KMeans(n_clusters=clusters, random_state=settings.training_random_state, n_init=10).fit_predict(matrix)
        score = float(silhouette_score(matrix, labels, sample_size=min(2000, len(features)), random_state=settings.training_random_state))
        if score > best_score: best_k, best_score, best_labels = clusters, score, labels
    counts = pd.Series(best_labels).value_counts().sort_index()
    return ClusteringResponse(dataset_id=dataset_id, selected_cluster_count=best_k, silhouette_score=round(best_score, 6), cluster_summaries=[ClusterSummary(cluster=int(key), row_count=int(value), percentage=round(value / len(features) * 100, 2)) for key, value in counts.items()], excluded_columns=excluded, text_columns=text, feature_count=int(matrix.shape[1]))
