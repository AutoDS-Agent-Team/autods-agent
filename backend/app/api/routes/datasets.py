from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.dependencies import get_db
from app.schemas.dataset import DatasetDeleteResponse, DatasetListResponse, DatasetProfileResponse, DatasetResponse, WorksheetSelectionRequest
from app.schemas.auto_analysis import DatasetAutoAnalysisResponse
from app.schemas.analytics import AnalyticsQuery, AnalyticsQueryResponse
from app.schemas.agents import DataAnalystResponse
from app.services.dataset_service import (
    create_dataset,
    delete_dataset,
    get_dataset_metadata,
    get_dataset_profile,
    list_user_datasets,
    load_dataset_dataframe,
    select_worksheet,
    worksheet_names,
)
from app.models.user import User
from app.services.security import get_current_user
from app.core.exceptions import DatasetNotFoundError
from app.core.rate_limit import limit
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.services.agent_service import analyze_profile
from app.schemas.clustering import ClusteringResponse
from app.services.clustering_service import cluster_dataset
from app.services.auto_analysis_service import analyze_dataset
from app.services.analytics_service import execute_analytics
from app.schemas.specialized_analysis import AnomalyRequest, AnomalyResponse, ForecastRequest, ForecastResponse
from app.services.specialized_analysis_service import detect_anomalies, forecast_series

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _response(dataset, settings: Settings) -> DatasetResponse:
    response = DatasetResponse.model_validate(dataset)
    return response.model_copy(update={"worksheet_names": worksheet_names(dataset, settings)})


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetListResponse:
    return DatasetListResponse(items=[_response(dataset, settings) for dataset in list_user_datasets(user.id, database)])


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(limit("write"))])
async def upload_dataset(
    file: Annotated[UploadFile, File(description="A structured CSV or XLSX dataset")],
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetResponse:
    dataset = await create_dataset(file, database, settings, user)
    return _response(dataset, settings)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def read_dataset(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return _response(dataset, settings)


@router.delete("/{dataset_id}", response_model=DatasetDeleteResponse, dependencies=[Depends(limit("write"))])
def remove_dataset(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
    cascade: bool = False,
) -> DatasetDeleteResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    delete_dataset(dataset, database, settings, cascade=cascade)
    return DatasetDeleteResponse(dataset_id=dataset_id)


@router.put("/{dataset_id}/worksheet", response_model=DatasetResponse, dependencies=[Depends(limit("write"))])
def update_worksheet(
    dataset_id: str,
    request: WorksheetSelectionRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    dataset = select_worksheet(dataset, request.worksheet_name, database, settings)
    return _response(dataset, settings)


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def read_dataset_profile(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetProfileResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return get_dataset_profile(dataset_id, database, settings)


@router.post("/{dataset_id}/analysis", response_model=DataAnalystResponse, dependencies=[Depends(limit("write"))])
def analyze_dataset_profile(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DataAnalystResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    profile = get_dataset_profile(dataset_id, database, settings)
    return analyze_profile(profile, (
        GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds),
        OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds),
    ), settings.llm_max_retries)


@router.get("/{dataset_id}/auto-analysis", response_model=DatasetAutoAnalysisResponse)
def auto_analyze_dataset(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> DatasetAutoAnalysisResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return analyze_dataset(dataset.id, load_dataset_dataframe(dataset, settings))


@router.post("/{dataset_id}/analytics/query", response_model=AnalyticsQueryResponse, dependencies=[Depends(limit("write"))])
def query_dataset(
    dataset_id: str,
    request: AnalyticsQuery,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> AnalyticsQueryResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return execute_analytics(load_dataset_dataframe(dataset, settings), request, dataset.original_filename)


@router.post("/{dataset_id}/clusters", response_model=ClusteringResponse, dependencies=[Depends(limit("write"))])
def cluster_uploaded_dataset(
    dataset_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> ClusteringResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return cluster_dataset(dataset_id, database, settings)


@router.post("/{dataset_id}/anomalies", response_model=AnomalyResponse, dependencies=[Depends(limit("write"))])
def analyze_anomalies(
    dataset_id: str,
    request: AnomalyRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> AnomalyResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return detect_anomalies(dataset_id, load_dataset_dataframe(dataset, settings), request, settings.training_random_state)


@router.post("/{dataset_id}/forecast", response_model=ForecastResponse, dependencies=[Depends(limit("write"))])
def forecast_dataset(
    dataset_id: str,
    request: ForecastRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> ForecastResponse:
    dataset = get_dataset_metadata(dataset_id, database, settings)
    if dataset.user_id != user.id:
        raise DatasetNotFoundError("Dataset not found.")
    return forecast_series(dataset_id, load_dataset_dataframe(dataset, settings), request)
