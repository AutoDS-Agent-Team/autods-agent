from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.dependencies import get_db
from app.schemas.experiment import (
    ExperimentAnalysisResponse,
    ExperimentConfirmRequest,
    ExperimentCreateRequest,
    ExperimentResponse,
)
from app.services.experiment_service import create_experiment, confirm_experiment
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.router import LLMRouter
from app.schemas.pipeline_plan import PlanOverrideRequest, PlanResponse
from app.services.planning_service import create_pipeline_plan, update_pipeline_overrides
from app.schemas.training import TrainingResponse
from app.services.training_service import train_experiment
from app.schemas.evaluation import EvaluationResponse, ExplainabilityResponse, OptimizationResponse, PredictionResponse, ReportResponse
from app.services.evaluation_service import evaluate_experiment
from app.services.explainability_service import get_explainability
from app.services.optimization_service import optimize_experiment
from app.services.prediction_service import create_predictions
from app.services.report_service import create_report, create_pdf_report
from app.schemas.job import JobResponse, ExperimentHistoryResponse, ExperimentDetailResponse
from app.schemas.agents import InsightReflectionResponse
from app.services.job_service import create_job, history, experiment_detail
from app.models.user import User
from app.services.security import get_current_user, require_experiment_owner
from app.core.rate_limit import limit
from app.services.reflection_service import create_reflection
from app.core.exceptions import TrainingNotReadyError

router = APIRouter(prefix="/experiments", tags=["experiments"], dependencies=[Depends(get_current_user)])

def _require_owned_path(experiment_id: str, database: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]) -> None:
    require_experiment_owner(experiment_id, user, database)

@router.get("", response_model=ExperimentHistoryResponse)
def experiment_history(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), database: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ExperimentHistoryResponse:
    return history(page, page_size, database, user.id)

@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment_detail(experiment_id: str, database: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]) -> ExperimentDetailResponse:
    require_experiment_owner(experiment_id, user, database)
    return experiment_detail(experiment_id, database)


def get_llm_router(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMRouter:
    return LLMRouter(
        primary=GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        fallback=OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        max_retries=settings.llm_max_retries,
    )


@router.post(
    "",
    response_model=ExperimentAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment_definition(
    request: ExperimentCreateRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> ExperimentAnalysisResponse:
    return create_experiment(request, database, settings, user)


@router.post("/{experiment_id}/confirm", response_model=ExperimentResponse, dependencies=[Depends(_require_owned_path)])
def confirm_experiment_definition(
    experiment_id: str,
    request: ExperimentConfirmRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExperimentResponse:
    return confirm_experiment(experiment_id, request, database, settings)


@router.post("/{experiment_id}/plan", response_model=PlanResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def plan_experiment_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    llm_router: Annotated[LLMRouter, Depends(get_llm_router)],
) -> PlanResponse:
    return create_pipeline_plan(experiment_id, database, settings, llm_router)


@router.patch("/{experiment_id}/plan/adaptive", response_model=PlanResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def override_adaptive_plan_definition(
    experiment_id: str,
    request: PlanOverrideRequest,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PlanResponse:
    return update_pipeline_overrides(experiment_id, request, database, settings)


@router.post("/{experiment_id}/train", response_model=TrainingResponse, dependencies=[Depends(_require_owned_path)])
def train_experiment_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrainingResponse:
    return train_experiment(experiment_id, database, settings)


@router.post("/{experiment_id}/evaluate", response_model=EvaluationResponse, dependencies=[Depends(_require_owned_path)])
def evaluate_experiment_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EvaluationResponse:
    return evaluate_experiment(experiment_id, database, settings)


@router.post("/{experiment_id}/optimize", response_model=OptimizationResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def optimize_experiment_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> OptimizationResponse:
    return optimize_experiment(experiment_id, database, settings)


@router.get("/{experiment_id}/explainability", response_model=ExplainabilityResponse, dependencies=[Depends(_require_owned_path)])
def explainability_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExplainabilityResponse:
    return get_explainability(experiment_id, database, settings)


@router.post("/{experiment_id}/predictions", response_model=PredictionResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
async def predictions_definition(
    experiment_id: str,
    file: Annotated[UploadFile, File(...)],
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictionResponse:
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise TrainingNotReadyError("Prediction CSV exceeds the configured upload limit.")
    return create_predictions(experiment_id, content, file.filename or "", database, settings)


@router.post("/{experiment_id}/report", response_model=ReportResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def report_definition(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReportResponse:
    return create_report(experiment_id, database, settings)

@router.post("/{experiment_id}/report/pdf", response_model=ReportResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def pdf_report_definition(experiment_id: str, database: Annotated[Session, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]) -> ReportResponse:
    return create_pdf_report(experiment_id, database, settings)

@router.post("/{experiment_id}/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def run_background_experiment(experiment_id: str, database: Annotated[Session, Depends(get_db)]) -> JobResponse:
    return create_job(experiment_id, "train", database)


@router.post("/{experiment_id}/reflection", response_model=InsightReflectionResponse, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def reflect_on_experiment(
    experiment_id: str,
    database: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[User, Depends(get_current_user)],
) -> InsightReflectionResponse:
    return create_reflection(experiment_id, user.id, database, settings, (
        GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds),
        OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds),
    ))

@router.post("/{experiment_id}/jobs/{job_type}", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(_require_owned_path), Depends(limit("write"))])
def run_background_stage(experiment_id: str, job_type: str, database: Annotated[Session, Depends(get_db)]) -> JobResponse:
    return create_job(experiment_id, job_type, database)
