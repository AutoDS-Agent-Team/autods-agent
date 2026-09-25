from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.rate_limit import limit
from app.db.dependencies import get_db
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.models.user import User
from app.schemas.assistant import AssistantQueryRequest, AssistantQueryResponse
from app.services.assistant_orchestration_service import answer_context_aware_question
from app.services.assistant_service import answer_question
from app.services.security import get_current_user


router = APIRouter(prefix="/assistant", tags=["assistant"])


def _available_providers(settings: Settings):
    """Keep Gemini → Ollama explicit while tests remain fully offline."""
    if settings.app_env == "test":
        return []
    providers = []
    if settings.gemini_api_key:
        providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds))
    if settings.ollama_base_url:
        providers.append(OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds))
    return providers


@router.post("/query", response_model=AssistantQueryResponse, dependencies=[Depends(limit("write"))])
def query(
    request: AssistantQueryRequest,
    database: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AssistantQueryResponse:
    return answer_context_aware_question(request, database, user, settings, _available_providers(settings), answer_question)
