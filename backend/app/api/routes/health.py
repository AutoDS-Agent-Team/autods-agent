from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.dependencies import get_db
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", service="autods-backend")


@router.get("/health/ready", response_model=HealthResponse)
def readiness_check(database: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        database.execute(text("SELECT 1"))
        from redis import Redis

        redis = Redis.from_url(settings.redis_url)
        redis.ping()
        redis.close()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Backend dependencies are not ready.") from error
    return HealthResponse(status="healthy", service="autods-backend")

